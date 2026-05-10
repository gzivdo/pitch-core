//! Cross-algorithm benchmark for pitch-core + pitch-core-onnx.
//!
//! Streams every `*.wav` under <songs_dir> through each available backend
//! in 1024-sample chunks (mimicking a realtime audio callback) and prints
//! a markdown report with throughput, voicing, confidence, octave-stability
//! and pairwise agreement.
//!
//! ```sh
//! cargo run --example bench --release -- <songs_dir> [models_dir] \
//!     [--out report.md] [--algos LIST|all]
//! ```
//!
//! `models_dir` defaults to `./models`. Auto-detected ONNX files:
//!   `swift_f0.onnx`, `crepe-tiny.onnx`, `rmvpe.onnx`, `fcpe.onnx`,
//!   `pesto_mir1k_g7_48k_960.onnx` (only with `--features pesto`).
//!
//! Backends with missing models are silently skipped.

use std::collections::BTreeMap;
use std::env;
use std::error::Error;
use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

use hound::WavReader;
use pitch_core::{
    EstimatorError, PitchEstimator, PitchFrame, PitchTracker, PraatAcEstimator, PyinEstimator,
    SwipeEstimator,
};
use pitch_core_onnx::{CrepeEstimator, FcpeEstimator, Mode, RmvpeEstimator, SwiftF0Estimator};

const VOICED_THRESHOLD: f32 = 0.3;
const OCTAVE_JUMP_RATIO: f32 = 1.7;
const CHUNK_SAMPLES: usize = 1024;
const AGREEMENT_TOL_CENTS: f32 = 50.0;
const COMMON_HOP_MS: f32 = 20.0;

type EstimatorResult = Result<Box<dyn PitchEstimator>, EstimatorError>;
type Builder = Box<dyn Fn() -> EstimatorResult>;

struct AlgoSpec {
    name: &'static str,
    available: bool,
    skip_reason: Option<String>,
    build: Builder,
}

#[derive(Default, Clone, Debug)]
struct Stats {
    audio_secs: f32,
    process_secs: f32,
    latency_first_frame_ms: f32,
    n_frames: usize,
    n_voiced: usize,
    sum_conf_voiced: f64,
    n_octave_jumps: usize,
    n_voiced_pairs: usize,
    common_grid: Vec<Option<f32>>,
}

impl Stats {
    fn realtime_factor(&self) -> f32 {
        if self.process_secs > 0.0 {
            self.audio_secs / self.process_secs
        } else {
            f32::INFINITY
        }
    }
    fn voicing_rate(&self) -> f32 {
        if self.n_frames > 0 {
            self.n_voiced as f32 / self.n_frames as f32
        } else {
            0.0
        }
    }
    fn mean_conf_voiced(&self) -> f32 {
        if self.n_voiced > 0 {
            (self.sum_conf_voiced / self.n_voiced as f64) as f32
        } else {
            0.0
        }
    }
    fn octave_jump_rate(&self) -> f32 {
        if self.n_voiced_pairs > 0 {
            self.n_octave_jumps as f32 / self.n_voiced_pairs as f32
        } else {
            0.0
        }
    }
}

fn load_wav_mono_f32(path: &Path) -> Result<(Vec<f32>, u32), Box<dyn Error>> {
    let mut reader = WavReader::open(path)?;
    let spec = reader.spec();
    let n_ch = spec.channels as usize;
    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader
            .samples::<f32>()
            .collect::<Result<Vec<_>, _>>()?,
        hound::SampleFormat::Int => {
            let max = ((1i64 << (spec.bits_per_sample - 1)) - 1) as f32;
            reader
                .samples::<i32>()
                .map(|s| s.map(|x| x as f32 / max))
                .collect::<Result<Vec<_>, _>>()?
        }
    };
    let mono: Vec<f32> = if n_ch == 1 {
        samples
    } else {
        samples
            .chunks_exact(n_ch)
            .map(|c| c.iter().sum::<f32>() / n_ch as f32)
            .collect()
    };
    Ok((mono, spec.sample_rate))
}

fn run_algo(spec: &AlgoSpec, audio: &[f32], in_sr: u32) -> Result<Stats, Box<dyn Error>> {
    let est = (spec.build)()?;
    let mut tracker = PitchTracker::from_boxed(est, in_sr, CHUNK_SAMPLES)?;
    let audio_secs = audio.len() as f32 / in_sr as f32;
    let mut s = Stats {
        audio_secs,
        ..Default::default()
    };
    let mut prev_voiced_pitch: Option<f32> = None;
    let mut got_first = false;
    let mut samples_consumed = 0usize;

    let common_grid_len = (audio_secs * 1000.0 / COMMON_HOP_MS).ceil() as usize;
    s.common_grid = vec![None; common_grid_len];

    let process_start = Instant::now();
    for chunk in audio.chunks(CHUNK_SAMPLES) {
        let frames: Vec<PitchFrame> = tracker.process(chunk)?;
        samples_consumed += chunk.len();
        let consumed_ms = samples_consumed as f32 * 1000.0 / in_sr as f32;
        for f in &frames {
            if !got_first {
                got_first = true;
                s.latency_first_frame_ms = consumed_ms;
            }
            s.n_frames += 1;
            let voiced = f.confidence >= VOICED_THRESHOLD;
            let grid_idx = (f.time_s * 1000.0 / COMMON_HOP_MS) as usize;
            if grid_idx < s.common_grid.len() {
                s.common_grid[grid_idx] = if voiced && f.pitch_hz > 0.0 {
                    Some(f.pitch_hz)
                } else {
                    None
                };
            }
            if voiced {
                s.n_voiced += 1;
                s.sum_conf_voiced += f.confidence as f64;
                if let Some(prev) = prev_voiced_pitch {
                    if prev > 0.0 && f.pitch_hz > 0.0 {
                        let ratio = f.pitch_hz / prev;
                        if ratio >= OCTAVE_JUMP_RATIO || ratio <= 1.0 / OCTAVE_JUMP_RATIO {
                            s.n_octave_jumps += 1;
                        }
                        s.n_voiced_pairs += 1;
                    }
                }
                prev_voiced_pitch = Some(f.pitch_hz);
            } else {
                prev_voiced_pitch = None;
            }
        }
    }
    s.process_secs = process_start.elapsed().as_secs_f32();
    Ok(s)
}

fn pairwise_agreement(a: &[Option<f32>], b: &[Option<f32>]) -> (f32, usize) {
    let n = a.len().min(b.len());
    let mut both_voiced = 0usize;
    let mut agree = 0usize;
    for i in 0..n {
        if let (Some(pa), Some(pb)) = (a[i], b[i]) {
            both_voiced += 1;
            let cents = 1200.0 * (pa / pb).abs().log2();
            if cents.abs() <= AGREEMENT_TOL_CENTS {
                agree += 1;
            }
        }
    }
    let rate = if both_voiced > 0 {
        agree as f32 / both_voiced as f32
    } else {
        0.0
    };
    (rate, both_voiced)
}

fn build_algos(models_dir: &Path) -> Vec<AlgoSpec> {
    let mut algos: Vec<AlgoSpec> = Vec::new();

    // Pure-DSP: always available.
    algos.push(AlgoSpec {
        name: "swipe-balanced",
        available: true,
        skip_reason: None,
        build: Box::new(|| Ok(Box::new(SwipeEstimator::with_max_window(8192)?))),
    });
    algos.push(AlgoSpec {
        name: "swipe-realtime",
        available: true,
        skip_reason: None,
        build: Box::new(|| Ok(Box::new(SwipeEstimator::with_max_window(4096)?))),
    });
    algos.push(AlgoSpec {
        name: "pyin",
        available: true,
        skip_reason: None,
        build: Box::new(|| Ok(Box::new(PyinEstimator::new()?))),
    });
    algos.push(AlgoSpec {
        name: "praat_ac",
        available: true,
        skip_reason: None,
        build: Box::new(|| Ok(Box::new(PraatAcEstimator::new(false)?))),
    });

    // ONNX backends: gate on model file presence.
    let onnx_specs: &[(
        &'static str,
        &'static str,
        fn(&str) -> EstimatorResult,
    )] = &[
        (
            "swiftf0",
            "swift_f0.onnx",
            |p| Ok(Box::new(SwiftF0Estimator::new(p, Mode::Balanced)?)),
        ),
        (
            "crepe-tiny",
            "crepe-tiny.onnx",
            |p| Ok(Box::new(CrepeEstimator::new(p)?)),
        ),
        (
            "rmvpe",
            "rmvpe.onnx",
            |p| Ok(Box::new(RmvpeEstimator::new(p)?)),
        ),
        (
            "fcpe",
            "fcpe.onnx",
            |p| Ok(Box::new(FcpeEstimator::new(p)?)),
        ),
    ];
    for (name, fname, ctor) in onnx_specs {
        let path = models_dir.join(fname);
        if path.exists() {
            let p = path.to_string_lossy().into_owned();
            let f = *ctor;
            algos.push(AlgoSpec {
                name,
                available: true,
                skip_reason: None,
                build: Box::new(move || f(&p)),
            });
        } else {
            algos.push(AlgoSpec {
                name,
                available: false,
                skip_reason: Some(format!("missing {}", path.display())),
                build: Box::new(|| {
                    Err(EstimatorError::InvalidInput("backend filtered".into()))
                }),
            });
        }
    }

    #[cfg(feature = "pesto")]
    {
        let path = models_dir.join("pesto_mir1k_g7_48k_960.onnx");
        if path.exists() {
            let p = path.to_string_lossy().into_owned();
            algos.push(AlgoSpec {
                name: "pesto",
                available: true,
                skip_reason: None,
                build: Box::new(move || {
                    Ok(Box::new(pitch_core_onnx::PestoEstimator::new(&p)?))
                }),
            });
        } else {
            algos.push(AlgoSpec {
                name: "pesto",
                available: false,
                skip_reason: Some(format!("missing {}", path.display())),
                build: Box::new(|| {
                    Err(EstimatorError::InvalidInput("backend filtered".into()))
                }),
            });
        }
    }

    algos
}

fn find_wavs(dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        let Ok(rd) = fs::read_dir(&d) else { continue };
        for ent in rd.flatten() {
            let p = ent.path();
            if p.is_dir() {
                stack.push(p);
            } else if p
                .extension()
                .and_then(|e| e.to_str())
                .map(|e| e.eq_ignore_ascii_case("wav"))
                .unwrap_or(false)
            {
                out.push(p);
            }
        }
    }
    out.sort();
    out
}

fn fmt_pct(x: f32) -> String {
    format!("{:>5.1}%", x * 100.0)
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!(
            "usage: {} <songs_dir> [models_dir] [--out report.md] [--algos LIST|all]\n  \
             models_dir defaults to ./models",
            args[0]
        );
        std::process::exit(1);
    }
    let songs_dir = PathBuf::from(&args[1]);
    let mut models_dir = PathBuf::from("models");
    let mut out_path: Option<PathBuf> = None;
    let mut algo_filter: Option<Vec<String>> = None;
    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--out" => {
                out_path = Some(PathBuf::from(&args[i + 1]));
                i += 2;
            }
            "--algos" => {
                let v = &args[i + 1];
                if v == "all" {
                    algo_filter = None;
                } else {
                    algo_filter = Some(v.split(',').map(|s| s.trim().to_string()).collect());
                }
                i += 2;
            }
            other => {
                models_dir = PathBuf::from(other);
                i += 1;
            }
        }
    }

    let wavs = find_wavs(&songs_dir);
    if wavs.is_empty() {
        eprintln!("no .wav files under {}", songs_dir.display());
        std::process::exit(1);
    }

    let mut algos = build_algos(&models_dir);
    if let Some(want) = &algo_filter {
        for a in algos.iter_mut() {
            if !want.iter().any(|w| w == a.name) {
                a.available = false;
                a.skip_reason = Some("filtered by --algos".into());
            }
        }
    }
    let active_names: Vec<&str> = algos.iter().filter(|a| a.available).map(|a| a.name).collect();
    if active_names.is_empty() {
        eprintln!("no algorithms enabled");
        std::process::exit(1);
    }

    let mut report = String::new();
    use std::fmt::Write as _;
    writeln!(report, "# pitch-core benchmark\n")?;
    writeln!(report, "Songs: **{}**\n", wavs.len())?;

    let skipped: Vec<&AlgoSpec> = algos.iter().filter(|a| !a.available).collect();
    if !skipped.is_empty() {
        writeln!(report, "_Skipped backends:_")?;
        for s in &skipped {
            writeln!(
                report,
                "- `{}` — {}",
                s.name,
                s.skip_reason.as_deref().unwrap_or("?")
            )?;
        }
        writeln!(report)?;
    }

    let mut agg: BTreeMap<&str, Stats> = BTreeMap::new();
    let mut all_grids: BTreeMap<String, BTreeMap<&str, Vec<Option<f32>>>> = BTreeMap::new();

    for wav in &wavs {
        let name = wav.file_name().unwrap().to_string_lossy().into_owned();
        let (audio, in_sr) = match load_wav_mono_f32(wav) {
            Ok(x) => x,
            Err(e) => {
                eprintln!("skip {}: {}", wav.display(), e);
                continue;
            }
        };
        let dur = audio.len() as f32 / in_sr as f32;
        eprintln!("[{}]  {:.1}s @ {} Hz", name, dur, in_sr);

        writeln!(report, "## `{}`", name)?;
        writeln!(report, "- duration: {:.1} s, input rate: {} Hz", dur, in_sr)?;
        writeln!(report)?;
        writeln!(
            report,
            "| backend | × realtime | first frame | voicing | mean conf | oct-jump | frames |"
        )?;
        writeln!(report, "|---|---:|---:|---:|---:|---:|---:|")?;

        let mut song_grids: BTreeMap<&str, Vec<Option<f32>>> = BTreeMap::new();
        for spec in &algos {
            if !spec.available {
                continue;
            }
            eprint!("  {:14} ... ", spec.name);
            let stats = match run_algo(spec, &audio, in_sr) {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("ERR {}", e);
                    writeln!(report, "| `{}` | ERR | — | — | — | — | — |", spec.name)?;
                    continue;
                }
            };
            eprintln!(
                "{:>5.1}× rt   voicing={}  jumps={}",
                stats.realtime_factor(),
                fmt_pct(stats.voicing_rate()),
                fmt_pct(stats.octave_jump_rate())
            );
            writeln!(
                report,
                "| `{}` | {:>5.1}× | {:>4.0} ms | {} | {:>5.2} | {} | {} |",
                spec.name,
                stats.realtime_factor(),
                stats.latency_first_frame_ms,
                fmt_pct(stats.voicing_rate()),
                stats.mean_conf_voiced(),
                fmt_pct(stats.octave_jump_rate()),
                stats.n_frames
            )?;
            let entry = agg.entry(spec.name).or_default();
            entry.audio_secs += stats.audio_secs;
            entry.process_secs += stats.process_secs;
            entry.latency_first_frame_ms = entry
                .latency_first_frame_ms
                .max(stats.latency_first_frame_ms);
            entry.n_frames += stats.n_frames;
            entry.n_voiced += stats.n_voiced;
            entry.sum_conf_voiced += stats.sum_conf_voiced;
            entry.n_octave_jumps += stats.n_octave_jumps;
            entry.n_voiced_pairs += stats.n_voiced_pairs;
            song_grids.insert(spec.name, stats.common_grid);
        }
        writeln!(report)?;
        all_grids.insert(name, song_grids);
    }

    writeln!(
        report,
        "## Aggregate ({} song{})\n",
        wavs.len(),
        if wavs.len() == 1 { "" } else { "s" }
    )?;
    writeln!(
        report,
        "| backend | × realtime | voicing | mean conf | oct-jump | frames |"
    )?;
    writeln!(report, "|---|---:|---:|---:|---:|---:|")?;
    for name in &active_names {
        let s = match agg.get(name) {
            Some(s) => s,
            None => continue,
        };
        writeln!(
            report,
            "| `{}` | {:>5.1}× | {} | {:>5.2} | {} | {} |",
            name,
            s.realtime_factor(),
            fmt_pct(s.voicing_rate()),
            s.mean_conf_voiced(),
            fmt_pct(s.octave_jump_rate()),
            s.n_frames
        )?;
    }
    writeln!(report)?;

    writeln!(
        report,
        "## Pairwise agreement (% frames within ±{} cents, both voiced)\n",
        AGREEMENT_TOL_CENTS as i32
    )?;
    write!(report, "| | ")?;
    for n in &active_names {
        write!(report, "{} | ", n)?;
    }
    writeln!(report)?;
    write!(report, "|---|")?;
    for _ in &active_names {
        write!(report, "---:|")?;
    }
    writeln!(report)?;

    for a in &active_names {
        write!(report, "| **{}** |", a)?;
        for b in &active_names {
            if a == b {
                write!(report, " — |")?;
                continue;
            }
            let mut total_agree = 0usize;
            let mut total_pairs = 0usize;
            for grids in all_grids.values() {
                if let (Some(ga), Some(gb)) = (grids.get(a), grids.get(b)) {
                    let (rate, n) = pairwise_agreement(ga, gb);
                    total_agree += (rate * n as f32).round() as usize;
                    total_pairs += n;
                }
            }
            if total_pairs > 0 {
                let rate = total_agree as f32 / total_pairs as f32;
                write!(report, " {} |", fmt_pct(rate))?;
            } else {
                write!(report, " — |")?;
            }
        }
        writeln!(report)?;
    }
    writeln!(report)?;
    writeln!(report, "---")?;
    writeln!(
        report,
        "_Methodology: streamed in {}-sample chunks at the file's native rate; \
         voiced = confidence ≥ {}; octave-jump = consecutive voiced frames whose \
         pitch ratio is ≥ {} or ≤ {:.4}; pairwise agreement projects each backend \
         onto a common {:.0} ms grid and counts cells where both are voiced \
         within ±{} cents._",
        CHUNK_SAMPLES,
        VOICED_THRESHOLD,
        OCTAVE_JUMP_RATIO,
        1.0 / OCTAVE_JUMP_RATIO,
        COMMON_HOP_MS,
        AGREEMENT_TOL_CENTS as i32
    )?;
    writeln!(report)?;

    println!("{}", report);
    if let Some(p) = out_path {
        let mut f = File::create(&p)?;
        f.write_all(report.as_bytes())?;
        eprintln!("wrote {}", p.display());
    }
    Ok(())
}
