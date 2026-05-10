//! RMVPE — Robust Model for Vocal Pitch Estimation (Wei et al., Interspeech
//! 2023, arXiv 2306.15412). Vocal-specific encoder originally trained for
//! singing-through-accompaniment; de-facto F0 backend in the RVC ecosystem.
//!
//! ## Model file
//!
//! pitch-core ships no weights. Use the public ONNX mirror at
//! `lj1995/VoiceConversionWebUI` on HuggingFace
//! (~362 MB; Apache-2.0 code, MIT mirror). Loaded via [`Session::commit_from_file`]
//! at construction time.
//!
//! ## I/O contract (matches Python `pitch_contour.py::_rmvpe`)
//!
//!   • input  :  log-mel `[1, 128, T]`, T padded to multiple of 32.
//!   • output :  salience `[1, T, 360]` — same 20-cent grid as CREPE.
//!
//! Mel preprocessing matches RVC's `infer/lib/rmvpe.py`:
//!   • librosa.stft, n_fft=1024, hop=160 (10 ms @ 16k), win=1024, Hann,
//!     **center=False** (streaming-friendly; introduces a constant
//!     32 ms time offset vs. the offline Python pipeline — pitch values
//!     are identical, only frame timestamps shift).
//!   • HTK mel filterbank, n_mels=128, fmin=30 Hz, fmax=8000 Hz.
//!   • magnitude (sqrt of power, NOT squared).
//!   • log(max(mel, 1e-5)).
//!
//! Decoder: same weighted-mean ±4-bin window as CREPE.
//!
//! ## Streaming
//!
//! Buffers audio at 16 kHz, runs inference once accumulated mel frames
//! reach `MIN_FRESH_FRAMES + CONTEXT_FRAMES`, emits only frames that
//! correspond to fresh (non-context) audio.

use pitch_core::{calibrate_confidence, EstimatorError, PitchEstimator, PitchFrame, Result};

/// Vocadito GT calibration: T=0.71. F1 0.938 → 0.965. See ANALYSIS.md.
const CALIBRATION_T: Option<f32> = Some(0.71);
use ndarray::Array3;
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Value;
use realfft::num_complex::Complex32;
use realfft::{RealFftPlanner, RealToComplex};
use std::sync::Arc;

const SR: u32 = 16_000;
const N_FFT: usize = 1024;
const HOP: usize = 160;            // 10 ms @ 16k
const N_MELS: usize = 128;
const FMIN_MEL_HZ: f32 = 30.0;
const FMAX_MEL_HZ: f32 = 8000.0;
const PAD_MULT: usize = 32;
const N_BINS: usize = 360;
const CENTS_PER_BIN: f32 = 20.0;
/// Same offset as CREPE: bin 0 → 1997.379 cents above 10 Hz.
const CENTS_OFFSET: f32 = 1997.379_4;
const DECODER_HALF_WINDOW: usize = 4;
const VOICED_THRESHOLD: f32 = 0.03; // RMVPE-specific (much lower than CREPE/SwiftF0)

/// How many mel frames of left-context to keep between calls. Roughly the
/// model's effective receptive field — 32 = ~320 ms.
const CONTEXT_FRAMES: usize = 32;
/// Run inference once we have at least this many fresh mel frames in addition
/// to the context. Smaller → lower latency, more inferences. 32 = ~320 ms.
const MIN_FRESH_FRAMES: usize = 32;

pub struct RmvpeEstimator {
    session: Session,
    in_name: String,
    out_name: String,
    fft: Arc<dyn RealToComplex<f32>>,
    window: Vec<f32>,                 // Hann, length N_FFT
    mel_basis: Vec<f32>,              // [N_MELS][N_FFT/2+1], row-major
    // Audio side
    audio_buffer: Vec<f32>,
    context_audio_held: usize,        // tail of audio_buffer that's "context", not yet-to-emit
    // Frame counter (in mel frames; ≡ output pitch frames)
    next_frame_index: u64,
    // Reusable scratch
    fft_input: Vec<f32>,
    fft_output: Vec<Complex32>,
    fft_scratch: Vec<Complex32>,
}

fn hann(n: usize) -> Vec<f32> {
    // Periodic (sym=False) Hann, matching librosa's STFT default.
    if n == 0 {
        return Vec::new();
    }
    (0..n)
        .map(|i| 0.5 - 0.5 * (std::f32::consts::TAU * i as f32 / n as f32).cos())
        .collect()
}

#[inline]
fn hz_to_mel_htk(hz: f32) -> f32 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

#[inline]
fn mel_to_hz_htk(mel: f32) -> f32 {
    700.0 * (10f32.powf(mel / 2595.0) - 1.0)
}

/// Build the HTK mel filterbank as a flat `[n_mels][n_fft/2+1]` matrix.
fn build_mel_basis() -> Vec<f32> {
    let n_freqs = N_FFT / 2 + 1;
    let mut basis = vec![0.0_f32; N_MELS * n_freqs];

    // Mel-frequency edges of the n_mels triangular filters.
    let mel_min = hz_to_mel_htk(FMIN_MEL_HZ);
    let mel_max = hz_to_mel_htk(FMAX_MEL_HZ);
    let mut mel_edges = Vec::with_capacity(N_MELS + 2);
    for i in 0..(N_MELS + 2) {
        let m = mel_min + (mel_max - mel_min) * i as f32 / (N_MELS + 1) as f32;
        mel_edges.push(mel_to_hz_htk(m));
    }
    // FFT bin frequencies.
    let bin_hz: Vec<f32> = (0..n_freqs)
        .map(|k| k as f32 * SR as f32 / N_FFT as f32)
        .collect();
    // Triangular filters.
    for m in 0..N_MELS {
        let lo = mel_edges[m];
        let mid = mel_edges[m + 1];
        let hi = mel_edges[m + 2];
        for k in 0..n_freqs {
            let f = bin_hz[k];
            let w = if f <= lo || f >= hi {
                0.0
            } else if f <= mid {
                (f - lo) / (mid - lo).max(1e-12)
            } else {
                (hi - f) / (hi - mid).max(1e-12)
            };
            basis[m * n_freqs + k] = w;
        }
        // Slaney-style normalization is NOT applied here (htk=True in
        // librosa.filters.mel skips the area normalization).
    }
    basis
}

impl RmvpeEstimator {
    pub fn new(model_path: &str) -> Result<Self> {
        let session = Session::builder()
            .map_err(|e| EstimatorError::Ort(e.to_string()))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| EstimatorError::Ort(e.to_string()))?
            .commit_from_file(model_path)
            .map_err(|e| EstimatorError::Ort(format!("load {model_path}: {e}")))?;

        // Different RMVPE exports use different I/O names ("input"/"output",
        // "mel"/"salience", "data"/"pitch", etc.) — query the model once.
        let in_name = session
            .inputs()
            .first()
            .ok_or_else(|| EstimatorError::Ort("rmvpe: model has no inputs".into()))?
            .name()
            .to_string();
        let out_name = session
            .outputs()
            .first()
            .ok_or_else(|| EstimatorError::Ort("rmvpe: model has no outputs".into()))?
            .name()
            .to_string();

        let mut planner = RealFftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(N_FFT);
        let fft_input = vec![0.0_f32; N_FFT];
        let fft_output = vec![Complex32::default(); N_FFT / 2 + 1];
        let fft_scratch = vec![Complex32::default(); fft.get_scratch_len()];

        Ok(Self {
            session,
            in_name,
            out_name,
            fft,
            window: hann(N_FFT),
            mel_basis: build_mel_basis(),
            audio_buffer: Vec::with_capacity(SR as usize),
            context_audio_held: 0,
            next_frame_index: 0,
            fft_input,
            fft_output,
            fft_scratch,
        })
    }

    /// Compute log-mel of one frame starting at `audio[start..start+N_FFT]`.
    /// Returns one column of length N_MELS appended to `out`.
    fn compute_log_mel_frame(&mut self, audio: &[f32], start: usize, out: &mut Vec<f32>) -> Result<()> {
        // Window into FFT input buffer.
        for i in 0..N_FFT {
            self.fft_input[i] = audio[start + i] * self.window[i];
        }
        self.fft
            .process_with_scratch(&mut self.fft_input, &mut self.fft_output, &mut self.fft_scratch)
            .map_err(|e| EstimatorError::Ort(format!("rmvpe stft: {e}")))?;

        // Magnitude (NOT squared).
        let n_freqs = N_FFT / 2 + 1;
        let mut mag = vec![0.0_f32; n_freqs];
        for k in 0..n_freqs {
            let c = self.fft_output[k];
            mag[k] = (c.re * c.re + c.im * c.im).sqrt();
        }

        // Mel projection + log-clamp.
        for m in 0..N_MELS {
            let basis_row = &self.mel_basis[m * n_freqs..(m + 1) * n_freqs];
            let mut acc = 0.0_f32;
            for k in 0..n_freqs {
                acc += basis_row[k] * mag[k];
            }
            out.push(acc.max(1e-5).ln());
        }
        Ok(())
    }
}

impl PitchEstimator for RmvpeEstimator {
    fn name(&self) -> &str {
        "rmvpe"
    }

    fn target_sample_rate(&self) -> u32 {
        SR
    }

    fn reset(&mut self) {
        self.audio_buffer.clear();
        self.context_audio_held = 0;
        self.next_frame_index = 0;
    }

    fn process(&mut self, audio: &[f32]) -> Result<Vec<PitchFrame>> {
        self.audio_buffer.extend_from_slice(audio);

        // How many complete frames fit in current audio_buffer?
        // Frame i uses audio_buffer[i*HOP .. i*HOP + N_FFT].
        let n_frames_total = if self.audio_buffer.len() >= N_FFT {
            (self.audio_buffer.len() - N_FFT) / HOP + 1
        } else {
            0
        };
        // Of those, context_audio_held / HOP have already been emitted as
        // context. fresh = total - context.
        let context_frames = self.context_audio_held / HOP;
        if n_frames_total < context_frames + MIN_FRESH_FRAMES {
            return Ok(Vec::new());
        }
        let fresh_frames = n_frames_total - context_frames;

        // Compute log-mel for ALL frames in the current buffer.
        let mut mel_flat: Vec<f32> = Vec::with_capacity(N_MELS * n_frames_total);
        // Transpose-friendly layout for the model: [n_mels, T].
        // We compute frame-by-frame and rearrange at the end.
        let mut per_frame: Vec<Vec<f32>> = Vec::with_capacity(n_frames_total);
        for f in 0..n_frames_total {
            let start = f * HOP;
            let mut col = Vec::with_capacity(N_MELS);
            self.compute_log_mel_frame(&self.audio_buffer.clone(), start, &mut col)?;
            per_frame.push(col);
        }
        // Pad to multiple of PAD_MULT with zeros (matches Python np.pad).
        let pad = (PAD_MULT - n_frames_total % PAD_MULT) % PAD_MULT;
        let t_padded = n_frames_total + pad;
        for _ in 0..pad {
            per_frame.push(vec![0.0; N_MELS]);
        }
        // Rearrange to [n_mels, T_padded] row-major.
        for m in 0..N_MELS {
            for t in 0..t_padded {
                mel_flat.push(per_frame[t][m]);
            }
        }

        // Run inference.
        let arr = Array3::from_shape_vec((1, N_MELS, t_padded), mel_flat)
            .map_err(|e| EstimatorError::Ort(format!("rmvpe shape: {e}")))?;
        let input = Value::from_array(arr)
            .map_err(|e| EstimatorError::Ort(format!("rmvpe value: {e}")))?;
        let outputs = self
            .session
            .run(ort::inputs![self.in_name.as_str() => input])
            .map_err(|e| EstimatorError::Ort(format!("rmvpe run: {e}")))?;
        let (_shape, salience): (_, &[f32]) = outputs[self.out_name.as_str()]
            .try_extract_tensor::<f32>()
            .map_err(|e| EstimatorError::Ort(format!("rmvpe extract: {e}")))?;

        // Decode: weighted-mean over ±4 bins, peak as confidence.
        // Time labelling: STFT here uses center=False (frame i covers
        // samples [i*HOP, i*HOP+N_FFT)), so the AUDIO CONTENT for frame i
        // is centered at sample (i*HOP + N_FFT/2). We label time_s with
        // the content time so pairwise comparisons against center=True
        // backends (PYin, FCPE) line up on the common time grid.
        let hop_s = HOP as f32 / SR as f32;
        let center_offset_s = (N_FFT as f32 / 2.0) / SR as f32;
        let mut out = Vec::with_capacity(fresh_frames);
        for f in context_frames..n_frames_total {
            let row = &salience[f * N_BINS..(f + 1) * N_BINS];
            let (peak, peak_p) = row
                .iter()
                .copied()
                .enumerate()
                .fold((0usize, f32::NEG_INFINITY), |acc, (j, p)| {
                    if p > acc.1 { (j, p) } else { acc }
                });
            let lo = peak.saturating_sub(DECODER_HALF_WINDOW);
            let hi = (peak + DECODER_HALF_WINDOW).min(N_BINS - 1);
            let mut num = 0.0_f32;
            let mut den = 0.0_f32;
            for j in lo..=hi {
                let p = row[j].max(0.0);
                let cents = j as f32 * CENTS_PER_BIN + CENTS_OFFSET;
                num += p * cents;
                den += p;
            }
            let cents = if den > 0.0 {
                num / den
            } else {
                peak as f32 * CENTS_PER_BIN + CENTS_OFFSET
            };
            let pitch_hz = if peak_p >= VOICED_THRESHOLD {
                10.0_f32 * 2.0_f32.powf(cents / 1200.0)
            } else {
                0.0
            };

            let abs_idx = self.next_frame_index + (f - context_frames) as u64;
            out.push(PitchFrame {
                frame_index: abs_idx,
                time_s: abs_idx as f32 * hop_s + center_offset_s,
                pitch_hz,
                confidence: calibrate_confidence(CALIBRATION_T, peak_p),
                is_preliminary: false,
            });
        }
        self.next_frame_index += fresh_frames as u64;

        // Retain `CONTEXT_FRAMES * HOP + (N_FFT - HOP)` audio samples for next
        // call: enough to keep the next call's frames consistent with this
        // call's frames.
        let keep_audio = CONTEXT_FRAMES * HOP + (N_FFT - HOP);
        if self.audio_buffer.len() > keep_audio {
            let drop_n = self.audio_buffer.len() - keep_audio;
            self.audio_buffer.drain(..drop_n);
        }
        self.context_audio_held = CONTEXT_FRAMES * HOP;

        Ok(out)
    }
}
