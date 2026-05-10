//! Minimal pitch-core example: load a WAV, run SWIPE, print voiced frames.
//!
//! Pure-DSP — no model files, no `ort` / ONNX Runtime, ~2 MB binary.
//!
//! Run:
//!     cargo run --example quickstart --release -- path/to/voice.wav
//!
//! For neural backends (CREPE/RMVPE/SwiftF0/FCPE) see the
//! `pitch-core-onnx` crate's `bench` example.

use hound::WavReader;
use pitch_core::{PitchTracker, SwipeEstimator};
use std::env;
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    let path = env::args()
        .nth(1)
        .ok_or("usage: quickstart <wav-file>")?;

    let mut reader = WavReader::open(&path)?;
    let spec = reader.spec();
    let sr = spec.sample_rate;
    let n_ch = spec.channels as usize;

    // Decode to f32 mono.
    let raw: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader
            .samples::<f32>()
            .collect::<Result<_, _>>()?,
        hound::SampleFormat::Int => {
            let max = ((1i64 << (spec.bits_per_sample - 1)) - 1) as f32;
            reader
                .samples::<i32>()
                .map(|s| s.map(|x| x as f32 / max))
                .collect::<Result<_, _>>()?
        }
    };
    let mono: Vec<f32> = if n_ch == 1 {
        raw
    } else {
        raw.chunks_exact(n_ch)
            .map(|c| c.iter().sum::<f32>() / n_ch as f32)
            .collect()
    };

    println!(
        "loaded {}: {:.2}s @ {} Hz",
        path,
        mono.len() as f32 / sr as f32,
        sr,
    );

    // Default SWIPE = Balanced preset (8192 max_window, ~170 ms latency,
    // 96.4% RPA on MIR-1K). For lower latency: SwipeEstimator::with_max_window(4096).
    let est = SwipeEstimator::new()?;
    let mut tracker = PitchTracker::new(est, sr, 1024)?;

    let frames = tracker.process(&mono)?;
    let voiced: Vec<_> = frames.iter().filter(|f| f.confidence >= 0.3).collect();

    println!(
        "{} frames, {} voiced ({:.1}%)",
        frames.len(),
        voiced.len(),
        100.0 * voiced.len() as f32 / frames.len().max(1) as f32,
    );

    println!("\nfirst 10 voiced frames:");
    for f in voiced.iter().take(10) {
        println!(
            "  {:6.3}s  {:6.1} Hz  conf={:.2}",
            f.time_s, f.pitch_hz, f.confidence,
        );
    }

    Ok(())
}
