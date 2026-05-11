//! pYIN (probabilistic YIN, Mauch & Dixon 2014) via the `pyin` crate (Sytronik,
//! pure-Rust port of sevagh/pitch-detection).
//!
//! pYIN is a probabilistic extension of YIN with HMM/Viterbi smoothing across
//! frames — gold-standard among classical pitch trackers, often used as a
//! reference for evaluating neural methods. It's BATCH-only (Viterbi needs the
//! whole sequence). We process in moderate-sized batches (~333 ms) so Viterbi
//! sees enough context within each batch and end-to-end latency stays bounded.
//!
//! NB: pyin operates on f64 internally. We convert from f32 on input.
//!
//! Default parameters:
//!   - fmin = 65 Hz, fmax = 1100 Hz   (vocal range, matches praat_ac)
//!   - frame_length = 2048             (~43 ms @ 48k)
//!   - hop_length   = 960              (20 ms @ 48k, matches PESTO)
//!   - batch        = 16000 samples    (~333 ms = 16 hops)

use crate::estimator::{PitchEstimator, PitchFrame, Result};
use pyin::{Framing, PYINExecutor, PadMode};

const PYIN_SR: u32 = 48000;
const FRAME_LENGTH: usize = 2048;
const HOP_LENGTH: usize = 960;
const FMIN: f64 = 65.0;
const FMAX: f64 = 1100.0;
const BATCH_SAMPLES: usize = 16000; // 333 ms — Viterbi context within each batch

pub struct PyinEstimator {
    executor: PYINExecutor<f64>,
    buffer: Vec<f64>,
    samples_processed: u64,
}

fn make_exec() -> PYINExecutor<f64> {
    PYINExecutor::<f64>::new(
        FMIN,
        FMAX,
        PYIN_SR,
        FRAME_LENGTH,
        None,             // win_length: defaults to frame_length
        Some(HOP_LENGTH), // hop_length
        None,             // resolution: default
    )
}

impl PyinEstimator {
    pub fn new() -> Result<Self> {
        Ok(Self {
            executor: make_exec(),
            buffer: Vec::with_capacity(BATCH_SAMPLES * 2),
            samples_processed: 0,
        })
    }
}

impl PitchEstimator for PyinEstimator {
    fn name(&self) -> &str {
        "pyin"
    }

    fn target_sample_rate(&self) -> u32 {
        PYIN_SR
    }

    fn reset(&mut self) {
        self.buffer.clear();
        self.samples_processed = 0;
        self.executor = make_exec();
    }

    fn process(&mut self, audio_target_sr: &[f32]) -> Result<Vec<PitchFrame>> {
        // f32 → f64
        self.buffer.extend(audio_target_sr.iter().map(|&s| s as f64));

        let hop_s = HOP_LENGTH as f32 / PYIN_SR as f32;
        let mut frames = Vec::new();

        while self.buffer.len() >= BATCH_SAMPLES {
            let chunk: Vec<f64> = self.buffer.drain(..BATCH_SAMPLES).collect();
            // Reflect pad mode matches librosa.pyin's default and gives
            // clean edges between batches. Zero-pad (the previous
            // setting) collapsed voiced_prob on the first/last ~40 ms
            // of each batch, which downstream consumers saw as periodic
            // 333 ms-cycle gaps in the trail on sustained voice.
            let (_timestamps, f0s, _voiced, voiced_prob) = self.executor.pyin(
                &chunk,
                f64::NAN, // unvoiced sentinel
                Framing::Center(PadMode::Reflect),
            );

            let time_offset = self.samples_processed as f32 / PYIN_SR as f32;
            let frame_idx_offset = self.samples_processed / HOP_LENGTH as u64;
            let n = f0s.len();
            for i in 0..n {
                let f0 = f0s[i];
                let vp = voiced_prob[i];
                let pitch = if f0.is_nan() { 0.0 } else { f0 as f32 };
                frames.push(PitchFrame {
                    frame_index: frame_idx_offset + i as u64,
                    time_s: time_offset + i as f32 * hop_s,
                    pitch_hz: pitch,
                    confidence: vp as f32,
                    is_preliminary: false,
                });
            }
            self.samples_processed += BATCH_SAMPLES as u64;
        }

        Ok(frames)
    }
}
