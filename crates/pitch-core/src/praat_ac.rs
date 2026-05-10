//! Praat-style autocorrelation pitch estimator (Boersma 1993).
//!
//! Implements the bias-corrected autocorrelation method used by Praat's
//! "Sound: To Pitch (ac)..." function. The key step is dividing the windowed
//! signal's autocorrelation by the window's own autocorrelation, which removes
//! the bias that vanilla autocorrelation has toward shorter lags.
//!
//! NOT implemented:
//!   - Multi-candidate Viterbi path optimization across frames (Praat finds
//!     the top 15 peaks per frame, then picks the best path with octave-jump
//!     and voicing-transition costs). For a streaming/realtime estimator
//!     emitting a frame as soon as it can, only the global maximum is used.
//!   - Pre-emphasis / silence handling. Voicing decision is purely from peak
//!     normalized correlation strength vs threshold.
//!
//! Reference:
//!   Boersma, P. (1993). "Accurate short-term analysis of the fundamental
//!   frequency and the harmonics-to-noise ratio of a sampled sound."
//!   IFA Proceedings 17: 97-110.

use crate::estimator::{EstimatorError, PitchEstimator, PitchFrame, Result};
use realfft::num_complex::Complex32;
use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};
use std::sync::Arc;

const PRAAT_SR: u32 = 48000;
const HOP_SAMPLES: usize = 960; // 20 ms — match PESTO for fair comparison
// Window length covers >= 3 periods of fmin. With fmin=65 Hz: 3/65 ≈ 46 ms.
// Round up to 2048 samples (~43 ms @ 48k) for nice power of 2 input region.
const WINDOW_SAMPLES: usize = 2048;
// FFT size must be >= window_size + max_lag to keep the autocorrelation linear.
// max_lag = sr/fmin = 48000/65 ≈ 738. Total need: 2048 + 738 = 2786 → 4096.
const FFT_SIZE: usize = 4096;

const FMIN_HZ: f32 = 65.0;
const FMAX_HZ: f32 = 1100.0;
const VOICING_THRESHOLD: f32 = 0.45; // Praat's default for AC method
// Per-frame penalty: prefer shorter lags (higher freq) to break ties between
// integer-multiple lag coincidences (e.g. 880 Hz @48k @ lag=600 gives r=1.000
// exactly tying lag=54.5). Praat's `OctaveCost`, default 0.01.
const OCTAVE_COST: f32 = 0.01;

// Optional Markov-step penalty: prefer continuity with previous frame's lag.
// Approximates Praat's cross-frame Viterbi `OctaveJumpCost` (default 0.35) but
// applied as a 1-step Markov chain (no lookahead, zero added latency).
// Only applied when `markov_step = true` AND we have a previous voiced lag.
const JUMP_COST: f32 = 0.35;
// Voicing threshold for updating prev_lag — stale lags from quiet/unvoiced
// frames shouldn't anchor the next inference.
const PREV_LAG_VOICING: f32 = 0.45;

pub struct PraatAcEstimator {
    fft_forward: Arc<dyn RealToComplex<f32>>,
    fft_inverse: Arc<dyn ComplexToReal<f32>>,
    window: Vec<f32>,           // Hanning window of length WINDOW_SAMPLES
    window_ac: Vec<f32>,        // autocorrelation of the window (FFT_SIZE long; first half meaningful)
    buffer: Vec<f32>,
    next_frame_index: u64,
    markov_step: bool,
    prev_lag: Option<f32>,      // last voiced lag, used when markov_step is on
    // Reusable scratch buffers
    fft_input: Vec<f32>,
    spectrum: Vec<Complex32>,
    fft_output: Vec<f32>,
    forward_scratch: Vec<Complex32>,
    inverse_scratch: Vec<Complex32>,
}

fn hanning(n: usize) -> Vec<f32> {
    if n <= 1 {
        return vec![1.0; n];
    }
    let denom = (n - 1) as f32;
    (0..n)
        .map(|i| 0.5 - 0.5 * (std::f32::consts::TAU * i as f32 / denom).cos())
        .collect()
}

impl PraatAcEstimator {
    pub fn new(markov_step: bool) -> Result<Self> {
        let mut planner = RealFftPlanner::<f32>::new();
        let fft_forward = planner.plan_fft_forward(FFT_SIZE);
        let fft_inverse = planner.plan_fft_inverse(FFT_SIZE);

        let window = hanning(WINDOW_SAMPLES);

        // Precompute autocorrelation of the window itself: AC = IFFT(|FFT(w)|^2).
        let mut win_padded = vec![0.0_f32; FFT_SIZE];
        win_padded[..WINDOW_SAMPLES].copy_from_slice(&window);
        let mut spec = vec![Complex32::default(); FFT_SIZE / 2 + 1];
        let mut fwd_scratch = vec![Complex32::default(); fft_forward.get_scratch_len()];
        let mut inv_scratch = vec![Complex32::default(); fft_inverse.get_scratch_len()];
        fft_forward
            .process_with_scratch(&mut win_padded, &mut spec, &mut fwd_scratch)
            .map_err(|e| EstimatorError::InvalidInput(format!("fft fwd window: {e}")))?;
        for c in spec.iter_mut() {
            *c = Complex32::new(c.norm_sqr(), 0.0);
        }
        let mut window_ac = vec![0.0_f32; FFT_SIZE];
        fft_inverse
            .process_with_scratch(&mut spec, &mut window_ac, &mut inv_scratch)
            .map_err(|e| EstimatorError::InvalidInput(format!("ifft window ac: {e}")))?;

        Ok(Self {
            fft_forward,
            fft_inverse,
            window,
            window_ac,
            buffer: Vec::with_capacity(WINDOW_SAMPLES * 2),
            next_frame_index: 0,
            markov_step,
            prev_lag: None,
            fft_input: vec![0.0; FFT_SIZE],
            spectrum: vec![Complex32::default(); FFT_SIZE / 2 + 1],
            fft_output: vec![0.0; FFT_SIZE],
            forward_scratch: fwd_scratch,
            inverse_scratch: inv_scratch,
        })
    }
}

impl PitchEstimator for PraatAcEstimator {
    fn name(&self) -> &str {
        "praat_ac"
    }

    fn target_sample_rate(&self) -> u32 {
        PRAAT_SR
    }

    fn reset(&mut self) {
        self.buffer.clear();
        self.next_frame_index = 0;
        self.prev_lag = None;
    }

    fn process(&mut self, audio: &[f32]) -> Result<Vec<PitchFrame>> {
        self.buffer.extend_from_slice(audio);
        let mut out = Vec::new();
        let hop_s = HOP_SAMPLES as f32 / PRAAT_SR as f32;
        // center=False: window covers [start, start+WINDOW_SAMPLES). Audio
        // content centered at start+WINDOW_SAMPLES/2. Label time_s with
        // content time so pairwise alignment with center=True backends works.
        let center_offset_s = (WINDOW_SAMPLES as f32 / 2.0) / PRAAT_SR as f32;

        let lag_min = (PRAAT_SR as f32 / FMAX_HZ).ceil() as usize; // ~44
        let lag_max_frame = (PRAAT_SR as f32 / FMIN_HZ).floor() as usize; // ~738
        let lag_max = lag_max_frame.min(WINDOW_SAMPLES - 1);

        while self.buffer.len() >= WINDOW_SAMPLES {
            // Window the leading window_size samples; zero-pad to FFT size.
            for i in 0..WINDOW_SAMPLES {
                self.fft_input[i] = self.buffer[i] * self.window[i];
            }
            for v in &mut self.fft_input[WINDOW_SAMPLES..] {
                *v = 0.0;
            }

            // FFT, |X|^2, IFFT → linear autocorrelation of windowed signal.
            self.fft_forward
                .process_with_scratch(
                    &mut self.fft_input,
                    &mut self.spectrum,
                    &mut self.forward_scratch,
                )
                .map_err(|e| EstimatorError::InvalidInput(format!("fft fwd: {e}")))?;
            for c in self.spectrum.iter_mut() {
                *c = Complex32::new(c.norm_sqr(), 0.0);
            }
            self.fft_inverse
                .process_with_scratch(
                    &mut self.spectrum,
                    &mut self.fft_output,
                    &mut self.inverse_scratch,
                )
                .map_err(|e| EstimatorError::InvalidInput(format!("ifft: {e}")))?;

            // r_x(τ) ≈ r_a(τ) / r_w(τ).  Both r_a and r_w have the same constant
            // FFT-size factor, so the ratio gives a clean estimate.
            let r0_window = self.window_ac[0];
            let r0_signal = self.fft_output[0];
            let r0_x = if r0_window.abs() > 1e-12 {
                r0_signal / r0_window
            } else {
                0.0
            };

            let mut best_lag = lag_min;
            let mut best_norm = f32::NEG_INFINITY;
            let mut best_score = f32::NEG_INFINITY;
            let lag_min_f = lag_min as f32;
            let prev_lag_for_score = if self.markov_step { self.prev_lag } else { None };
            for lag in lag_min..=lag_max {
                let w_lag = self.window_ac[lag];
                if w_lag.abs() < 1e-12 {
                    continue;
                }
                let r_x = self.fft_output[lag] / w_lag;
                let norm = if r0_x.abs() > 1e-12 { r_x / r0_x } else { 0.0 };
                let mut score = norm - OCTAVE_COST * (lag as f32 / lag_min_f).log2();
                if let Some(prev) = prev_lag_for_score {
                    score -= JUMP_COST * (lag as f32 / prev).log2().abs();
                }
                if score > best_score {
                    best_score = score;
                    best_norm = norm;
                    best_lag = lag;
                }
            }

            // Parabolic interpolation around best peak (in normalized r_x).
            let normalized = |k: usize| -> f32 {
                let w = self.window_ac[k];
                if w.abs() < 1e-12 || r0_x.abs() < 1e-12 {
                    return 0.0;
                }
                (self.fft_output[k] / w) / r0_x
            };
            let lag_refined = if best_lag > lag_min && best_lag < lag_max {
                let y0 = normalized(best_lag - 1);
                let y1 = normalized(best_lag);
                let y2 = normalized(best_lag + 1);
                let denom = y0 - 2.0 * y1 + y2;
                let delta = if denom.abs() > 1e-10 {
                    0.5 * (y0 - y2) / denom
                } else {
                    0.0
                };
                best_lag as f32 + delta.clamp(-1.0, 1.0)
            } else {
                best_lag as f32
            };

            let pitch_hz = if lag_refined > 0.0 {
                PRAAT_SR as f32 / lag_refined
            } else {
                0.0
            };
            let confidence = best_norm.clamp(0.0, 1.0);

            // Voiced gating built into confidence: if peak strength below threshold,
            // we still report the lag's pitch but the consumer can filter on conf.
            let _voiced = confidence >= VOICING_THRESHOLD;

            out.push(PitchFrame {
                frame_index: self.next_frame_index,
                time_s: self.next_frame_index as f32 * hop_s + center_offset_s,
                pitch_hz,
                confidence,
                is_preliminary: false,
            });
            self.next_frame_index += 1;

            // Update prev_lag for next frame's Markov-step (only when voiced).
            if self.markov_step {
                self.prev_lag = if confidence >= PREV_LAG_VOICING {
                    Some(lag_refined)
                } else {
                    None
                };
            }

            self.buffer.drain(..HOP_SAMPLES);
        }

        Ok(out)
    }
}
