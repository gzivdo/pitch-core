//! CREPE via ONNX Runtime.
//!
//! Uses the pre-converted ONNX models from `yqzhishen/onnxcrepe` (MIT). All
//! five capacities (tiny / small / medium / large / full) share the same
//! I/O contract:
//!
//!   input:  `frames`         shape `[n_frames, 1024]` float32
//!   output: `probabilities`  shape `[n_frames, 360]`  float32
//!
//! Each frame is a 1024-sample window at 16 kHz (= 64 ms of audio). 360
//! output bins span ~32.7 Hz to ~2006 Hz at 20 cents per bin. Confidence is
//! the post-sigmoid peak probability; pitch is decoded from a 9-bin
//! weighted average around the argmax — same recipe as the original CREPE
//! `weighted_argmax` decoder.
//!
//! Streaming: keep the last `WINDOW − HOP` samples as left context, emit
//! one frame per `HOP` of new audio. Default hop is 160 samples (10 ms),
//! configurable via `Capacity`'s factory if we ever expose it.

use pitch_core::{calibrate_confidence, EstimatorError, PitchEstimator, PitchFrame, Result};

/// Vocadito GT calibration: T=0.47. F1 0.918 → 0.945. See ANALYSIS.md.
const CALIBRATION_T: Option<f32> = Some(0.47);
use ndarray::Array2;
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Value;

const SR: u32 = 16000;
const WINDOW: usize = 1024;
const HOP: usize = 160; // 10 ms at 16 kHz
const PITCH_BINS: usize = 360;
const CENTS_PER_BIN: f32 = 20.0;
/// CREPE's bin-0 cents value relative to a 10 Hz reference. `bin → cents`
/// is `bin * 20 + 1997.379...`, then `cents → Hz = 10 * 2^(cents/1200)`.
const CENTS_OFFSET: f32 = 1997.379_4;

/// CREPE model capacity. Bigger = more accurate but slower / larger file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Capacity {
    Tiny,
    Small,
    Medium,
    Large,
    Full,
}

impl Capacity {
    pub fn parse(s: &str) -> std::result::Result<Self, String> {
        match s {
            "tiny" => Ok(Self::Tiny),
            "small" => Ok(Self::Small),
            "medium" => Ok(Self::Medium),
            "large" => Ok(Self::Large),
            "full" => Ok(Self::Full),
            other => Err(format!(
                "unknown crepe capacity: {other}; expected tiny|small|medium|large|full"
            )),
        }
    }
    pub fn short(self) -> &'static str {
        match self {
            Self::Tiny => "tiny",
            Self::Small => "small",
            Self::Medium => "medium",
            Self::Large => "large",
            Self::Full => "full",
        }
    }
}

pub struct CrepeEstimator {
    session: Session,
    /// Rolling audio buffer: kept long enough that we always have at least
    /// one full WINDOW available before slicing a frame.
    buffer: Vec<f32>,
    /// Sample index of `buffer[0]` in absolute estimator-stream coordinates.
    buffer_origin: u64,
    /// Sample index where the *next* frame's window starts.
    next_window_start: u64,
    /// Output frame counter, used for `PitchFrame::frame_index`.
    next_frame_index: u64,
}

impl CrepeEstimator {
    pub fn new(model_path: &str) -> Result<Self> {
        let session = Session::builder()
            .map_err(|e| EstimatorError::Ort(e.to_string()))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| EstimatorError::Ort(e.to_string()))?
            .commit_from_file(model_path)
            .map_err(|e| EstimatorError::Ort(format!("load {model_path}: {e}")))?;
        Ok(Self {
            session,
            buffer: Vec::with_capacity(WINDOW * 4),
            buffer_origin: 0,
            next_window_start: 0,
            next_frame_index: 0,
        })
    }
}

impl PitchEstimator for CrepeEstimator {
    fn name(&self) -> &str {
        "crepe"
    }

    fn target_sample_rate(&self) -> u32 {
        SR
    }

    fn reset(&mut self) {
        self.buffer.clear();
        self.buffer_origin = 0;
        self.next_window_start = 0;
        self.next_frame_index = 0;
    }

    fn process(&mut self, audio: &[f32]) -> Result<Vec<PitchFrame>> {
        self.buffer.extend_from_slice(audio);
        let hop_s = HOP as f32 / SR as f32;

        // Collect every window we can slice given the current buffer.
        let mut frames_flat: Vec<f32> = Vec::new();
        let mut emit_indices: Vec<u64> = Vec::new();
        loop {
            let win_start = self.next_window_start;
            let win_end = win_start + WINDOW as u64;
            let buf_end = self.buffer_origin + self.buffer.len() as u64;
            if win_end > buf_end {
                break;
            }
            let local = (win_start - self.buffer_origin) as usize;
            let frame = &self.buffer[local..local + WINDOW];

            // Per-frame zero-mean unit-stddev normalisation (the input
            // CREPE was trained on).
            let mean = frame.iter().copied().sum::<f32>() / WINDOW as f32;
            let var =
                frame.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / WINDOW as f32;
            let std = var.sqrt() + 1e-7;
            frames_flat.reserve(WINDOW);
            for &x in frame {
                frames_flat.push((x - mean) / std);
            }
            emit_indices.push(self.next_frame_index);
            self.next_frame_index += 1;
            self.next_window_start += HOP as u64;
        }

        if emit_indices.is_empty() {
            return Ok(Vec::new());
        }

        let n = emit_indices.len();
        let arr = Array2::from_shape_vec((n, WINDOW), frames_flat)
            .map_err(|e| EstimatorError::Ort(format!("frames shape: {e}")))?;
        let input = Value::from_array(arr)
            .map_err(|e| EstimatorError::Ort(format!("frames val: {e}")))?;

        let outputs = self
            .session
            .run(ort::inputs!["frames" => input])
            .map_err(|e| EstimatorError::Ort(format!("forward: {e}")))?;
        let (_shape, probs) = outputs["probabilities"]
            .try_extract_tensor::<f32>()
            .map_err(|e| EstimatorError::Ort(format!("probs extract: {e}")))?;

        // Decode each frame: weighted-mean cents over a ±4 bin window
        // around the argmax. Confidence = peak probability.
        //
        // Time labelling: each window covers samples [start, start+WINDOW)
        // with center=False semantics, so audio content is centered at
        // start+WINDOW/2. Label time_s with content time so pairwise
        // alignment with center=True backends works.
        let center_offset_s = (WINDOW as f32 / 2.0) / SR as f32;
        let mut out = Vec::with_capacity(n);
        for (i, &abs_idx) in emit_indices.iter().enumerate() {
            let row = &probs[i * PITCH_BINS..(i + 1) * PITCH_BINS];
            let (peak, peak_p) = row
                .iter()
                .copied()
                .enumerate()
                .fold((0usize, f32::NEG_INFINITY), |acc, (j, p)| {
                    if p > acc.1 { (j, p) } else { acc }
                });
            let lo = peak.saturating_sub(4);
            let hi = (peak + 4).min(PITCH_BINS - 1);
            let mut num = 0.0f32;
            let mut den = 0.0f32;
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
            let pitch_hz = 10.0_f32 * 2.0_f32.powf(cents / 1200.0);

            out.push(PitchFrame {
                frame_index: abs_idx,
                time_s: abs_idx as f32 * hop_s + center_offset_s,
                pitch_hz,
                confidence: calibrate_confidence(CALIBRATION_T, peak_p),
                is_preliminary: false,
            });
        }

        // Drop buffer prefix that we've already passed: keep the last
        // `WINDOW − HOP` samples for the next-frame context.
        let next_local = (self.next_window_start - self.buffer_origin) as usize;
        if next_local > 0 {
            let drop_n = next_local.saturating_sub(WINDOW - HOP);
            if drop_n > 0 {
                self.buffer.drain(..drop_n);
                self.buffer_origin += drop_n as u64;
            }
        }

        Ok(out)
    }
}
