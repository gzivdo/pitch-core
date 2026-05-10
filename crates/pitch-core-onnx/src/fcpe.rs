//! FCPE — Fast Context-based Pitch Estimation (CN_ChiTu, arXiv 2509.15140).
//!
//! Lynx-Net architecture with depth-wise separable convolutions. Per paper:
//! 96.79% RPA on MIR-1K, 5.3× faster than RMVPE, 77× faster than CREPE.
//!
//! ## Model file
//!
//! pitch-core ships no weights. Run `tools/download-models.sh --fcpe-only`
//! from the repo root to fetch the pre-built ONNX from HuggingFace, or
//! build it locally via `tools/fcpe_export.py` (loads the bundled
//! `torchfcpe.spawn_bundled_infer_model()` checkpoint and exports via
//! `torch.onnx.export` with a small monkey-patch keeping complex-type
//! STFT out of the ONNX graph). MIT throughout — code, weights,
//! exported ONNX.
//!
//! ## I/O contract
//!
//!   • input  :  audio  `[1, n_samples, 1]` float32 at 16 kHz
//!   • output :  f0_hz  `[1, n_frames, 1]` float32, **0 = unvoiced**
//!     (model gates internally with threshold=0.006).
//!
//! Hop = 160 samples @ 16k = 10 ms. Confidence in this wrapper is
//! binary (1.0 if f0 > 0 else 0.0) since the ONNX export only exposes
//! gated f0 — the underlying voicing logits aren't surfaced.
//!
//! ## Streaming
//!
//! Buffer audio, run inference once we have at least `MIN_FRESH_SAMPLES`
//! plus `CONTEXT_SAMPLES` of context. Same overlap-and-save pattern as
//! SwiftF0 / RMVPE.

use pitch_core::{calibrate_confidence, EstimatorError, PitchEstimator, PitchFrame, Result};

/// Vocadito GT calibration: T=5.10. FCPE confidence is binary (1.0/0.0)
/// because the export gates internally; calibration is essentially a
/// no-op. Kept for API symmetry. See ANALYSIS.md.
const CALIBRATION_T: Option<f32> = Some(5.10);
use ndarray::Array3;
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Value;

const SR: u32 = 16_000;
const HOP: usize = 160;            // 10 ms @ 16k

/// Left-context audio kept across calls.
///
/// Empirically (sweep on a 130 s vocal): left context has **no effect**
/// on streaming accuracy — varied 16/32/48/64 at fixed fresh=16,
/// agreement-vs-batch stayed flat at 27.4%. The model needs
/// right-context (future audio) per frame, not left. We keep 32 frames
/// of context anyway as a safety margin.
const CONTEXT_SAMPLES: usize = 32 * HOP;

/// Minimum fresh audio before triggering inference. **Hard floor on
/// streaming latency** — each emitted frame depends on up to
/// `MIN_FRESH_SAMPLES / HOP` future frames being in the inference
/// buffer.
///
/// Empirical sweep (vs offline batch on the same audio):
///
/// | fresh frames | latency | agreement-vs-batch |
/// |---|---|---|
/// | 32 | 320 ms | **97.6%** |
/// | 16 | 192 ms |   27.4% ❌ |
/// |  8 | 128 ms |   11.1% ❌ |
/// |  4 |  64 ms |   11.0% ❌ |
///
/// Below 32 the model collapses.
const MIN_FRESH_SAMPLES: usize = 32 * HOP;

pub struct FcpeEstimator {
    session: Session,
    in_name: String,
    out_name: String,
    audio_buffer: Vec<f32>,
    context_held: usize,           // tail samples already emitted as context
    next_frame_index: u64,
}

impl FcpeEstimator {
    pub fn new(model_path: &str) -> Result<Self> {
        let session = Session::builder()
            .map_err(|e| EstimatorError::Ort(e.to_string()))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| EstimatorError::Ort(e.to_string()))?
            .commit_from_file(model_path)
            .map_err(|e| EstimatorError::Ort(format!("load {model_path}: {e}")))?;

        let in_name = session
            .inputs()
            .first()
            .ok_or_else(|| EstimatorError::Ort("fcpe: model has no inputs".into()))?
            .name()
            .to_string();
        let out_name = session
            .outputs()
            .first()
            .ok_or_else(|| EstimatorError::Ort("fcpe: model has no outputs".into()))?
            .name()
            .to_string();

        Ok(Self {
            session,
            in_name,
            out_name,
            audio_buffer: Vec::with_capacity(SR as usize),
            context_held: 0,
            next_frame_index: 0,
        })
    }
}

impl PitchEstimator for FcpeEstimator {
    fn name(&self) -> &str {
        "fcpe"
    }

    fn target_sample_rate(&self) -> u32 {
        SR
    }

    fn reset(&mut self) {
        self.audio_buffer.clear();
        self.context_held = 0;
        self.next_frame_index = 0;
    }

    fn process(&mut self, audio: &[f32]) -> Result<Vec<PitchFrame>> {
        self.audio_buffer.extend_from_slice(audio);
        let fresh_samples = self.audio_buffer.len() - self.context_held;
        if fresh_samples < MIN_FRESH_SAMPLES {
            return Ok(Vec::new());
        }

        // Build input tensor [1, n_samples, 1].
        let n_samples = self.audio_buffer.len();
        let arr = Array3::from_shape_vec(
            (1, n_samples, 1),
            self.audio_buffer.clone(),
        )
        .map_err(|e| EstimatorError::Ort(format!("fcpe shape: {e}")))?;
        let input = Value::from_array(arr)
            .map_err(|e| EstimatorError::Ort(format!("fcpe value: {e}")))?;

        let outputs = self
            .session
            .run(ort::inputs![self.in_name.as_str() => input])
            .map_err(|e| EstimatorError::Ort(format!("fcpe run: {e}")))?;
        let (_shape, f0_data): (_, &[f32]) = outputs[self.out_name.as_str()]
            .try_extract_tensor::<f32>()
            .map_err(|e| EstimatorError::Ort(format!("fcpe extract: {e}")))?;

        // torchfcpe returns n_samples // hop + 1 frames where the LAST
        // is the center-pad edge at the very end of the buffer (audio
        // time = buffer_end). We never emit it — it'll be recomputed
        // properly next call when right-context is available. Without
        // this skip, every call emits the trailing edge and the next
        // call duplicates that absolute time, drifting +1 frame per
        // call → cumulative 80%+ disagreement vs offline batch.
        let context_frames = self.context_held / HOP;
        let fresh_frames = (self.audio_buffer.len() - self.context_held) / HOP;

        let hop_s = HOP as f32 / SR as f32;
        let mut out = Vec::with_capacity(fresh_frames);
        for i in 0..fresh_frames {
            let f0 = f0_data[context_frames + i];
            // Voiced iff f0 is finite positive — torchfcpe occasionally
            // emits NaN on quiet/silent frames (internal log(0)). Treat
            // NaN/inf as unvoiced.
            let voiced = f0.is_finite() && f0 > 0.0;
            let abs_idx = self.next_frame_index + i as u64;
            out.push(PitchFrame {
                frame_index: abs_idx,
                time_s: abs_idx as f32 * hop_s,
                pitch_hz: if voiced { f0 } else { 0.0 },
                confidence: calibrate_confidence(
                    CALIBRATION_T,
                    if voiced { 1.0 } else { 0.0 },
                ),
                is_preliminary: false,
            });
        }
        self.next_frame_index += fresh_frames as u64;

        // Retain CONTEXT_SAMPLES tail for next call.
        let keep = CONTEXT_SAMPLES.min(self.audio_buffer.len());
        if self.audio_buffer.len() > keep {
            let drop_n = self.audio_buffer.len() - keep;
            self.audio_buffer.drain(..drop_n);
        }
        self.context_held = keep;

        Ok(out)
    }
}
