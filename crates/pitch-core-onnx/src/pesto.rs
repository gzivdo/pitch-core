//! PESTO via ONNX Runtime.
//!
//! Stateful streaming: maintains a `cache` tensor that the model reads as
//! input and updates as output `cache_out`.
//!
//! ## Model file
//!
//! pitch-core does not ship PESTO weights. Build the ONNX locally via
//! `tools/pesto_export.py`, which loads the upstream PESTO checkpoint
//! (from the `pesto-pitch` PyPI package) and re-exports it with the
//! HCQT magnitude computed without `torch.view_as_complex` so that
//! ONNX Runtime can execute the graph. `tools/download-models.sh
//! --pesto` automates the venv + export.
//!
//! ## Licensing
//!
//! Upstream PESTO (`SonyCSLParis/pesto`, `pesto-pitch` on PyPI) is
//! licensed under **LGPL-3.0**. The pretrained weights have no separate
//! license declared upstream and so inherit LGPL-3.0 by default. The
//! ONNX file produced by the export script is a derivative of those
//! weights — when redistributing it, treat it as LGPL-3.0 unless
//! upstream clarifies otherwise.
//!
//! This file (`pesto.rs`) only loads and runs ONNX bytes via `ort`; it
//! does **not** import any PESTO code, so the LGPL-3.0 obligation does
//! not propagate to pitch-core itself (dual MIT/Apache-2.0).

use pitch_core::{EstimatorError, PitchEstimator, PitchFrame, Result};
use ndarray::Array2;
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Value;

const PESTO_SR: u32 = 48000;
const CHUNK_SAMPLES: usize = 960;
const CACHE_SIZE: usize = 3616;

pub struct PestoEstimator {
    session: Session,
    cache: Vec<f32>, // length CACHE_SIZE; persists across calls
    buffer: Vec<f32>,
    next_frame_index: u64,
}

impl PestoEstimator {
    pub fn new(model_path: &str) -> Result<Self> {
        let session = Session::builder()
            .map_err(|e| EstimatorError::Ort(e.to_string()))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| EstimatorError::Ort(e.to_string()))?
            .commit_from_file(model_path)
            .map_err(|e| EstimatorError::Ort(format!("load {model_path}: {e}")))?;
        Ok(Self {
            session,
            cache: vec![0.0; CACHE_SIZE],
            buffer: Vec::with_capacity(CHUNK_SAMPLES * 4),
            next_frame_index: 0,
        })
    }
}

impl PitchEstimator for PestoEstimator {
    fn name(&self) -> &str {
        "pesto"
    }
    fn target_sample_rate(&self) -> u32 {
        PESTO_SR
    }
    fn reset(&mut self) {
        self.cache.fill(0.0);
        self.buffer.clear();
        self.next_frame_index = 0;
    }

    fn process(&mut self, audio_target_sr: &[f32]) -> Result<Vec<PitchFrame>> {
        self.buffer.extend_from_slice(audio_target_sr);
        let hop_s = CHUNK_SAMPLES as f32 / PESTO_SR as f32;
        let mut out = Vec::new();

        while self.buffer.len() >= CHUNK_SAMPLES {
            let chunk: Vec<f32> = self.buffer.drain(..CHUNK_SAMPLES).collect();
            let audio_arr = Array2::from_shape_vec((1, CHUNK_SAMPLES), chunk)
                .map_err(|e| EstimatorError::Ort(format!("audio shape: {e}")))?;
            let cache_arr = Array2::from_shape_vec((1, CACHE_SIZE), self.cache.clone())
                .map_err(|e| EstimatorError::Ort(format!("cache shape: {e}")))?;
            let audio_val = Value::from_array(audio_arr)
                .map_err(|e| EstimatorError::Ort(format!("audio val: {e}")))?;
            let cache_val = Value::from_array(cache_arr)
                .map_err(|e| EstimatorError::Ort(format!("cache val: {e}")))?;

            let outputs = self
                .session
                .run(ort::inputs!["audio" => audio_val, "cache" => cache_val])
                .map_err(|e| EstimatorError::Ort(format!("forward: {e}")))?;

            let (_p_shape, pred) = outputs["prediction"]
                .try_extract_tensor::<f32>()
                .map_err(|e| EstimatorError::Ort(format!("pred extract: {e}")))?;
            let (_c_shape, conf) = outputs["confidence"]
                .try_extract_tensor::<f32>()
                .map_err(|e| EstimatorError::Ort(format!("conf extract: {e}")))?;
            let (_co_shape, cache_out) = outputs["cache_out"]
                .try_extract_tensor::<f32>()
                .map_err(|e| EstimatorError::Ort(format!("cache_out extract: {e}")))?;

            // PESTO returns MIDI float; convert to Hz.
            let midi = pred[0];
            let confidence = conf[0];
            let pitch_hz = 440.0_f32 * 2.0_f32.powf((midi - 69.0) / 12.0);

            // Update cache for next call.
            self.cache.clear();
            self.cache.extend_from_slice(cache_out);

            out.push(PitchFrame {
                frame_index: self.next_frame_index,
                time_s: self.next_frame_index as f32 * hop_s,
                pitch_hz,
                confidence,
                is_preliminary: false,
            });
            self.next_frame_index += 1;
        }

        Ok(out)
    }
}
