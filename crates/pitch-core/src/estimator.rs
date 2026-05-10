use thiserror::Error;

#[derive(Debug, Clone, Copy)]
pub struct PitchFrame {
    /// Monotonic absolute frame index from estimator start.
    /// Same `frame_index` may appear twice in progressive mode: first as
    /// preliminary, later as settled.
    pub frame_index: u64,
    pub time_s: f32,
    pub pitch_hz: f32,
    pub confidence: f32,
    /// True only in progressive mode for the latest edge-of-buffer frame
    /// that will be re-emitted later with refined values.
    pub is_preliminary: bool,
}

#[derive(Debug, Error)]
pub enum EstimatorError {
    #[error("ONNX runtime error: {0}")]
    Ort(String),
    #[error("TorchScript error: {0}")]
    TorchScript(String),
    #[error("resample error: {0}")]
    Resample(String),
    #[error("invalid input: {0}")]
    InvalidInput(String),
    #[error("not implemented: {0}")]
    NotImplemented(String),
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, EstimatorError>;

pub trait PitchEstimator: Send {
    fn name(&self) -> &str;
    fn target_sample_rate(&self) -> u32;
    fn process(&mut self, audio_target_sr: &[f32]) -> Result<Vec<PitchFrame>>;
    fn reset(&mut self) {}
}

/// Apply sigmoid-temperature scaling to a raw confidence value, gated
/// by the `confidence-calibration` cargo feature.
///
/// `t` is the per-backend temperature constant fitted by NLL minimization
/// on Vocadito GT (see ANALYSIS.md). When the feature is on and `t` is
/// `Some`, the calibrated confidence is `sigmoid(logit(raw) / t)`. When
/// the feature is off OR `t` is `None`, `raw` is returned unchanged.
#[inline]
pub fn calibrate_confidence(t: Option<f32>, raw: f32) -> f32 {
    #[cfg(feature = "confidence-calibration")]
    if let Some(t_val) = t {
        let p = raw.clamp(1e-7, 1.0 - 1e-7);
        let logit = (p / (1.0 - p)).ln();
        return 1.0 / (1.0 + (-logit / t_val).exp());
    }
    let _ = t;
    raw
}
