use ndarray::Array2;
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Value;
use pitch_core::{calibrate_confidence, EstimatorError, PitchEstimator, PitchFrame, Result};

/// Vocadito GT calibration: T=1.96. F1 0.906 → 0.937. See ANALYSIS.md.
const CALIBRATION_T: Option<f32> = Some(1.96);

// SwiftF0 is a temporal CNN over an STFT — single isolated frames give garbage
// confidence (~0.3); ~4 frames of context give ~0.88; ~16 frames give ~1.0.
// We use overlap-and-save: keep the last `context_samples` audio samples as
// context for next call, accumulate at least `min_fresh_samples` of new audio
// before running, then emit only frames covering the new audio.
const HOP_SAMPLES: usize = 256;

/// Streaming latency profile for [`SwiftF0Estimator`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// 64 ms context + 64 ms fresh, ~70 ms peak latency, ~16 Hz UI rate, low CPU.
    Balanced,
    /// 112 ms context + 16 ms fresh, ~22 ms latency, 60 Hz UI rate, ~4× inferences.
    LowLatency,
    /// Same buffers as `LowLatency`, plus the newest frame is emitted preliminary
    /// and re-emitted ~48 ms later as settled (use `is_preliminary` + `frame_index`
    /// in the UI to dedup).
    Progressive,
}

impl Mode {
    pub fn parse(s: &str) -> std::result::Result<Self, String> {
        match s {
            "balanced" => Ok(Self::Balanced),
            "low_latency" => Ok(Self::LowLatency),
            "progressive" => Ok(Self::Progressive),
            other => Err(format!(
                "unknown mode: {other}; supported: balanced, low_latency, progressive"
            )),
        }
    }

    fn streaming_params(self) -> (usize, usize, bool, usize) {
        match self {
            Self::Balanced => (1024, 1024, false, 0),
            Self::LowLatency => (1792, 256, false, 0),
            Self::Progressive => (1792, 256, true, 3),
        }
    }
}

pub struct SwiftF0Estimator {
    session: Session,
    target_sr: u32,
    buffer: Vec<f32>,
    context_held: usize,
    next_frame_index: u64,
    context_samples: usize,
    min_fresh_samples: usize,
    progressive: bool,
    settle_lag_frames: usize,
}

impl SwiftF0Estimator {
    /// Construct from a [`Mode`] preset. Use this in normal code.
    pub fn new(model_path: &str, mode: Mode) -> Result<Self> {
        let (context, fresh, progressive, settle) = mode.streaming_params();
        Self::with_params(model_path, context, fresh, progressive, settle)
    }

    /// Low-level constructor exposing every streaming knob. Use this only
    /// if [`Mode`] doesn't fit your latency/CPU budget.
    pub fn with_params(
        model_path: &str,
        context_samples: usize,
        min_fresh_samples: usize,
        progressive: bool,
        settle_lag_frames: usize,
    ) -> Result<Self> {
        if context_samples % HOP_SAMPLES != 0 || min_fresh_samples % HOP_SAMPLES != 0 {
            return Err(EstimatorError::InvalidInput(format!(
                "context_samples ({context_samples}) and min_fresh_samples ({min_fresh_samples}) must be multiples of hop ({HOP_SAMPLES})"
            )));
        }
        if min_fresh_samples == 0 {
            return Err(EstimatorError::InvalidInput(
                "min_fresh_samples must be > 0".into(),
            ));
        }
        if progressive && settle_lag_frames == 0 {
            return Err(EstimatorError::InvalidInput(
                "progressive mode requires settle_lag_frames >= 1".into(),
            ));
        }
        let session = Session::builder()
            .map_err(|e| EstimatorError::Ort(e.to_string()))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| EstimatorError::Ort(e.to_string()))?
            .commit_from_file(model_path)
            .map_err(|e| EstimatorError::Ort(e.to_string()))?;
        Ok(Self {
            session,
            target_sr: 16000,
            buffer: Vec::with_capacity(context_samples + min_fresh_samples * 2),
            context_held: 0,
            next_frame_index: 0,
            context_samples,
            min_fresh_samples,
            progressive,
            settle_lag_frames,
        })
    }
}

impl PitchEstimator for SwiftF0Estimator {
    fn name(&self) -> &str {
        "swiftf0"
    }

    fn target_sample_rate(&self) -> u32 {
        self.target_sr
    }

    fn reset(&mut self) {
        self.buffer.clear();
        self.context_held = 0;
        self.next_frame_index = 0;
    }

    fn process(&mut self, audio_target_sr: &[f32]) -> Result<Vec<PitchFrame>> {
        self.buffer.extend_from_slice(audio_target_sr);
        let fresh = self.buffer.len() - self.context_held;
        if fresh < self.min_fresh_samples {
            return Ok(Vec::new());
        }

        // Run on whole buffer (context + fresh, aligned to HOP).
        let n = (self.buffer.len() / HOP_SAMPLES) * HOP_SAMPLES;
        let arr = Array2::from_shape_vec((1, n), self.buffer[..n].to_vec())
            .map_err(|e| EstimatorError::Ort(format!("shape: {e}")))?;
        let input = Value::from_array(arr).map_err(|e| EstimatorError::Ort(e.to_string()))?;

        let outputs = self
            .session
            .run(ort::inputs!["input_audio" => input])
            .map_err(|e| EstimatorError::Ort(e.to_string()))?;

        let (_p_shape, pitch_data) = outputs["pitch_hz"]
            .try_extract_tensor::<f32>()
            .map_err(|e| EstimatorError::Ort(e.to_string()))?;
        let (_c_shape, conf_data) = outputs["confidence"]
            .try_extract_tensor::<f32>()
            .map_err(|e| EstimatorError::Ort(e.to_string()))?;

        let total_frames = pitch_data.len();
        let context_frames = self.context_held / HOP_SAMPLES;
        let fresh_frames = total_frames.saturating_sub(context_frames);

        let hop_s = HOP_SAMPLES as f32 / self.target_sr as f32;
        // Absolute frame index of inference output position 0:
        let buffer_first_idx = self.next_frame_index - context_frames as u64;

        let mut out = Vec::with_capacity(fresh_frames + 1);

        // Progressive: re-emit a previously-preliminary frame as settled now that
        // it has `settle_lag_frames` hops of right context.
        if self.progressive && self.next_frame_index >= 1 + self.settle_lag_frames as u64 {
            let settled_idx = self.next_frame_index - 1 - self.settle_lag_frames as u64;
            if settled_idx >= buffer_first_idx {
                let pos = (settled_idx - buffer_first_idx) as usize;
                if pos < total_frames {
                    out.push(PitchFrame {
                        frame_index: settled_idx,
                        time_s: settled_idx as f32 * hop_s,
                        pitch_hz: pitch_data[pos],
                        confidence: calibrate_confidence(CALIBRATION_T, conf_data[pos]),
                        is_preliminary: false,
                    });
                }
            }
        }

        // Emit fresh frames. In progressive mode, only the very newest one is
        // marked preliminary (it will be re-emitted later as settled).
        for i in 0..fresh_frames {
            let pos = context_frames + i;
            let abs_idx = self.next_frame_index + i as u64;
            let is_prelim = self.progressive && (i + 1 == fresh_frames);
            out.push(PitchFrame {
                frame_index: abs_idx,
                time_s: abs_idx as f32 * hop_s,
                pitch_hz: pitch_data[pos],
                confidence: calibrate_confidence(CALIBRATION_T, conf_data[pos]),
                is_preliminary: is_prelim,
            });
        }
        self.next_frame_index += fresh_frames as u64;

        // Retain trailing context_samples for next call's left context.
        let keep = self.context_samples.min(n);
        self.buffer.drain(..(n - keep));
        self.context_held = keep;

        Ok(out)
    }
}
