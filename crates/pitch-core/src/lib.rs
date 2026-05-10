//! Streaming pitch (f0) tracker. Pure-DSP backends only — neural
//! backends live in the companion crate `pitch-core-onnx`.
//!
//! # Quick start
//!
//! ```no_run
//! use pitch_core::{PitchTracker, SwipeEstimator};
//!
//! # fn main() -> Result<(), pitch_core::EstimatorError> {
//! let est = SwipeEstimator::new()?;
//! let mut tracker = PitchTracker::new(est, 48_000, 1024)?;
//!
//! # let chunk = vec![0.0f32; 4800];
//! for frame in tracker.process(&chunk)? {
//!     if frame.confidence > 0.3 {
//!         println!("{:.3}s  {:.1} Hz", frame.time_s, frame.pitch_hz);
//!     }
//! }
//! # Ok(()) }
//! ```
//!
//! # Adding ONNX backends
//!
//! Add `pitch-core-onnx` as a dependency and pass any of its estimators
//! to the same [`PitchTracker::new`]. The trait surface is identical:
//!
//! ```ignore
//! use pitch_core::PitchTracker;
//! use pitch_core_onnx::{SwiftF0Estimator, Mode};
//!
//! let est = SwiftF0Estimator::new("path/to/swift_f0.onnx", Mode::Balanced)?;
//! let mut tracker = PitchTracker::new(est, 48_000, 1024)?;
//! ```

pub mod estimator;
pub mod praat_ac;
pub mod pyin_est;
pub mod resample;
pub mod swipe;

pub use estimator::{calibrate_confidence, EstimatorError, PitchEstimator, PitchFrame, Result};
pub use praat_ac::PraatAcEstimator;
pub use pyin_est::PyinEstimator;
pub use swipe::SwipeEstimator;

/// High-level streaming pitch tracker. Combines any [`PitchEstimator`]
/// with a linear resampler that converts the host's sample rate to the
/// estimator's target rate.
///
/// Feed it mono `f32` audio at `input_sample_rate` via [`process`](Self::process).
/// The tracker buffers internally and returns frames as they become
/// available.
pub struct PitchTracker {
    estimator: Box<dyn PitchEstimator>,
    resampler: resample::LinearResampler,
    input_sr: u32,
    target_sr: u32,
}

impl PitchTracker {
    /// Build a tracker from any concrete [`PitchEstimator`].
    pub fn new<E: PitchEstimator + 'static>(
        estimator: E,
        input_sample_rate: u32,
        resample_chunk: usize,
    ) -> Result<Self> {
        Self::from_boxed(Box::new(estimator), input_sample_rate, resample_chunk)
    }

    /// Build a tracker from an already-boxed estimator. Useful when the
    /// estimator type is decided at runtime (e.g. from a CLI flag).
    pub fn from_boxed(
        estimator: Box<dyn PitchEstimator>,
        input_sample_rate: u32,
        resample_chunk: usize,
    ) -> Result<Self> {
        let target_sr = estimator.target_sample_rate();
        let resampler =
            resample::LinearResampler::new(input_sample_rate, target_sr, resample_chunk)?;
        Ok(Self {
            estimator,
            resampler,
            input_sr: input_sample_rate,
            target_sr,
        })
    }

    pub fn algorithm(&self) -> &str {
        self.estimator.name()
    }

    pub fn input_sample_rate(&self) -> u32 {
        self.input_sr
    }

    pub fn target_sample_rate(&self) -> u32 {
        self.target_sr
    }

    pub fn reset(&mut self) {
        self.resampler.reset();
        self.estimator.reset();
    }

    /// Push mono `f32` audio at `input_sample_rate`. Returns whatever
    /// frames the estimator produced after consuming this chunk. Empty
    /// at the start of the stream and after [`reset`](Self::reset)
    /// while the estimator's internal buffer fills.
    pub fn process(&mut self, audio: &[f32]) -> Result<Vec<PitchFrame>> {
        let resampled = self.resampler.push(audio)?;
        if resampled.is_empty() {
            return Ok(Vec::new());
        }
        self.estimator.process(&resampled)
    }
}
