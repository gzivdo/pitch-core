//! SWIPE'/SWIPE pitch estimator — thin adapter over the public
//! `swipe-rs` crate (Apache-2.0). The pitch detection algorithm itself
//! is implemented in <https://crates.io/crates/swipe-rs>; this file just
//! plugs it into pitch-core's [`PitchEstimator`] trait and adds the
//! `is_preliminary: false` field that swipe-rs (deliberately) doesn't
//! know about.
//!
//! Reuses a single internal `Vec<swipe_rs::PitchFrame>` across `process`
//! calls via [`swipe_rs::Swipe::process_into`] — no per-call allocation
//! beyond the one we hand back to the caller.

use crate::estimator::{EstimatorError, PitchEstimator, PitchFrame, Result};
use swipe_rs::{Swipe, SAMPLE_RATE};

pub use swipe_rs::DEFAULT_MAX_WINDOW;

pub struct SwipeEstimator {
    inner: Swipe,
    /// Reused across calls to avoid allocations; `Swipe::process_into`
    /// appends to this and `process()` clears it first.
    scratch: Vec<swipe_rs::PitchFrame>,
}

impl SwipeEstimator {
    /// Same as [`Swipe::with_max_window`]. `max_window` is rounded up to
    /// the next power of two and clamped to the crate's supported range.
    pub fn with_max_window(max_window: usize) -> Result<Self> {
        let inner = Swipe::with_max_window(max_window)
            .map_err(|e| EstimatorError::Ort(format!("swipe init: {e}")))?;
        Ok(Self {
            inner,
            scratch: Vec::new(),
        })
    }

    pub fn new() -> Result<Self> {
        let inner =
            Swipe::new().map_err(|e| EstimatorError::Ort(format!("swipe init: {e}")))?;
        Ok(Self {
            inner,
            scratch: Vec::new(),
        })
    }
}

impl PitchEstimator for SwipeEstimator {
    fn name(&self) -> &str {
        "swipe"
    }
    fn target_sample_rate(&self) -> u32 {
        SAMPLE_RATE
    }
    fn reset(&mut self) {
        self.inner.reset();
        self.scratch.clear();
    }
    fn process(&mut self, audio: &[f32]) -> Result<Vec<PitchFrame>> {
        self.scratch.clear();
        self.inner
            .process_into(audio, &mut self.scratch)
            .map_err(|e| EstimatorError::Ort(format!("swipe fft: {e}")))?;
        Ok(self
            .scratch
            .iter()
            .map(|f| PitchFrame {
                frame_index: f.frame_index,
                time_s: f.time_s,
                pitch_hz: f.pitch_hz,
                confidence: f.confidence,
                is_preliminary: false,
            })
            .collect())
    }
}
