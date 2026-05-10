//! ONNX backends for [`pitch-core`](https://crates.io/crates/pitch-core).
//!
//! Each estimator implements `pitch_core::PitchEstimator` and can be
//! plugged directly into `PitchTracker::new`. The crate ships **zero
//! model weights** — every constructor takes a path to a user-supplied
//! `.onnx` file.
//!
//! # Example
//!
//! ```no_run
//! use pitch_core::PitchTracker;
//! use pitch_core_onnx::{SwiftF0Estimator, Mode};
//!
//! # fn main() -> Result<(), pitch_core::EstimatorError> {
//! let est = SwiftF0Estimator::new("models/swift_f0.onnx", Mode::Balanced)?;
//! let mut tracker = PitchTracker::new(est, 48_000, 1024)?;
//! # Ok(()) }
//! ```
//!
//! # Backends and feature flags
//!
//! | backend | feature | upstream license |
//! |---|---|---|
//! | [`SwiftF0Estimator`] | always on | MIT |
//! | [`CrepeEstimator`] | always on | MIT |
//! | [`RmvpeEstimator`] | always on | Apache-2.0 (code) / MIT (mirror) |
//! | [`FcpeEstimator`] | always on | MIT |
//! | `PestoEstimator` | `pesto` (off by default) | **LGPL-3.0** — see [`pesto`] module |
//!
//! # Model files
//!
//! Pull weights from upstream. Suggested sources:
//!
//! - **SwiftF0**: `lars76/swift-f0` PyPI wheel (MIT)
//! - **CREPE**: `yqzhishen/onnxcrepe` (MIT) — five capacities, all share the
//!   same I/O contract
//! - **RMVPE**: `lj1995/VoiceConversionWebUI/rmvpe.onnx` on HuggingFace
//!   (~362 MB). Vocal-tuned, polyphony-robust — best for singing through
//!   light accompaniment. Code is Apache-2.0 (Dream-High/RMVPE), the ONNX
//!   mirror is MIT.
//! - **PESTO**: re-export from PyPI `pesto-pitch` package; the conversion
//!   patch lives in `tools/pesto_export.py`. Upstream code is **LGPL-3.0**.

pub mod crepe;
pub mod fcpe;
pub mod rmvpe;
pub mod swiftf0;

pub use crepe::{Capacity, CrepeEstimator};
pub use fcpe::FcpeEstimator;
pub use rmvpe::RmvpeEstimator;
pub use swiftf0::{Mode, SwiftF0Estimator};

#[cfg(feature = "pesto")]
pub mod pesto;
#[cfg(feature = "pesto")]
pub use pesto::PestoEstimator;
