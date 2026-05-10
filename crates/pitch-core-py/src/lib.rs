//! Python bindings for `pitch-core` (PyO3).
//!
//! DSP backends (`swipe`, `pyin`, `praat_ac`) are always available.
//! Neural backends (`swiftf0`, `crepe`, `pesto`) are gated behind the
//! `onnx` cargo feature; `pesto` requires both `onnx` and `pesto`.

use numpy::{IntoPyArray, PyReadonlyArray1};
use pitch_core::{EstimatorError, PitchEstimator, PitchTracker as CorePitchTracker};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

fn map_err(e: EstimatorError) -> PyErr {
    match e {
        EstimatorError::InvalidInput(m) => PyValueError::new_err(m),
        other => PyRuntimeError::new_err(other.to_string()),
    }
}

#[cfg_attr(not(feature = "onnx"), allow(unused_variables))]
fn build_estimator(
    algorithm: &str,
    model_path: &str,
    mode: &str,
    markov_step: bool,
    swipe_max_window: usize,
) -> PyResult<Box<dyn PitchEstimator>> {
    match algorithm {
        "swipe" => {
            let est = pitch_core::SwipeEstimator::with_max_window(swipe_max_window)
                .map_err(map_err)?;
            Ok(Box::new(est))
        }
        "pyin" => {
            let est = pitch_core::PyinEstimator::new().map_err(map_err)?;
            Ok(Box::new(est))
        }
        "praat_ac" => {
            let est = pitch_core::PraatAcEstimator::new(markov_step).map_err(map_err)?;
            Ok(Box::new(est))
        }

        #[cfg(feature = "onnx")]
        "swiftf0" => {
            let m = pitch_core_onnx::Mode::parse(mode).map_err(PyValueError::new_err)?;
            let est = pitch_core_onnx::SwiftF0Estimator::new(model_path, m).map_err(map_err)?;
            Ok(Box::new(est))
        }
        #[cfg(feature = "onnx")]
        "crepe" => {
            let est = pitch_core_onnx::CrepeEstimator::new(model_path).map_err(map_err)?;
            Ok(Box::new(est))
        }
        #[cfg(feature = "onnx")]
        "rmvpe" => {
            let est = pitch_core_onnx::RmvpeEstimator::new(model_path).map_err(map_err)?;
            Ok(Box::new(est))
        }
        #[cfg(feature = "onnx")]
        "fcpe" => {
            let est = pitch_core_onnx::FcpeEstimator::new(model_path).map_err(map_err)?;
            Ok(Box::new(est))
        }
        #[cfg(all(feature = "onnx", feature = "pesto"))]
        "pesto" => {
            let est = pitch_core_onnx::PestoEstimator::new(model_path).map_err(map_err)?;
            Ok(Box::new(est))
        }

        other => Err(PyValueError::new_err(format!(
            "unknown or disabled algorithm: {other}; available in this build: {}",
            available_backends().join(", ")
        ))),
    }
}

fn available_backends() -> Vec<&'static str> {
    #[cfg_attr(not(feature = "onnx"), allow(unused_mut))]
    let mut v = vec!["swipe", "pyin", "praat_ac"];
    #[cfg(feature = "onnx")]
    {
        v.push("swiftf0");
        v.push("crepe");
        v.push("rmvpe");
        v.push("fcpe");
    }
    #[cfg(all(feature = "onnx", feature = "pesto"))]
    v.push("pesto");
    v
}

#[pyclass(unsendable)]
struct PitchTracker {
    inner: CorePitchTracker,
}

#[pymethods]
impl PitchTracker {
    #[new]
    #[pyo3(signature = (
        algorithm,
        model_path = "",
        input_sample_rate = 48000,
        mode = "balanced",
        resample_chunk = 1024,
        markov_step = false,
        swipe_max_window = 8192,
    ))]
    fn new(
        algorithm: &str,
        model_path: &str,
        input_sample_rate: u32,
        mode: &str,
        resample_chunk: usize,
        markov_step: bool,
        swipe_max_window: usize,
    ) -> PyResult<Self> {
        let est = build_estimator(algorithm, model_path, mode, markov_step, swipe_max_window)?;
        let inner = CorePitchTracker::from_boxed(est, input_sample_rate, resample_chunk)
            .map_err(map_err)?;
        Ok(Self { inner })
    }

    #[getter]
    fn algorithm(&self) -> &str {
        self.inner.algorithm()
    }
    #[getter]
    fn input_sample_rate(&self) -> u32 {
        self.inner.input_sample_rate()
    }
    #[getter]
    fn target_sample_rate(&self) -> u32 {
        self.inner.target_sample_rate()
    }

    fn reset(&mut self) {
        self.inner.reset();
    }

    /// Process a chunk of audio at `input_sample_rate` (mono float32).
    /// Returns a dict with numpy arrays: `pitch_hz`, `confidence`,
    /// `timestamps_s`, `frame_indices`, `is_preliminary`.
    fn process<'py>(
        &mut self,
        py: Python<'py>,
        audio: PyReadonlyArray1<'py, f32>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let audio_slice = audio.as_slice()?;
        let frames = self.inner.process(audio_slice).map_err(map_err)?;

        let mut pitch = Vec::with_capacity(frames.len());
        let mut conf = Vec::with_capacity(frames.len());
        let mut times = Vec::with_capacity(frames.len());
        let mut indices = Vec::with_capacity(frames.len());
        let mut prelim = Vec::with_capacity(frames.len());
        for f in &frames {
            pitch.push(f.pitch_hz);
            conf.push(f.confidence);
            times.push(f.time_s);
            indices.push(f.frame_index as i64);
            prelim.push(f.is_preliminary);
        }

        let dict = PyDict::new(py);
        dict.set_item("pitch_hz", pitch.into_pyarray(py))?;
        dict.set_item("confidence", conf.into_pyarray(py))?;
        dict.set_item("timestamps_s", times.into_pyarray(py))?;
        dict.set_item("frame_indices", indices.into_pyarray(py))?;
        dict.set_item("is_preliminary", prelim)?;
        Ok(dict)
    }
}

/// Module registration. The function name must match `[lib].name` in
/// Cargo.toml so `import pitch_core_py` resolves the cdylib directly.
#[pymodule]
fn pitch_core_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PitchTracker>()?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("available_backends", available_backends())?;
    Ok(())
}
