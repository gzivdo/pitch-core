use crate::estimator::{EstimatorError, Result};
use rubato::{
    Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction,
};

pub struct LinearResampler {
    inner: Option<SincFixedIn<f32>>,
    chunk_in: usize,
    pending: Vec<f32>,
}

impl LinearResampler {
    pub fn new(in_sr: u32, out_sr: u32, chunk_in: usize) -> Result<Self> {
        if in_sr == out_sr {
            return Ok(Self {
                inner: None,
                chunk_in,
                pending: Vec::new(),
            });
        }
        let params = SincInterpolationParameters {
            sinc_len: 128,
            f_cutoff: 0.95,
            interpolation: SincInterpolationType::Linear,
            oversampling_factor: 128,
            window: WindowFunction::BlackmanHarris2,
        };
        let ratio = out_sr as f64 / in_sr as f64;
        let inner = SincFixedIn::<f32>::new(ratio, 1.0, params, chunk_in, 1)
            .map_err(|e| EstimatorError::Resample(e.to_string()))?;
        Ok(Self {
            inner: Some(inner),
            chunk_in,
            pending: Vec::with_capacity(chunk_in * 2),
        })
    }

    pub fn push(&mut self, audio: &[f32]) -> Result<Vec<f32>> {
        if self.inner.is_none() {
            return Ok(audio.to_vec());
        }
        self.pending.extend_from_slice(audio);
        let mut out: Vec<f32> = Vec::new();
        let inner = self.inner.as_mut().unwrap();
        while self.pending.len() >= self.chunk_in {
            let chunk: Vec<f32> = self.pending.drain(..self.chunk_in).collect();
            let resampled = inner
                .process(&[chunk], None)
                .map_err(|e| EstimatorError::Resample(e.to_string()))?;
            out.extend_from_slice(&resampled[0]);
        }
        Ok(out)
    }

    pub fn reset(&mut self) {
        self.pending.clear();
        if let Some(inner) = self.inner.as_mut() {
            inner.reset();
        }
    }
}
