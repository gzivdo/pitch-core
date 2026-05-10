---
license: mit
tags:
  - audio
  - pitch-estimation
  - f0
  - vocal
  - onnx
  - fcpe
language:
  - en
library_name: onnxruntime
pipeline_tag: audio-to-audio
---

# FCPE ONNX — unofficial export

Pre-converted ONNX export of [FCPE](https://github.com/CNChTu/FCPE)
(Fast Context-based Pitch Estimation, CN_ChiTu, arXiv 2509.15140).

This is an **unofficial community export** of the bundled torchfcpe
checkpoint, intended for use without the PyTorch dependency. The
weights and architecture are unchanged — only the runtime is
swapped from torch to ONNX Runtime.

## Provenance

- **Upstream code & weights**: <https://github.com/CNChTu/FCPE>
  (MIT — see [LICENSE](https://github.com/CNChTu/FCPE/blob/main/LICENSE))
- **Upstream paper**: Tu, "FCPE: A Fast Context-based Pitch Estimation
  Model", arXiv [2509.15140](https://arxiv.org/abs/2509.15140), 2025
- **Bundled checkpoint version**: `torchfcpe == 0.0.4` (PyPI)
- **Export script** (this conversion): [pitch-core/tools/fcpe_export.py](https://github.com/gzivdo/pitch-core/blob/main/tools/fcpe_export.py)
  (MIT OR Apache-2.0, copyright 2026 gzivdo)
- **Reproduction**: `python tools/fcpe_export.py --out fcpe.onnx`
  (requires `pip install torch torchfcpe`)

This export is **not endorsed by, affiliated with, or sponsored by**
the FCPE authors. It is provided as a convenience for the open-source
community.

## I/O contract

```
input:  audio   float32  [1, n_samples, 1]   raw mono audio @ 16 kHz
output: f0_hz   float32  [1, n_frames, 1]    f0 in Hz (0 = unvoiced)
```

- Sample rate: **16 000 Hz** (resample your input before feeding)
- Hop: **160 samples** = 10 ms
- Output frames: `n_samples // 160 + 1`
- Voicing gate: model applies internal `threshold=0.006` on confidence;
  frames with confidence below it are returned as `f0=0`. Some quiet
  frames may also return `NaN` (internal `log(0)`) — treat as unvoiced.

## Usage (Python)

```python
import numpy as np
import onnxruntime as ort
import librosa

audio, _ = librosa.load("vocal.wav", sr=16_000, mono=True)
sess = ort.InferenceSession("fcpe.onnx", providers=["CPUExecutionProvider"])
f0 = sess.run(["f0_hz"], {"audio": audio.astype(np.float32)[None, :, None]})[0]
f0 = f0[0, :, 0]
voiced = np.isfinite(f0) & (f0 > 0)
print(f"voiced: {voiced.sum()}/{len(f0)} frames")
```

## Usage (Rust via pitch-core-onnx)

```rust
use pitch_core::PitchTracker;
use pitch_core_onnx::FcpeEstimator;

let est = FcpeEstimator::new("fcpe.onnx")?;
let mut tracker = PitchTracker::new(est, 48_000, 1024)?;
for frame in tracker.process(&audio_chunk)? { /* ... */ }
```

See <https://crates.io/crates/pitch-core-onnx> for the full crate.

## Citation

If you use this model in academic work, cite the upstream paper, not
this export:

```bibtex
@article{tu2025fcpe,
  title  = {FCPE: A Fast Context-based Pitch Estimation Model},
  author = {CN\_ChiTu},
  journal = {arXiv preprint arXiv:2509.15140},
  year   = {2025},
  url    = {https://arxiv.org/abs/2509.15140}
}
```

## License

This ONNX file inherits the MIT license from the FCPE upstream:

> MIT License
>
> Copyright (c) 2023 CN_ChiTu
>
> Permission is hereby granted, free of charge, to any person obtaining
> a copy of this software and associated documentation files (the
> "Software"), to deal in the Software without restriction […]
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
> EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
> MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
> NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
> BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY […]

Full text: <https://github.com/CNChTu/FCPE/blob/main/LICENSE>

## Disclaimer

The export script `tools/fcpe_export.py` applies a small monkey-patch
to `torch.stft` so the legacy ONNX tracer can handle the complex-typed
output from torchfcpe's mel extractor. The patch wraps the real-tensor
output in a `_FakeComplex` shim that exposes `.real` / `.imag` as
indexed views — semantically equivalent to the original. Numerical
output should match the upstream torchfcpe model bit-for-bit modulo
floating-point rounding in the ORT runtime.

This file is provided "AS IS", per the MIT license above. The
maintainer makes no claims about its accuracy on data outside the
ranges tested by upstream and provides no warranty of fitness for any
particular purpose.

If the upstream FCPE project releases an official ONNX export, prefer
that. If you find a discrepancy between this export and upstream
torchfcpe inference, please open an issue at
<https://github.com/gzivdo/pitch-core/issues>.
