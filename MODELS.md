# Model files for pitch-core-onnx

`pitch-core-onnx` ships **zero model weights**. Each ONNX backend
constructor takes a path to a user-supplied `.onnx` file. This document
explains where each model comes from and how to obtain it.

## TL;DR

```bash
# Default: fetches the 4 permissive backends (~570 MB total)
tools/download-models.sh

# Add PESTO (LGPL-3.0):
tools/download-models.sh --pesto

# Single backend:
tools/download-models.sh --fcpe-only
```

After the script: `models/` contains the ONNX files. Pass each path to
the corresponding constructor:

```rust
use pitch_core::PitchTracker;
use pitch_core_onnx::{SwiftF0Estimator, Mode, RmvpeEstimator, FcpeEstimator, CrepeEstimator};

let est = FcpeEstimator::new("models/fcpe.onnx")?;
let tracker = PitchTracker::new(est, 48_000, 1024)?;
```

## Per-backend acquisition

### SwiftF0 (~389 KB, MIT)

Bundled inside the `swift-f0` PyPI wheel by `lars76`. The download
script grabs the wheel from PyPI and extracts `swift_f0/model.onnx`.

Source: <https://pypi.org/project/swift-f0/>

### CREPE (~165 MB total, MIT)

Pre-converted ONNX files by `yqzhishen/onnxcrepe`, hosted on GitHub
Releases. Five capacities — tiny / small / medium / large / full —
share the same I/O contract. Pick whichever fits your size/accuracy
budget; pitch-core-onnx's `CrepeEstimator` works with any.

Source: <https://github.com/yqzhishen/onnxcrepe/releases/tag/v1.1.0>

### RMVPE (~362 MB, Apache-2.0 / MIT)

Original code: `Dream-High/RMVPE` (Apache-2.0).
Public ONNX mirror used by RVC: `lj1995/VoiceConversionWebUI` on
HuggingFace (MIT-licensed mirror of Apache-2.0 weights). The download
script pulls directly from HF.

Source: <https://huggingface.co/lj1995/VoiceConversionWebUI/blob/main/rmvpe.onnx>

### FCPE (~42 MB, MIT)

**Two paths** — pre-built or local-build:

1. **Pre-built** (preferred, fast): hosted on this repo's GitHub
   Releases. The download script tries this first.
2. **Local build** (fallback): if the pre-built URL is unreachable
   (network, behind firewall, custom hop_size, etc.), the script falls
   back to running `tools/fcpe_export.py` inside a local venv with
   `pip install torch torchfcpe`. Takes 1–2 min plus a one-time
   ~200 MB torch download.

Source code: <https://github.com/CNChTu/FCPE> (MIT — both code and
bundled checkpoint).

### PESTO (~17 MB) — ⚠ LGPL-3.0 weights

**Opt-in only.** Run `tools/download-models.sh --pesto` to fetch.

Upstream `SonyCSLParis/pesto` (`pesto-pitch` on PyPI) is **LGPL-3.0**.
Pretrained weights have no separate license declared upstream and
inherit LGPL-3.0 by default. The exported ONNX is therefore an
LGPL-3.0 derivative.

What this means in practice for redistribution:
- ✅ Use in your own projects (any license, including proprietary,
  thanks to LGPL-3 dynamic-linking allowance).
- ⚠ If you redistribute the .onnx file (e.g., bundle it in a binary
  release), you must comply with LGPL-3.0 §6: provide source / object
  files needed to relink, or use shared linking.
- ⚠ If your project license is incompatible with LGPL-3.0 (e.g.,
  some proprietary EULAs), **don't ship PESTO weights**; use the
  permissive 4 instead.

The download script prints this notice and refuses to fetch unless
`--pesto` is passed explicitly.

Source: <https://github.com/SonyCSLParis/pesto>,
<https://pypi.org/project/pesto-pitch/>

## Building a different model

The export scripts are deliberately kept simple — feel free to fork:

- `tools/fcpe_export.py` — wraps `torchfcpe.spawn_bundled_infer_model()`.
  Change the `--sr` flag to export at a different sample rate. Patch
  `torch.stft` survives ONNX legacy tracer (FakeComplex shim).
- `tools/pesto_export.py` — wraps PESTO's MIR-1K@48k checkpoint with
  HCQT preprocessor baked into the graph.

Both scripts accept `--out <path>` and use `python tools/<script>.py`
or via the auto-venv from `download-models.sh`.

## Contributing additional backends

The trait surface (`pitch_core::PitchEstimator`) is small — a new ONNX
backend is typically <200 lines of Rust + an export script. Open an
issue with the model paper / repo and we can sketch the I/O contract.
