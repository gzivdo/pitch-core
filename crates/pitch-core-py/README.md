# pitch-core-py

Python bindings (PyO3) for [`pitch-core`](https://crates.io/crates/pitch-core).

Two install profiles:

| profile | feature flag | what's included | extra download |
|---|---|---|---|
| **DSP-only** (default) | none | SWIPE', pYIN, Praat-AC | none |
| **`onnx`** | `onnx` | + CREPE, RMVPE, SwiftF0, FCPE | ONNX Runtime (~50 MB) |
| **`pesto`** | `pesto` (implies `onnx`) | + PESTO (LGPL-3.0 weights) | as above |

Build with `maturin develop` for DSP-only; `maturin develop --features onnx`
for the full neural set; `maturin develop --features pesto` to also
include the LGPL-3.0 PESTO backend.

The package itself ships **no model weights**. ONNX-backend constructors
take a `model_path` argument; download the weights you need separately
(see upstream model cards for licenses).

## Authors

Created and maintained by **gzivdo** — design, architecture, research
direction, and code review.

Co-author (implementation): **Claude Opus 4.7** (Anthropic), under
gzivdo's direction. AI is not a copyright holder; copyright is held
entirely by the human maintainer (per US Copyright Office guidance,
Jan 2025). See [`NOTICE`](NOTICE) for the formal copyright line.
