# pitch-core workspace

Three crates for pitch (f0) estimation in Rust:

| crate | role | dependencies |
|---|---|---|
| [`pitch-core`](crates/pitch-core/) | streaming tracker + pure-DSP backends (SWIPE', pYIN, Praat-AC) | no neural runtime, no model files |
| [`pitch-core-onnx`](crates/pitch-core-onnx/) | ONNX backends (SwiftF0, CREPE, RMVPE, FCPE; PESTO behind feature) on top of `pitch-core` | adds `ort` (~50 MB ONNX Runtime download) |
| [`pitch-core-py`](crates/pitch-core-py/) | PyO3 bindings, ONNX behind a feature flag | inherits both, depending on feature set |

The split lets you depend on `pitch-core` alone for pure-DSP use cases
(realtime pitch trackers, embedded targets, WASM) without dragging in
ONNX Runtime. Add `pitch-core-onnx` only if you actually need a neural
backend.

## Examples

| file | runs |
|---|---|
| [`crates/pitch-core/examples/quickstart.rs`](crates/pitch-core/examples/quickstart.rs) | DSP-only minimal example. SWIPE on a WAV file, printed voiced frames. |
| [`crates/pitch-core-onnx/examples/bench.rs`](crates/pitch-core-onnx/examples/bench.rs) | Full benchmark across all backends (DSP + ONNX) on a directory of WAVs. Markdown report with throughput, voicing, octave-error, pairwise agreement. |
| [`crates/pitch-core-py/python/quickstart.py`](crates/pitch-core-py/python/quickstart.py) | Python equivalent of `quickstart.rs` via `pitch-core-py` (PyO3). |

```bash
# Rust quickstart (no models needed):
cargo run --example quickstart --release -p pitch-core -- voice.wav

# Full benchmark (ONNX backends require models in ./models/):
cargo run --example bench --release -p pitch-core-onnx -- \
    songs_dir/ ./models --out report.md

# Python (after `maturin develop -m crates/pitch-core-py/Cargo.toml --release`):
python crates/pitch-core-py/python/quickstart.py voice.wav
```

## Models

`tools/download-models.sh` fetches all permissive ONNX backends
(SwiftF0, CREPE, RMVPE, FCPE — ~570 MB total) into `models/`. PESTO
is opt-in via `--pesto` (LGPL-3.0 weights). See
[MODELS.md](MODELS.md) for per-backend sources and licenses.

## Algorithm survey

[ANALYSIS.md](ANALYSIS.md) is an in-depth survey of the bundled
backends — strengths, weaknesses, what improves each group with and
without retraining, why note-level decoders are cleaner than any
frame-level pitch, and a sketch of what an ideal next-generation
realtime pitch detector would look like. Useful as background reading
before choosing a backend or designing a downstream system.

## Roadmap

- **v0.1** — first publish: 7 backends (3 DSP + 4 ONNX), `pesto` behind
  feature flag with LGPL-3.0 caveat, Python bindings, FCPE pre-built ONNX
  hosted on GitHub Releases.
- **v0.x** (no commitment) — `no_std` once `realfft`/`rustfft` gain
  `no_std + alloc`; finer pitch grid for retuning workflows.

## Status

Pre-1.0. The trait surface in `pitch-core` is intentionally minimal so
it should solidify quickly. Bug reports and PRs welcome.

## Authors

Created and maintained by **gzivdo** — design, architecture, research
direction, and code review.

Co-author (implementation): **Claude Opus 4.7** (Anthropic), under
gzivdo's direction. AI is not a copyright holder; copyright is held
entirely by the human maintainer (per US Copyright Office guidance,
Jan 2025). See [`NOTICE`](NOTICE) for the formal copyright line.

## License

Dual [Apache-2.0](LICENSE-APACHE) / [MIT](LICENSE-MIT) at your option.
Each crate carries its own LICENSE files (cargo requires them
per-crate). See each crate's `NOTICE` for algorithm citations and
upstream model-license caveats — pitch-core itself ships zero model
weights, but redistributing exported PESTO ONNX requires LGPL-3.0
compliance, see `crates/pitch-core-onnx/NOTICE`.
