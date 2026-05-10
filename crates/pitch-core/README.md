# pitch-core

Streaming pitch (f0) tracker for Rust. Three pure-DSP backends behind one
trait. **No model files. No GPU. No ONNX Runtime.**

```rust
use pitch_core::{PitchTracker, SwipeEstimator};

let est = SwipeEstimator::new()?;                        // 96.4% RPA on MIR-1K
let mut tracker = PitchTracker::new(est, 48_000, 1024)?;

for chunk in mic_chunks {
    for f in tracker.process(&chunk)? {
        if f.confidence > 0.3 {
            println!("{:6.3}s  {:6.1} Hz  conf={:.2}",
                     f.time_s, f.pitch_hz, f.confidence);
        }
    }
}
```

Add to `Cargo.toml`:
```toml
[dependencies]
pitch-core = "0.1"
```

For neural backends (CREPE, RMVPE, SwiftF0, FCPE; PESTO behind feature)
add the companion crate
[`pitch-core-onnx`](https://crates.io/crates/pitch-core-onnx).

## Backends

| algorithm | constructor | target rate | hop | latency | notes |
|---|---|---|---|---|---|
| SWIPE'/SWIPE | `SwipeEstimator::new()` (Balanced) / `with_max_window()` | 48 kHz | 10 ms | 85 / 170 / 340 ms | strongest pure-DSP option; wraps [`swipe-rs`](https://crates.io/crates/swipe-rs) |
| pYIN | `PyinEstimator::new()` | 48 kHz | 20 ms | ~333 ms (batched) | conservative voicing — high precision, low recall; wraps [`pyin`](https://crates.io/crates/pyin) |
| Praat-AC | `PraatAcEstimator::new(markov_step: bool)` | 48 kHz | 20 ms | ~43 ms | Boersma 1993, original FFT impl; cheap & fast for speech |

All three implement the [`PitchEstimator`] trait and slot into
[`PitchTracker`] interchangeably. The trait surface is small enough
that custom backends are trivial to add — see `examples/custom_backend.rs`.

## Streaming model

Feed mono `f32` audio at the host sample rate via
[`PitchTracker::process`]. A built-in linear resampler converts to the
backend's target rate, the backend buffers internally, and `process`
returns whatever frames are ready. An empty `Vec` is returned while the
backend is filling its first window — that's normal at the start of the
stream and after [`reset`](PitchTracker::reset).

`PitchFrame::is_preliminary` is reserved for backends that emit
edge-of-buffer estimates twice (preliminary + settled). None of the
DSP backends in this crate do; SwiftF0 in `pitch-core-onnx` does.

## Acknowledgements

- A. Camacho & J. G. Harris — SWIPE/SWIPE' (JASA 2008)
- D. Marttila & J. D. Reiss — multi-resolution mel-axis SWIPE (ISMIR 2025)
- M. Mauch & S. Dixon — pYIN (ICASSP 2014)
- P. Boersma — autocorrelation-with-window-AC method (Praat, 1993)
- Sytronik — `pyin` Rust crate

## Authors

Created and maintained by **gzivdo** — design, architecture, research
direction, and code review.

Co-author (implementation): **Claude Opus 4.7** (Anthropic), under
gzivdo's direction. AI is not a copyright holder; copyright is held
entirely by the human maintainer (per US Copyright Office guidance,
Jan 2025).

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE)
or [MIT license](LICENSE-MIT) at your option. Algorithms themselves are
not copyrightable (US §102(b); Feist v. Rural, 1991; EU Directive
2009/24/EC) — all Rust code in this crate is independent
reimplementation or original glue.

Unless you explicitly state otherwise, any contribution intentionally
submitted for inclusion in this crate by you, as defined in the
Apache-2.0 license, shall be dual-licensed as above, without any
additional terms or conditions.
