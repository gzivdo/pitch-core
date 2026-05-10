"""Minimal pitch-core-py example: load a WAV, run SWIPE (pure DSP),
print voiced frames.

Run:

    # Build the Rust extension into the current Python env (requires `pip
    # install maturin` first):
    cd <repo-root>
    maturin develop -m crates/pitch-core-py/Cargo.toml --release

    # Then:
    pip install soundfile numpy
    python crates/pitch-core-py/python/quickstart.py path/to/voice.wav

For neural backends (CREPE, RMVPE, SwiftF0, FCPE), build with the
`onnx` feature instead:

    maturin develop -m crates/pitch-core-py/Cargo.toml --release \\
        --features onnx

then pass `algorithm="rmvpe"` (or any other) and `model_path=...` to
`PitchTracker(...)`. See `pitch_core_py.available_backends` for the
list compiled into your build.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import soundfile as sf

import pitch_core_py as pcp


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: quickstart.py <wav-file>", file=sys.stderr)
        return 1

    path = Path(sys.argv[1])
    audio, sr = sf.read(str(path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1).astype(np.float32)

    print(f"loaded {path}: {len(audio)/sr:.2f} s @ {sr} Hz", file=sys.stderr)
    print(f"available backends: {pcp.available_backends}", file=sys.stderr)

    # Pure-DSP SWIPE — no model file, no GPU. ~96.4% RPA on MIR-1K.
    tracker = pcp.PitchTracker(
        algorithm="swipe",
        input_sample_rate=sr,
        resample_chunk=1024,
        swipe_max_window=8192,
    )

    result = tracker.process(audio)
    pitch_hz = result["pitch_hz"]
    confidence = result["confidence"]
    timestamps = result["timestamps_s"]

    voiced = confidence >= 0.3
    n_voiced = int(voiced.sum())
    print(
        f"\n{len(pitch_hz)} frames, {n_voiced} voiced "
        f"({100*n_voiced/max(len(pitch_hz), 1):.1f}%)"
    )

    print("\nfirst 10 voiced frames:")
    voiced_idx = np.where(voiced)[0][:10]
    for i in voiced_idx:
        print(f"  {timestamps[i]:6.3f}s  {pitch_hz[i]:6.1f} Hz  "
              f"conf={confidence[i]:.2f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
