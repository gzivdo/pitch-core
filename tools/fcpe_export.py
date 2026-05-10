"""Export the bundled torchfcpe model to ONNX for use by pitch-core-onnx.

SPDX-License-Identifier: MIT OR Apache-2.0
Copyright 2026 gzivdo (this script). torchfcpe upstream is MIT.

torchfcpe's `spawn_bundled_infer_model()` returns a ready-to-use module that
takes audio of any sample rate (resampled internally) and returns f0 in Hz.
We wrap it so the ONNX graph takes a flat 1-D audio tensor and a sample
rate, returning f0 + voicing-prob tensors with dynamic time axis.

Run on the host with torch+torchfcpe installed:

    pip install torch torchfcpe
    python tools/fcpe_export.py --out models/fcpe.onnx [--sr 16000]

(`download-models.sh` does this automatically inside a local venv.)

The exported ONNX I/O:
    inputs:
        audio    float32  [1, n_samples, 1]   raw mono audio @ `sr`
    outputs:
        f0_hz    float32  [1, n_frames, 1]    f0 in Hz (0 = unvoiced — model
                                              gates internally with threshold=0.006)

The internal hop is 160 samples @ 16 kHz = 10 ms — same hop as CREPE/RMVPE.

License notes: torchfcpe is MIT (CNChTu); the bundled model checkpoint
is MIT (per the project's `setup.py` classifier). Exported ONNX inherits
MIT — fully permissive, no LGPL/NC restrictions.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torchfcpe import spawn_bundled_infer_model


# ─── ONNX-tracer compat patch ────────────────────────────────────────────────
# torchfcpe's mel_extractor calls `torch.stft(..., return_complex=True)` which
# the legacy ONNX tracer rejects. We monkey-patch torch.stft to emit a real
# tensor `[..., 2]`, then immediately combine to magnitude via view_as_complex
# (ONNX opset ≥ 18 supports this). Downstream `.real`/`.imag` then trace fine.

_orig_stft = torch.stft


class _FakeComplex:
    """Quacks like a complex tensor for `.real` / `.imag` access only.

    The ONNX tracer records `obj.real` and `obj.imag` as real-tensor index
    ops on the underlying `[..., 2]` real-pair. The Python wrapper itself
    isn't traced — only the tensor ops it triggers.
    """
    __slots__ = ("_t",)

    def __init__(self, real_pair):
        self._t = real_pair  # shape [..., 2]

    @property
    def real(self):
        return self._t[..., 0]

    @property
    def imag(self):
        return self._t[..., 1]


def _stft_for_export(input, n_fft, hop_length=None, win_length=None,
                     window=None, center=True, pad_mode="reflect",
                     normalized=False, onesided=None, return_complex=None):
    # Force return_complex=False internally → real-tensor [..., 2] output.
    real_pair = _orig_stft(
        input, n_fft, hop_length, win_length, window,
        center, pad_mode, normalized, onesided, return_complex=False,
    )
    # Wrap in a fake-complex shim. torchfcpe uses only `.real`/`.imag`,
    # immediately combining via `sqrt(.real² + .imag²)` — those tensor ops
    # trace cleanly without any complex type ever entering the graph.
    return _FakeComplex(real_pair)


torch.stft = _stft_for_export


class _FcpeExportWrapper(torch.nn.Module):
    """Thin wrapper exposing only the (audio, sr) → (f0, uv) call surface."""

    def __init__(self, sr: int) -> None:
        super().__init__()
        self.model = spawn_bundled_infer_model(device="cpu")
        self.sr = sr

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        # audio: [B, n_samples, 1] float32.
        # Default threshold=0.006 — model gates low-confidence frames to f0=0.
        # We expose f0 only; Rust derives voicing from f0 > 0. This keeps the
        # ONNX graph simple (no uv interpolation that needs target_length).
        f0 = self.model.infer(
            audio,
            sr=self.sr,
            decoder_mode="local_argmax",
            threshold=0.006,
        )
        return f0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="models/fcpe.onnx")
    ap.add_argument("--sr", type=int, default=16000,
                    help="sample rate of audio fed at inference (default 16000)")
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model = _FcpeExportWrapper(sr=args.sr).eval()

    # Dummy input: 2 seconds of silence at the chosen sr.
    n = args.sr * 2
    dummy = torch.zeros(1, n, 1, dtype=torch.float32)

    # Sanity-check the wrapper before exporting.
    with torch.no_grad():
        f0 = model(dummy)
    print(f"[sanity] f0 shape  = {tuple(f0.shape)}", file=sys.stderr)
    print(f"[sanity] f0 range  = [{f0.min():.2f}, {f0.max():.2f}]", file=sys.stderr)

    # `dynamo=False` forces the legacy tracer — torchfcpe's mel extractor uses
    # data-dependent control flow (`if torch.min(y) < -1.`) which torch.export
    # rejects. The legacy tracer just records the path taken on the dummy
    # input, which is fine for export here.
    torch.onnx.export(
        model,
        (dummy,),
        str(out_path),
        input_names=["audio"],
        output_names=["f0_hz"],
        dynamic_axes={
            "audio":   {1: "n_samples"},
            "f0_hz":   {1: "n_frames"},
        },
        opset_version=args.opset,
        do_constant_folding=True,
        dynamo=False,
    )
    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"[ok] wrote {out_path}  ({size_mb:.1f} MB)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
