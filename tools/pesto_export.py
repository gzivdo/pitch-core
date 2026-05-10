#!/usr/bin/env python3
"""Export PESTO MIR-1K@48k to a single ONNX file with the HCQT
preprocessor baked in.

SPDX-License-Identifier: MIT OR Apache-2.0
Copyright 2026 gzivdo (this script).

NOTE: PESTO upstream (`SonyCSLParis/pesto`, `pesto-pitch` on PyPI) is
licensed under **LGPL-3.0**. This export script is MIT/Apache, but the
PESTO checkpoint it loads and the resulting ONNX inherit upstream
LGPL-3.0 by default (no separate weights license declared upstream).
Treat the exported `pesto_*.onnx` as LGPL-3.0 derivative when
redistributing.


PESTO ships its weights as a tiny PyTorch checkpoint (~534 KB) in the
`pesto-pitch` PyPI wheel. The ResNet1D encoder itself is only 130k
parameters, but at runtime PESTO instantiates an HCQT (harmonic CQT)
preprocessor with sample-rate-specific cosine filterbanks. To avoid
shipping a torch dependency at app runtime, we trace the whole stack
(audio → HCQT → encoder → activations + pitch + confidence) and export
the lot to a single ONNX graph. The HCQT kernels become a constant
~16 MB tensor inside the graph, which is why the resulting file is
~17 MB.

Output: $REPO/models/pesto_mir1k_g7_48k_960.onnx by default.
Naming: g7 = the gain-7 PESTO MIR-1K checkpoint; 48k = sample rate;
960 = 20 ms hop in samples (20e-3 * 48000).

Requires (will be auto-installed by download-models.sh if missing):
    torch ≥ 2.0
    pesto-pitch ≥ 2.0
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO_ROOT / "models" / "pesto_mir1k_g7_48k_960.onnx"
TARGET_SR = 48000
HOP_MS = 20.0  # 20 ms @ 48 kHz = 960 samples
SAMPLE_INPUT_SECONDS = 1.0  # any length ≥ HCQT window — only used for tracing


class PestoExportable(nn.Module):
    """Wraps the live PESTO model so its `forward()` has the fixed
    signature ONNX needs: `(audio,) -> (pitch_hz, confidence)`.

    The reference PESTO API takes optional kwargs (`sr=`, `convert_to_freq=`,
    `return_activations=`) which `torch.onnx.export` cannot trace. We
    pin the values that matter for our use case (48 kHz, frequency
    output, activations not needed) and discard the rest.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model
        # PESTO's reduction defaults to "alwa" — argmax-with-local-average.
        # Keep the default for parity with the published evaluation.

    def forward(self, audio: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # audio: (B, num_samples) mono float32 at TARGET_SR.
        # The model's built-in preprocessor was constructed with
        # sampling_rate=TARGET_SR, so passing sr=None re-uses that.
        preds, confidence, _vol, _act = self.model(
            audio,
            sr=None,
            convert_to_freq=True,   # output Hz, not semitones
            return_activations=True,
        )
        return preds, confidence


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n", maxsplit=1)[0])
    p.add_argument(
        "--output",
        "-o",
        type=Path,
        default=DEFAULT_OUT,
        help="output .onnx path (default: %(default)s)",
    )
    p.add_argument(
        "--checkpoint",
        default="mir-1k_g7",
        help="PESTO checkpoint name (built-in: mir-1k, mir-1k_g7) "
        "or path to a .ckpt file. Default: %(default)s",
    )
    p.add_argument("--opset", type=int, default=17)
    args = p.parse_args()

    # Lazy imports so --help works without torch / pesto installed.
    try:
        from pesto.loader import load_model
    except ImportError:
        print(
            "error: pesto-pitch is not installed in this Python environment.\n"
            "    Install with: pip install pesto-pitch torch",
            file=sys.stderr,
        )
        return 2

    print(f"loading PESTO {args.checkpoint!r} (step_size={HOP_MS} ms, sr={TARGET_SR})")
    pesto = load_model(args.checkpoint, step_size=HOP_MS, sampling_rate=TARGET_SR)
    pesto.eval()

    wrapped = PestoExportable(pesto).eval()

    # Trace input — length doesn't matter for the exported graph, the
    # actual input length will be a dynamic axis.
    n_samples = int(SAMPLE_INPUT_SECONDS * TARGET_SR)
    dummy = torch.randn(1, n_samples, dtype=torch.float32)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(f"exporting → {args.output} (opset {args.opset})")
    torch.onnx.export(
        wrapped,
        (dummy,),
        str(args.output),
        input_names=["audio"],
        output_names=["pitch_hz", "confidence"],
        # Both batch (B) and the per-sample length (T) vary at inference.
        # PESTO emits one frame per HOP_MS, so the output time dim scales
        # with the input.
        dynamic_axes={
            "audio": {0: "B", 1: "T"},
            "pitch_hz": {0: "B", 1: "Tout"},
            "confidence": {0: "B", 1: "Tout"},
        },
        opset_version=args.opset,
        do_constant_folding=True,
        # Stick to the legacy TorchScript exporter — pesto's HCQT path
        # uses a couple of constructs that don't yet trace cleanly under
        # the dynamo-based exporter (torch ≥ 2.10).
        dynamo=False,
    )

    size_mb = args.output.stat().st_size / 1024 / 1024
    print(f"ok ({size_mb:.1f} MiB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
