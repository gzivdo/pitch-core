# pitch-core-onnx

ONNX backends for [`pitch-core`](https://crates.io/crates/pitch-core):
CREPE, RMVPE, SwiftF0, FCPE (always on); PESTO (behind `pesto` feature).
Pulls in [`ort`](https://crates.io/crates/ort) (~50 MB binary download
by default). The crate ships **zero model weights** — every backend
takes a path to a user-supplied `.onnx` file.

## Getting model files

```bash
# From the workspace root, fetches the 4 permissive backends (~570 MB):
tools/download-models.sh

# Add PESTO (LGPL-3.0):
tools/download-models.sh --pesto
```

See [MODELS.md](../../MODELS.md) for per-backend details — sources,
licenses, and how to build/customize. Each upstream model carries its
own license; you must comply with each when redistributing a binary
that bundles or downloads them. PESTO is **opt-in** because its
weights inherit upstream **LGPL-3.0**.

## Authors

Created and maintained by **gzivdo** — design, architecture, research
direction, and code review.

Co-author (implementation): **Claude Opus 4.7** (Anthropic), under
gzivdo's direction. AI is not a copyright holder; copyright is held
entirely by the human maintainer (per US Copyright Office guidance,
Jan 2025). See [`NOTICE`](NOTICE) for the formal copyright line.
