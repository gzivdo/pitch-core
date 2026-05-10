# Pitch detection algorithms — survey, evaluation, and improvement paths

This document is the algorithm-level companion to pitch-core's per-crate
READMEs. It covers (a) the practical strengths and weaknesses we
observed in the bundled backends, (b) what could be improved without
retraining, and (c) what an ideal modern realtime pitch detector would
look like if built from scratch.

> The numbers in this document come from a benchmark on three datasets —
> MIR-1K (1000 Chinese karaoke clips), Vocadito (40 multi-language solo
> vocals, CC-BY-4.0), and 12 real-world songs with separator-extracted
> vocals. See [BENCH.md](https://github.com/gzivdo/pitch-core/blob/main/BENCH.md)
> for reproduction instructions.

## What the benchmark revealed

After fixing center=False/True time-labelling alignment between
backends (swipe-rs 0.3.0 release notes and per-backend doc-comments in
pitch-core/-onnx), pairwise agreement matrices show two natural
clusters:

| group | members | shared characteristic |
|---|---|---|
| **A — general / DSP** | pyin, praat_ac, swiftf0, crepe, pesto, swipe-* | trained on diverse audio (speech + music + instruments) → averages out expressive vocal behavior |
| **B — vocal-tuned neural** | rmvpe, fcpe | trained predominantly on singing data → captures vibrato / glissando / breath transitions more aggressively |

Cross-backend agreement is **strongly data-dependent**. Same models,
same fix-set, different inputs:

| dataset | character | rmvpe ↔ crepe | rmvpe ↔ fcpe |
|---|---|---:|---:|
| MIR-1K (clean karaoke, 1000 clips) | steady pitch, low expression | **98.5%** | 98.2% |
| Vocadito (multi-language solo, 40 clips) | high expression | 45.3% | 96.8% |
| Real-world extracted (12 songs) | expression + separator artefacts | **18.9%** | 91.4% |

Two factors stack to produce the divergence:

1. **Expression complexity** drops cross-class agreement from ~98% on
   karaoke to ~45% on expressive solo (−53 pp)
2. **Real-world acoustic noise** drops it further to ~19% on
   separator-extracted vocals (another −26 pp)

**Group B (rmvpe ↔ fcpe) stays cohesive across all datasets** — even
on the noisiest data, RMVPE and FCPE agree at 91%. They were trained
on similar singing distributions and converged to similar
interpretations of expressive vocal.

This is not a bug. **Pairwise agreement between backends is a poor
accuracy proxy on real expressive vocal.** Two backends agreeing 95%
on clean karaoke can disagree 19% on real songs without either being
"wrong" — they're interpreting expression differently.

For tasks that decode pitch into discrete notes (transcription,
scoring), **the choice between A and B matters less than the
note-decoder quality** — see "Why note-level output is cleaner" below.

## Octave error rate — the hard accuracy signal

When pairwise agreement is unreliable, octave-error rate (% of voiced
frames where backends disagree by ≥ 600 cents = half octave) gives a
data-independent quality signal. Real f0 values rarely jump by half
an octave between adjacent frames; high OER means the backend is
systematically wrong somewhere.

| backend | clean karaoke | expressive solo | real-world | observation |
|---|---:|---:|---:|---|
| **rmvpe** | **0.0%** | **0.0%** | **0.1%** | gold standard for octave consistency |
| **fcpe** | **0.0%** | 0.1% | 0.8% | excellent, slight degradation on real |
| pyin | 0.0% | 0.0% | 0.0% | conservative voicing helps here |
| crepe | 0.9% | 0.7% | 2.3% | mostly fine; drift on real |
| swiftf0 | 1.0% | 0.8% | 3.3% | similar to CREPE |
| swipe | 1.7-1.9% | 2.3% | 3.8% | DSP-typical octave-up bias |
| pesto | 0.9% | **2.5%** | **6.0%** | degrades sharply on expressive/real |
| praat_ac | **13.3%** | **14.9%** | **20.7%** | classical autocorrelation — large octave-up bias |

Vocal-tuned neural (RMVPE/FCPE) wins octave stability by an order of
magnitude on every dataset. Praat-AC's 13-21% octave-up rate is the
classical autocorrelation signature; for non-research use it should
be paired with octave-fold post-processing (see "Without retraining"
below).

## Recommendations by use case

| use case | best choice | notes |
|---|---|---|
| Realtime UI tracker, clean studio vocal | SwiftF0 (LowLatency) | 22 ms latency, 90%+ inter-agreement everywhere |
| Realtime UI tracker, expressive solo | SwiftF0 or CREPE-tiny | both stay in group A, predictable |
| Realtime UI tracker, separator-extracted vocal | CREPE-tiny | avoid PESTO (OER 6%) and Praat-AC (OER 21%) on this material |
| Offline transcription, clean | any modern backend | pick by CPU budget |
| Offline transcription, expressive solo | RMVPE or FCPE | near-zero octave errors, vocal-tuned interpretation |
| Offline transcription, real-world | **ensemble** (CREPE + RMVPE + SwiftF0, median pitch + voting) | beats single best by ~1pp RPA, reduces OER ~9× vs CREPE alone |
| ML training labels | RMVPE + cross-check with PYin | disagreement flags ambiguous frames for review |

For tasks where the f0 contour feeds a downstream **note decoder**
(ROSVOT-style HMM/Viterbi with onset boundaries and lyrics priors),
backend choice matters far less — the note-level abstraction
absorbs frame-level noise. See "Why note-level output is cleaner".

## Improving group A (general / DSP)

### Without retraining

These can be applied today, no GPU required:

1. **Multi-scale aggregation** — run 2-3 backends and take per-frame
   median. Roughly ×2-3 CPU but closes ~50% of single-backend errors.
2. **Confidence recalibration** — current confidence outputs are
   miscalibrated on expressive vocal. A sigmoid temperature scalar
   (`sigmoid(logit/T)` with T fitted on a small validation set) gives
   3-17 pp better voicing F1 (depending on backend; PYin is the big
   winner with +17 pp). **Already implemented** in pitch-core as the
   `confidence-calibration` cargo feature (default on); per-backend T
   constants were fitted on Vocadito GT. Disable via
   `--no-default-features` to recover raw model confidence.
3. **Smaller hop for CNN backends** — CREPE/PESTO are hop-agnostic at
   inference. Halving the hop (10ms → 5ms) doubles CPU but tracks
   vibrato up to 7 Hz cleanly.
4. **Octave-fold post-process** — for backends with octave-up bias
   (notably SWIPE on harmonically-rich vocals), fold detected jumps
   when ≥2 reference backends agree on the lower octave. Drops
   octave-error rate from 5-15% to ≤2%.
5. **Cross-class voting for voicing** — majority-vote between 3+
   backends gives ~99% precision at ~85% recall — practically
   gold-standard.

### With retraining

If GPU resources are available:

6. **Vocal-specific fine-tuning** — fine-tune CREPE/SwiftF0/PESTO on
   singing data at LR=1e-5 for ~50k steps. Estimated +3-5pp RPA on
   vocal, octave errors halved. ~$200-500 cloud cost.
7. **Octave-class head** — re-train with two outputs: `octave`
   (8-class softmax) + `pitch_in_octave` (12*N bins within an octave).
   Final pitch = octave * 12 + pitch_in_octave. Decomposition removes
   50%+ of octave errors. Full retrain (~$1-2k cloud).
8. **Augmentation-heavy retrain** — vibrato injection, breathiness,
   separator-artefact simulation. Especially valuable for use-cases
   on extracted vocals from real-world tracks.

## Improving group B (vocal-tuned neural)

Their weakness is **specialization in a narrow space**. RMVPE was
trained predominantly on RWC-pop (Japanese pop). FCPE on DDSP-200K
(synthetic). Result:

- "Alien" interpretations on Western expressive vocal (Vocadito)
- Catch separator artefacts as pitch on noisy/extracted material
- Excellent within their domain (96.8% mutual agreement) but the
  domain is small

### Without retraining

1. **Domain detection + routing** — train a small classifier (MFCC +
   2-layer MLP) to detect "singing-clean" vs "singing-extracted" vs
   "speech" and route to the best backend. ~10k params, free CPU.
2. **Hybrid pitch+voicing** — RMVPE entangles voicing with pitch
   confidence. Take pitch from RMVPE/FCPE, voicing from PYin (better
   calibrated). Hybrid often beats either alone.

### With retraining

3. **Multi-language vocal training** — extend to English / Spanish /
   Portuguese / Russian (Vocadito + IDMT-SMT-bass + Tonas + PJS +
   personal collections). Tens of GPU-hours.
4. **Adversarial alignment with general models** — add CREPE-tiny as
   teacher with λ=0.1 KL divergence in loss. Pushes rmvpe-vs-crepe
   above 85% "for free", removes group-B isolation.
5. **Domain-adaptation adapter (LoRA-style)** — small ~1MB adapter
   for perceptive "domains" (singing/speech/instrumental). Cheaper
   than full retrain (~$50 cloud), allows on-the-fly switching.
6. **Voicing+pitch dual-head architecture** — re-train with two
   independent outputs. Currently RMVPE's voicing comes from
   pitch-confidence, which fails on short/quiet fragments.

## Why note-level output is cleaner than any frame-level pitch

Frame-level f0 is a **noisy raw signal** — 50-100 spurious events per
minute on real vocal: vibrato, breath, octave flips, edge-of-voice
glitches. At the frame level, every algorithm has 5-20% bad frames.

A note-decoder (e.g., ROSVOT, basic-pitch's note layer) projects f0
into discrete notes:

- 1 note spans 200-1500ms = 20-150 frames
- ~50% of these can be wrong; the median of the rest gives the
  correct note
- Onset boundaries from a separate detector act as "fences" — a
  pitch frame falling between notes is silently dropped
- HMM/Viterbi smoothing prevents 1-frame note flicker
- Lyrics alignment (when available) provides voicing prior
  independent of acoustic confidence

Result: frame-level 90% accuracy → note-level 97-99% accuracy. The
single biggest gain is **the projection from continuous pitch to
discrete notes** — it kills most noise in one step.

**Practical implication:** if your task allows note-level output
(transcription, scoring), don't try to improve frame-level pitch —
improve the note-decoder. If your task is frame-level (real-time
pitch trainer, pitch correction), frame-level accuracy still matters.

## What an ideal new realtime pitch detector would look like

This section is design-only — no implementation, no published model
backs it. Treat it as a coherent direction for future work.

### Goals

| metric | target | rationale |
|---|---|---|
| Latency | ≤30 ms | parity with SwiftF0 LowLatency (22 ms) |
| RPA on MIR-1K | ≥97% | matches FCPE/RMVPE on the standard benchmark |
| Octave error rate | <2% | vs 5-15% in current DSP / 1-3% in current CNN |
| Cross-class agreement | ≥90% with PYin/CREPE | closes the group-B isolation we observed |
| Model size | ≤2 MB | between SwiftF0 (400 KB) and FCPE (42 MB) |

### Architecture sketch

**Frontend** — multi-resolution mel + harmonic CQT (HCQT). Stack three
mel-spectrograms at hop {5, 10, 20} ms, plus a harmonic-stacked CQT
(6 octaves × 60 bins/octave × 4 harmonics). Bake all preprocessing
into the ONNX graph as fixed-weight Conv1d.

**Backbone** — 4 layers of depth-wise separable convolutions
(FCPE Lynx-Net style) for CPU-friendly main fabric, plus 1 layer of
linear attention on a 64-frame window for long-range temporal
consistency without quadratic cost. ~250k parameters total.

**Streaming-friendly causality** — train with right-context masking;
inference uses 30 ms right-context, 10 ms hop. Per-frame inference
~1 ms on a modern CPU core.

**Multi-task output heads** — five parallel outputs:

1. `pitch_in_octave` — 60 bins, 20¢ resolution within one octave
2. `octave` — 7-class softmax (C2..C8), explicit decomposition that
   eliminates the dominant octave-error class
3. `voicing` — binary BCE, **independent** of pitch (fixes the
   pitch-confidence-as-voicing entanglement of RMVPE)
4. `vibrato_rate` — 4-class auxiliary task that improves
   regularization and temporal smoothness
5. `confidence` — scalar sigmoid, calibrated on a held-out set
   (not entangled with pitch class probability)

Final pitch = `octave * 12 + (pitch_in_octave / 60 * 12)` in MIDI.

### Loss

```
L = L_pitch + L_octave + L_voicing
    + λ_vib * L_vibrato
    + λ_adv * L_adversarial(model, CREPE-tiny)

λ_adv = 0.1 — KL divergence with a general-purpose teacher,
              prevents group-B-style isolation from the rest of
              the ecosystem.
```

### What's actually novel

| feature | in prior models | in this design |
|---|---|---|
| Explicit octave-class head | no (CREPE/RMVPE/FCPE use flat 360-bin output) | **yes** |
| Adversarial cross-class consistency | no | **yes** (new contribution) |
| Multi-resolution input combining mel & HCQT at different hops | partial | **yes** |
| Vibrato-rate auxiliary task | no | **yes** |
| Streaming causal training with right-context masking | common in TTS, rare in pitch | **yes** |

### Honest scope

This is a "strong short paper" target — combination of known techniques
plus one novel idea (adversarial cross-class consistency). Not a
paradigm shift. Realistic outcomes:

- ~$2-4k cloud GPU + 2-4 weeks of engineering iteration
- High probability of ≥95% RPA on MIR-1K
- Medium probability of beating RMVPE on vocal-specific benchmarks
- Low risk of fundamental failure (architecture is conservative)

### When this would be worth doing

- ✅ A new MIT-licensed pitch model that frees pitch-core-onnx from
  PESTO's LGPL-3 dependency for the "best vocal" niche
- ✅ A purpose-built realtime trainer where ≤30 ms latency + low
  octave error are simultaneously required
- ❌ When an existing backend already meets the use-case (don't
  build a new model just because you can)

### When *not* to do it

If the goal is note-level transcription, the highest-leverage
improvement is **a better note-decoder on top of existing pitch**, not
a new pitch model. See the previous section.

## Practical no-training next step

For users wanting better results today, the highest-value engineering
work is a **post-process layer on top of pitch-core's existing
backends**:

1. Run 2-3 backends in parallel
2. Detect known error classes (octave-up, false-voicing, breath,
   glissando) by cross-backend disagreement
3. Apply per-class corrections via majority vote + Viterbi smoothing
4. Output a cleaner unified contour

This pattern (median pitch + majority voicing of CREPE + RMVPE +
SwiftF0) has been measured offline against Vocadito ground truth:
**+1 pp RPA vs single best backend (RMVPE), ~9× lower octave-error
rate vs CREPE alone**. Most striking on clips where one backend has
catastrophic octave errors (e.g., 11% OER) — voted out by the median.

We may add this as `pitch_core::PitchTracker::ensemble(...)` in a
future release if there's realtime demand. The 3× CPU cost fits a
modern multi-core CPU but is steep for embedded/WASM.
