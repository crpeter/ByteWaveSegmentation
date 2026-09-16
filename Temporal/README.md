# Owned EdgeTAM temporal export

The first-frame iPhone probe works. The next dependency is a temporal export whose
runtime we can inspect and validate. This directory contains an owned implementation
from Meta's pinned source, a bounded memory manager, and a 20-frame comparison/export
command. It does not replace the currently bundled models or add tracking to the
iPhone app yet. Read `CONTRACT.md` for the exact interface.

No inference, tests, tracing or conversion were run by the assistant. Python syntax
and source wiring were reviewed only. The commands below are for local validation.

## Prepare once on your Mac

Use Python 3.11 and a separate environment. Full Core ML validation needs macOS 15+
(the generated models target iOS 18). Installation downloads Python dependencies
and the pinned public upstream source/checkpoint; the validation command itself
uses local files only. [Core ML Tools 9 supports PyTorch 2.7](https://github.com/apple/coremltools/releases/tag/9.0).

From the ByteWaveSegmentation repository:

```bash
python3.11 -m venv .venv-temporal
.venv-temporal/bin/python -m pip install -r Temporal/requirements.txt
git clone https://github.com/facebookresearch/EdgeTAM.git .upstream-edgetam
git -C .upstream-edgetam checkout 7711e012a30a2402c4eaab637bdb00a521302c91
```

If you already created either directory, reuse it. The loader checks source and
checkpoint identity and rejects modified upstream tracked files. No SAM2 package
installation or CUDA build is required.

## Run the state checks

```bash
.venv-temporal/bin/python -m unittest discover -s Temporal -p 'test_*.py'
```

They cover startup, conditioning-frame retention, eviction, timestamps, failed
commits, absent-object outputs and stale results after reset.

## Compare and export

Choose a short local clip with at least 20 frames and a subject visible at the
beginning. Point coordinates are normalized x/y from the top-left of the displayed
frame. The example point below is approximately the point in the user's dog probe;
adjust it if using another clip.

```bash
.venv-temporal/bin/python Temporal/validate_export.py \
  --upstream .upstream-edgetam \
  --video "/absolute/path/to/dog.mov" \
  --point 0.4912 0.4706 \
  --output .temporal-runs/dog-01
```

The command first compares the owned PyTorch runtime with original EdgeTAM, then
exports, then compares Core ML predictions. It can take time on CPU. It writes
`status.json`, stage reports, twenty frame/mask image sets, and four new model
packages under the output directory. It does not decode or cache the full video.
Each output directory must be new so previous diagnostics cannot be overwritten.

Send `status.json` and any failing stage's `report.json`, or the first traceback
if setup/export fails. Do not copy the generated packages into `Models/`: they
have a different contract and need a dedicated iPhone runtime after validation.

For the reference comparison without conversion, add `--reference-only` and use
a new output directory. That mode deliberately never marks device readiness.

## What passing means

Passing establishes agreement for this small fixture, including memory startup
and eviction. It does not establish tracking quality across videos, iPhone FPS,
Neural Engine execution, or production readiness. Next comes the source-matched
Swift runtime and sustained real-clip evaluation on device.
