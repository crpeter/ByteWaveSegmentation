# ByteWave Segmentation Probe

A standalone iPhone app that checks the hardware compatibility of all four components in a candidate video-segmentation model and tests real first-frame subject predictions. All model bytes are included; two large weight files are stored in parts. The setup command below restores them locally. Model inference needs no network access; Photos may download a selected iCloud video. It does not change ByteWave.

## Next device check: predict a subject mask

**Continuation:** The user has now run first-frame predictions successfully in
all three modes. The next step is the owned temporal export and Mac comparison
in [Temporal/README.md](Temporal/README.md). Those models are not generated or
validated yet. Results from the first-frame runs are preserved in
`Audit/device-first-frame-summary.json`.

The initial placement checks and subsequent first-frame predictions passed their
recorded checks on the user's phones. The summaries are in `Audit/`. Instructions
below remain available for repeating the first-frame diagnostic.

1. Open the project and run on your physical iPhone as below.
2. Tap **Test a video frame** → **Choose video**. Start with a short clip whose
   subject is clearly visible near the beginning.
3. Tap the subject in the displayed frame, then **Predict mask** with
   **CPU + Neural Engine** selected. Keep the app open until it finishes.
4. Check that the blue overlay matches the selected subject. **Show mask** toggles
   it; tapping another point clears the previous result. Test an off-center subject
   and a portrait clip to check orientation and point mapping.
5. **Share prediction report**, then repeat **CPU + GPU** on the same frame/point.
   Send both reports and a screenshot of the overlay. A prediction failure is also
   saved in the report. Each mode keeps its latest report, overwriting earlier runs.

This executes the image encoder, initializer, and initial memory encoder: one
untimed warm-up plus three measured repetitions on the same frame. It shows a
thresholded mask and checks final output tensors for shape/type and finite values.
The reported `bestIoU` is the model's estimate, not measured ground-truth accuracy.
Model loading, preprocessing and individual prediction-call timings are separate.
These timings do not establish playback FPS or actual hardware utilization.

It decodes a frame near the start and records its actual timestamp. It retains
one preview, removes its temporary video import after extraction, and creates no
video-wide mask cache. This is still **not temporal tracking**. The published
conversion lacks the bank-building runtime needed to establish the exact
propagator inputs; see `Audit/temporal-contract.md` for the source audit and the
specific remaining contract requirements.

## Run

1. From the cloned repository, run `python3 restore_models.py`, then open `ByteWaveSegmentationProbe.xcodeproj` in Xcode 16 or newer.
2. Select the **ByteWaveSegmentationProbe** target → **Signing & Capabilities** → choose your development team. Change the bundle identifier if Xcode requests it.
3. Select your physical iPhone, with iOS 18 or newer, and run.
4. Choose **CPU + Neural Engine**, tap **Inspect models**, then **Share report** and keep the JSON file.
5. Repeat with **CPU + GPU**. Send back both JSON reports. **Automatic** is an optional third comparison.

Each pass compiles and loads the four models one at a time. Compilation may take a while; keep the app open. A model failure is recorded and the probe continues. The simulator is intentionally blocked because it cannot answer the phone's Neural Engine question.

## What this establishes

The JSON records the device, OS, model revision, load/compile errors, and the preferred and supported compute devices reported for each operation by Apple's [MLComputePlan](https://developer.apple.com/documentation/coreml/mlcomputeplan-1w21n). CPU + Neural Engine permits CPU fallback. A successful load alone does not prove meaningful Neural Engine use. Operation counts are not percentages of execution time.

**The Inspect models screen is a compatibility probe, not a live tracking benchmark.** That screen runs no predictions and measures no tracking quality, playback FPS, runtime hardware utilization, or sustained thermal performance. Compile/load times are diagnostics, not frame-processing times. The separate first-frame screen runs actual predictions as described above; full tracking still needs an integrated temporal loop.

## Why this candidate needs checking

The [EdgeTAM paper](https://arxiv.org/html/2501.07256v1) reports roughly 16 FPS on an iPhone 15 Pro Max using CPU plus Neural Engine. That is encouraging evidence for this direction, not a promise of 30 FPS inside ByteWave.

The [official repository's Core ML example](https://github.com/facebookresearch/EdgeTAM/tree/main/coreml) exports the image encoder, prompt encoder, and mask decoder. Those components alone omit the temporal memory needed for full video tracking.

This probe instead bundles the separate [community four-model video export](https://huggingface.co/AhmadYarAI/EdgeTAM-CoreML-Video/tree/6bfdd4765e42508c7707566fff52e65add8b8e3a). Its documentation recommends CPU + GPU because some graph operations may fail with Neural Engine enabled. We therefore need to test the actual exported graphs on your phone before choosing or re-exporting the model. This is a candidate under evaluation, not a final model selection.

## What follows

Use the reports to locate unsupported graphs or CPU-heavy placement. Then implement and profile a complete temporal tracking loop, including memory updates, subject selection, seeking/reset behavior, and mask edges. Compare quality and sustained latency on real clips before connecting woven text.

The supplied `BackgroundRemovalSource.swift` currently counts/decodes the full clip and generates a mask texture slice for every frame before preparation finishes. The proposed replacement should produce masks as frames are requested, with a bounded working set and explicit timing. This probe does not implement that replacement yet.

## Included and verified

All four original `.mlpackage` bundles are unchanged, pinned to revision `6bfdd4765e42508c7707566fff52e65add8b8e3a`. Their checksums are in `Audit/download-manifest.json`; parsed model interfaces are in `Audit/model-inspection.json`. In that manifest, `models/` maps to this project's `Models/`; upstream text files map to `ThirdParty/`.

The model license, notice, and original model card are in `ThirdParty/`. The user successfully built and ran both the placement probe and first-frame predictions on iPhone. The owned temporal export is **implemented but not yet executed or validated**. No builds or tests are run on the user's behalf in this environment.
