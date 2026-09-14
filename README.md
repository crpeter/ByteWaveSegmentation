# ByteWave Segmentation Probe

A standalone iPhone app that checks the hardware compatibility of all four components in a candidate video-segmentation model. All model bytes are included; two large weight files are stored in parts. The setup command below restores them locally. The app needs no network access. It does not change ByteWave.

## Run

1. From the cloned repository, run `python3 restore_models.py`, then open `ByteWaveSegmentationProbe.xcodeproj` in Xcode 16 or newer.
2. Select the **ByteWaveSegmentationProbe** target → **Signing & Capabilities** → choose your development team. Change the bundle identifier if Xcode requests it.
3. Select your physical iPhone, with iOS 18 or newer, and run.
4. Choose **CPU + Neural Engine**, tap **Inspect models**, then **Share report** and keep the JSON file.
5. Repeat with **CPU + GPU**. Send back both JSON reports. **Automatic** is an optional third comparison.

Each pass compiles and loads the four models one at a time. Compilation may take a while; keep the app open. A model failure is recorded and the probe continues. The simulator is intentionally blocked because it cannot answer the phone's Neural Engine question.

## What this establishes

The JSON records the device, OS, model revision, load/compile errors, and the preferred and supported compute devices reported for each operation by Apple's [MLComputePlan](https://developer.apple.com/documentation/coreml/mlcomputeplan-1w21n). CPU + Neural Engine permits CPU fallback. A successful load alone does not prove meaningful Neural Engine use. Operation counts are not percentages of execution time.

**This is a compatibility probe, not a live tracking benchmark.** It runs no predictions and measures no tracking quality, playback FPS, runtime hardware utilization, or sustained thermal performance. Compile/load times are diagnostics, not frame-processing times. Actual placement and timing still require on-device profiling of an integrated tracking loop.

## Why this candidate needs checking

The [EdgeTAM paper](https://arxiv.org/html/2501.07256v1) reports roughly 16 FPS on an iPhone 15 Pro Max using CPU plus Neural Engine. That is encouraging evidence for this direction, not a promise of 30 FPS inside ByteWave.

The [official repository's Core ML example](https://github.com/facebookresearch/EdgeTAM/tree/main/coreml) exports the image encoder, prompt encoder, and mask decoder. Those components alone omit the temporal memory needed for full video tracking.

This probe instead bundles the separate [community four-model video export](https://huggingface.co/AhmadYarAI/EdgeTAM-CoreML-Video/tree/6bfdd4765e42508c7707566fff52e65add8b8e3a). Its documentation recommends CPU + GPU because some graph operations may fail with Neural Engine enabled. We therefore need to test the actual exported graphs on your phone before choosing or re-exporting the model. This is a candidate under evaluation, not a final model selection.

## What follows

Use the reports to locate unsupported graphs or CPU-heavy placement. Then implement and profile a complete temporal tracking loop, including memory updates, subject selection, seeking/reset behavior, and mask edges. Compare quality and sustained latency on real clips before connecting woven text.

The supplied `BackgroundRemovalSource.swift` currently counts/decodes the full clip and generates a mask texture slice for every frame before preparation finishes. The proposed replacement should produce masks as frames are requested, with a bounded working set and explicit timing. This probe does not implement that replacement yet.

## Included and verified

All four original `.mlpackage` bundles are unchanged, pinned to revision `6bfdd4765e42508c7707566fff52e65add8b8e3a`. Their checksums are in `Audit/download-manifest.json`; parsed model interfaces are in `Audit/model-inspection.json`. In that manifest, `models/` maps to this project's `Models/`; upstream text files map to `ThirdParty/`.

The model license, notice, and original model card are in `ThirdParty/`. The generated Swift app and project were statically reviewed; they have **not been compiled with Xcode or run on an iPhone** in this environment.
