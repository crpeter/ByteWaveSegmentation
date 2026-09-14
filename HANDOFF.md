# ByteWave segmentation: continuation handoff

## User goal and decisions

Cody Peter is building ByteWave, a local video editor. The goal is general-subject segmentation and temporal tracking as video frames are requested, eventually enabling woven text (text behind selected subjects). Avoid full-video mask precomputation. There is no rush: choose a sound architecture instead of minimizing implementation effort. Neural Engine acceleration is a hypothesis to evaluate, not an established outcome. Keep explanations short. Do not ask for the Metal renderer at this stage. Do not run iOS builds/tests/servers on Cody's behalf; Cody validates on his Mac and physical iPhone 17 Pro.

## What exists here

This repository contains a standalone iOS 18+ SwiftUI hardware-placement diagnostic and a new first-frame prediction screen, NOT a working video tracker or ByteWave integration. `SegmentationProbeApp.swift` compiles and loads four Core ML models sequentially and examines `MLComputePlan` under CPU + Neural Engine, CPU + GPU, or Automatic. It exports a JSON report with device/OS, failures, timings, and preferred/supported devices for model operations.

The user successfully built and ran that original placement screen on iPhone18,1, OS Version 27.0 (Build 24A435), on 2026-09-14. All four components loaded and produced plans in all three modes, without reported errors. `Audit/device-placement-summary.json` preserves counts and input hashes from the three supplied reports. CPU + Neural Engine preferred nonconstant operations: image 278 NE / 2 CPU; initializer 149 NE / 82 CPU / 31 unspecified; memory 249 NE / 1 unspecified; propagator 635 NE / 11 CPU. These are operation counts, not timing shares or measured hardware execution.

`FirstFrameProbe.swift` has now been built and run by the user. Three reports on iPhone15,2 / OS 26.5.2 (23F84) show nonempty masks and successful recorded tensor checks: CPU + Neural Engine mean prediction total 50.3 ms, CPU + GPU 122.7 ms, Automatic 51.0 ms; mask coverage approximately 3.71% in each. These are repeated first-frame predictions, not video FPS. NE used a slightly different point; GPU/Automatic shared the point. This is a different phone from the earlier iPhone18,1 placement reports. See `Audit/device-first-frame-summary.json` for report contents and hashes. The user initially saw an empty mask and an E5RT conv_transpose shape warning, then obtained a mask after moving the point; the warning's cause remains unresolved. Tracking quality, sustained frame rate, and runtime Neural Engine utilization remain unmeasured.

`Temporal/` now implements an owned source-controlled export/runtime, separate from the four community packages. The next step is Mac validation, not another first-frame iPhone probe. It pins original EdgeTAM source/checkpoint, keeps one conditioning memory plus six recent memories and sixteen pointers, puts rotary construction and validity masking inside the graph, and compares 20 consecutive frames against original dynamic-history PyTorch tracking. Only after that passes does it export and run a Core ML CPU comparison. No inference, tests, tracing or conversion have been run by the assistant. The code has only been statically reviewed. Read `Temporal/README.md` and `Temporal/CONTRACT.md`; do not claim the owned temporal models are generated, validated or integrated yet.

`Models/` and `ModelParts/` contain all model bytes. Two weight files exceeded the chat connector's upload-request limit and were split. Run `python3 restore_models.py` after cloning; it reconstructs them atomically and verifies all upstream file hashes. It makes no network requests. Then open `ByteWaveSegmentationProbe.xcodeproj`, choose a signing team, and run on a physical phone. Run CPU + Neural Engine and CPU + GPU separately; share both JSON reports for analysis. See README.md.

## Research findings that matter

- [EdgeTAM paper](https://arxiv.org/html/2501.07256v1): reports roughly 16 FPS on iPhone 15 Pro Max with CPU plus Neural Engine. This is not evidence of 30 FPS inside ByteWave.
- [Official Core ML example](https://github.com/facebookresearch/EdgeTAM/tree/main/coreml): exports image encoder, prompt encoder, mask decoder; omits temporal memory required for full tracking. Do not mistake per-frame image segmentation for temporal tracking.
- The bundled [community full-video export](https://huggingface.co/AhmadYarAI/EdgeTAM-CoreML-Video/tree/6bfdd4765e42508c7707566fff52e65add8b8e3a) supplies image encoder, initializer, memory encoder, propagator. Its model card recommends CPU + GPU because Neural Engine-enabled loading may fail for some graphs. This export is a candidate, not a final selection.
- Metal GPU ML execution and the dedicated Neural Engine are different execution paths. Unified memory does not guarantee zero transfer/synchronization overhead, arbitrary graph fusion, or real-time performance.

Model revision: `6bfdd4765e42508c7707566fff52e65add8b8e3a`. Licenses/model card: `ThirdParty/`. Original SHA256 checksums: `Audit/download-manifest.json`. Model schemas/operator inventory: `Audit/model-inspection.json`. The propagator's external memory assembly, attention bias, rotary inputs, and temporal state update semantics still need to be established from an authoritative runtime/export source. Do not invent them from tensor shapes.

## Existing ByteWave pipeline reviewed in the previous chat

Cody supplied `BackgroundRemovalSource.swift` as a pasted attachment. That app source is NOT in this repository. The following are review notes, not a substitute for its source when editing it:

- `prepare(clipURL:data:progress:completion:)` downsizes to a 360-pixel long edge and uses nominal FPS with a 30 FPS fallback.
- Cache key incorporates clip path/modification, target size, and `data.maskCacheSignature`.
- A full AVAssetReader decoding pass counts frames. A second pass generates masks.
- Allocates R8Unorm shared 2D texture arrays, chunks up to 2048 slices, one mask per frame; preparation completes after all frames are processed.
- Fresh `VNGenerateForegroundInstanceMaskRequest`/`VNImageRequestHandler` per frame. Selection follows the previous center, seeded from `selectedInstanceAnchor`, or uses all instances. This is not robust identity tracking.
- Scaled float masks are converted to UInt8 on CPU and written with `texture.replace`. No explicit compute selection was shown. Mapping uses frame index/nominal FPS; no explicit presentation-timestamp association was shown.

## Next work

1. Have Cody run the state checks and `Temporal/validate_export.py` locally on a short real clip. Request `status.json`, stage `report.json` on failure, or the first setup/export traceback. The required Mac commands are in `Temporal/README.md`; use a separate Python 3.11 environment. Do not run tests/inference/builds on his behalf.
2. Resolve concrete parity/export failures without relaxing thresholds to hide a defect. Current gates are explicit engineering policies, not previously measured results. Obtain representative visual review and retain the fixture results. The community contract remains unknown; the owned implementation does not claim to reconstruct it.
3. When both comparisons pass, integrate a source-matched Swift temporal runner for the new `BWTemporal*` packages and profile on iPhone. Keep the community packages and existing diagnostic usable until a validated replacement exists. The new export has different input names and no application-supplied rotary tensor; it is not a drop-in replacement.
4. Evaluate mask edges, hair, occlusion/reappearance, fast motion, tracking drift, latency, and sustained performance before integrating woven text.

The user authorized pushing this project to `crpeter/ByteWaveSegmentation`. The initial GitHub permission issue was fixed by granting repository access. No ByteWave production files have been changed. The previous downloadable ZIP failed to expand on macOS; use this repository instead.
