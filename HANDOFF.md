# ByteWave segmentation: continuation handoff

## User goal and decisions

Cody Peter is building ByteWave, a local video editor. The goal is general-subject segmentation and temporal tracking as video frames are requested, eventually enabling woven text (text behind selected subjects). Avoid full-video mask precomputation. There is no rush: choose a sound architecture instead of minimizing implementation effort. Neural Engine acceleration is a hypothesis to evaluate, not an established outcome. Keep explanations short. Do not ask for the Metal renderer at this stage. Do not run iOS builds/tests/servers on Cody's behalf; Cody validates on his Mac and physical iPhone 17 Pro.

## What exists here

This repository contains a standalone iOS 18+ SwiftUI hardware-placement diagnostic and a new first-frame prediction screen, NOT a working video tracker or ByteWave integration. `SegmentationProbeApp.swift` compiles and loads four Core ML models sequentially and examines `MLComputePlan` under CPU + Neural Engine, CPU + GPU, or Automatic. It exports a JSON report with device/OS, failures, timings, and preferred/supported devices for model operations.

The user successfully built and ran that original placement screen on iPhone18,1, OS Version 27.0 (Build 24A435), on 2026-09-14. All four components loaded and produced plans in all three modes, without reported errors. `Audit/device-placement-summary.json` preserves counts and input hashes from the three supplied reports. CPU + Neural Engine preferred nonconstant operations: image 278 NE / 2 CPU; initializer 149 NE / 82 CPU / 31 unspecified; memory 249 NE / 1 unspecified; propagator 635 NE / 11 CPU. These are operation counts, not timing shares or measured hardware execution.

`FirstFrameProbe.swift` is the new, unbuilt device-validation step. Choose a video from Photos, tap a subject, and run image encoder → initializer → initial memory encoder. It displays a blue thresholded mask, checks final mask/memory tensors, and exports actual per-call timings (one untimed warm-up, three measured repetitions of the same frame). It uses a single oriented preview frame with an actual timestamp, cleans up the picker import, and creates no whole-video mask cache. No predictions from this new screen have been reported yet. Tracking quality, sustained frame rate, and runtime Neural Engine utilization remain unmeasured.

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

1. Help Cody build/run **Test a video frame**, resolve concrete errors, and inspect CPU + Neural Engine and CPU + GPU prediction reports plus the visible overlay. Check an off-center subject in portrait/landscape clips. Keep the same point when comparing modes.
2. Read `Audit/temporal-contract.md`. The public model inventory has no exporter/runtime. Original EdgeTAM memory assembly is understood, but it does not establish this community conversion's fixed-bank attention-bias/rotary construction. Obtain the exporter's bank-building source plus startup/full-bank parity fixtures, or implement an owned export/runtime from the original EdgeTAM source. Do not infer these rules from tensor dimensions. Nothing in the loading results currently requires abandoning Neural Engine evaluation.
3. Once that contract is established, implement and profile a complete live tracking loop with bounded memory, timestamps, subject selection, temporal state, seek/reset behavior, and real clip inputs.
4. Evaluate mask edges, hair, occlusion/reappearance, fast motion, tracking drift, latency, and sustained performance before integrating woven text.

The user authorized pushing this project to `crpeter/ByteWaveSegmentation`. The initial GitHub permission issue was fixed by granting repository access. No ByteWave production files have been changed. The previous downloadable ZIP failed to expand on macOS; use this repository instead.
