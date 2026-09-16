# ByteWave segmentation: continuation handoff

## Current step: requested video frames (implementation awaiting user build)

`OwnedVideoTracking.swift` adds a separate **Track a video** screen using the
existing verified `DeviceValidationData/memoryfp16` packages with CPU + GPU.
It shares the fixture schema and unchanged OwnedTemporalSession/state runtime;
normal exports, the fixed 20-frame comparison and the community probe remain.
The loader checks diagnostic/source metadata, exact package file set and hashes,
then all four loaded model interfaces before inference. No re-export is needed.

One actor owns the imported movie, AVAssetReader, current prepared frame and
session. It decodes the next presentation-ordered sample only when requested,
uses the exact decoded CMTime, converts the track presentation transform into
Core Image coordinates, renders sRGB 1024-square model input and an aspect-correct
preview. Point selection uses normalized top-left coordinates. The UI supports
initialization on the displayed frame, next frame, continuous demand-driven
tracking, pause after the in-flight frame, seek and restart. Seeking/reselection
starts fresh state and requires a new subject point. No frames are dropped to
catch a clock; no audio or real-time playback-rate claim. A separate real-time
renderer/scheduler is not implemented in this probe.

Inference is synchronous inside the actor. UI task cancellation and a generation
check across loading awaits prevent stale results after navigation/reset. On
prediction/decode failure the state is invalidated and stale UI pixels are hidden;
seek/restart recovers. The temporary import and compiled packages are released on
replacement/close. Only bounded 7/16 banks and the current image survive requests;
no per-video mask or decoded-frame cache. Diagnostic frame records are capped at
120, with incremental warm timing totals and no retained tensors. Reports include
source package provenance, actual timestamps, reset/segment counts, orientation,
mask fraction/presence score, model/decode/render timings and thermal snapshots.
There is no automatic quality/parity pass for an unseen video. Snapshot and native
prediction checkpoint files overwrite fixed names instead of accumulating files.

Assistant validation: static source/lifecycle/ownership review, project source
registration and diff checks only. No Swift compiler/Apple SDK is available here;
no tests, builds, decoding, inference or conversion were run. User validation is
next: Release iPhone 17 Pro build, dog clip visual orientation/subject/step/run,
then pause/seek/reselect and share **video tracking report** plus a mask screenshot.
A portrait clip with an off-center subject is useful for transform/point mapping.
Setup and report-sharing instructions belong in chat, not README.

API references checked: Apple [decoded sample ordering and EOF status](https://developer.apple.com/documentation/avfoundation/avassetreaderoutput/copynextsamplebuffer%28%29),
[reader time range](https://developer.apple.com/documentation/avfoundation/avassetreader/timerange),
and [track presentation transform](https://developer.apple.com/documentation/avfoundation/avassettrack/preferredtransform).
These support the decoder design, not device validation of this new path.

## iPhone 17 Pro comparison passed: original versus memoryfp16

Added `Temporal/prepare_attention_device.py`: read-only verification and copying,
no conversion or inference. Requires matching passed source run, CPU/GPU temporal
candidate evidence, full 76-pair GPU report and existing prepared original device
fixture. Verifies package/input/reference bytes, shapes/finiteness and source RGB
hashes, copies original reference tensors and BGRA inputs unchanged, and substitutes
only the diagnostic Propagator in a NEW folder. The original baseline stays intact.
Publishes fixture.json last with explicit diagnostic contract/variant, effective
precision policy and hashes of baseline fixture, temporal report and paired report.

Swift adds an Original / Memory FP16 candidate picker in the temporal probe.
The candidate folder is DeviceValidationData/memoryfp16; original is still baseline.
Only CPU-only and CPU+GPU are offered for the candidate; plan inspection stays on
the original. Fixture choice must match its diagnostic metadata. Only Propagator
may use the exact memoryfp16 diagnostic contract; all other component metadata,
interfaces, references and numerical gates retain existing validation. Reports
identify the variant, effective diagnostic precision policy and source provenance.
The serialized candidate inherited the original package's source precision field;
its diagnostic contract plus variant describes the explicit override. The normal
exporter and installed baseline remain unchanged. These Swift/preparation changes
have only static source/AST and diff review, no assistant tests or builds.

The user built and ran both variants successfully on iPhone 17 Pro (iPhone18,1),
iOS 27.0 (24A435), Release, CPU + GPU. All 20 frames pass in both runs with
identical input hashes and matching original reference provenance. Thermal states
are nominal and final banks are 7/16. Warm medians (frames 1–19): original versus
memoryfp16 encoder 21.711/21.911 ms, propagator 66.509/32.329 ms, prediction and
state 99.337/65.259 ms. Candidate latency reductions are 51.39% for propagation
and 34.31% for prediction/state. Both minimum mask IoUs are 0.996804; candidate
minimum pointer/memory cosines are 0.998931/0.996140. Full uploaded reports and
summary are in Audit/iphone17pro-memoryfp16-*.json. These are separate short
runs, original first; no sustained FPS or broader quality claim.

The reverse-order repeat also passed on the same device/OS/Release GPU setup:
candidate completed at 00:31:46Z, original at 00:32:10Z on 2026-09-16. Warm
original/candidate propagator medians are 66.763/32.351 ms, prediction/state
99.997/64.895 ms, encoder 21.967/21.792 ms. Reductions are 51.54% and 35.10%.
Fixture and input identities match the first runs; every per-frame numerical
comparison metric and bounded state exactly matches its same-variant first run.
Both pass all 20 frames with nominal thermals. Reports are archived as
Audit/iphone17pro-memoryfp16-repeat-*-report.json; comparison-summary.json
contains both orders. The speed benefit survives reversed run order. No more
repeats of this fixed fixture are needed. Carry memoryfp16 forward to requested-
video-frame operation and broader visual/sustained validation. Original normal
export remains selected until deliberate integration; this evidence does not
validate unseen clips or sustained playback.

## Mac pass: isolated memory-attention precision

The chunk256 rewrite is slower on Mac despite passing correctness, so the
original normal export remains selected. The next diagnostic `memoryfp16`
changes only the four validated large memory SDPA operations to FP16: casts of
existing Q/K/V and optional bias, fused attention, then restores FP32 output.
The initializer and all mask-decoder attention, normalization, projections,
learned weights and external interfaces keep their existing precision. This
explicitly tests whether memory SDPA alone can tolerate reduced precision; it
is NOT a claim that the earlier FP16 attention corruption was generally solved.
No scaling changes, key pruning, head changes, clamping or weakened gates.

`diagnose_propagator_attention.py --experiment memoryfp16` uses a new output
folder and the same unchanged CPU control plus full candidate CPU/GPU temporal
comparisons. Exact unrelated/new operation signature checks cover the casts,
mask source and attention dtypes. Nonfinite outputs or parity failure stop the
run. The paired benchmark accepts `--variant memoryfp16` only with matching
passed temporal evidence and package hashes. Device readiness stays false;
the user now passed all three 20-frame comparisons and 76 paired GPU checks.
Median Mac latency is 14.3839 ms original versus 13.7715 ms memoryfp16 (4.257%
lower); both order strata agree. GPU mask/pointer/memory minima are
0.996804/0.999023/0.996592, final banks 7/16. See
`Audit/mac-attention-memoryfp16-summary.json`. iPhone results are recorded above.
User commands remain in chat, not README.

## Completed Mac experiment: explicit FP32 memory attention

`Temporal/diagnose_propagator_attention.py` prepares an unchanged reconversion
control and a diagnostic `chunk256` propagator. The rewrite selects exactly four
FP32 SDPA operations with query [1,1,4096,256]: two self-attentions with 4096 keys
and no mask, and two cross-attentions with 3648 keys and the original broadcast
FP32 validity bias. Each becomes 16 query chunks of 256 rows, using FP32 QK^T,
scale 1/16, unchanged bias, softmax over ALL original keys, and FP32 weighted
values before concatenating queries. No key pruning, head change, learned-weight
change or precision reduction. This tests a different GPU implementation of the
same attention equation; floating-point accumulation can still differ.

The helper verifies unrelated operation signatures/constants, new operation
signatures/precision and external interfaces after conversion. The unchanged
control and candidate CPU/GPU each replay all 20 saved frames against independent
original PyTorch with existing thresholds and bounded state. The unchanged CPU
control also requires same-input allclose against the normal CPU model. Failures
stop remaining modes; subprocess logs and checkpoints are retained. Candidate
contract is `bytewave.propagator-attention.diagnostic.v1`, never device-ready.

The existing paired benchmark accepts `--variant chunk256` and requires its
passing CPU/GPU temporal evidence, verified source/package hashes, GPU compute
units, and no linear-inspection report. It compares original versus candidate
on identical CPU-generated state, four balanced-order pairs per propagated frame,
with checked warm-up and every output gated. Existing conv2 CLI defaults and its
legacy summary keys remain available. Shared replay helpers accept an explicit
contract/script; the old convolution diagnostic retains its required inspection.

The user passed all three full 20-frame comparisons and all 76 paired GPU
checks. Paired medians: original 14.3616 ms, chunk256 34.5261 ms (2.404 times
original latency); both call-order strata agree. No promotion. Archive:
`Audit/mac-attention-chunk256-summary.json`. No candidate iPhone timing exists.
Assistant work remains static review and uploaded-report parsing only. Normal
exporter, README, installed dog-09 model set and Swift app are unchanged. Chunk
score tensors are 4 MiB (self) or 3.5625 MiB (cross), but this is not a guarantee
of peak allocation: scheduling and buffer lifetimes remain backend-dependent.

## Latest profiling: iPhone 17 Pro GPU attention hotspot

The user supplied a second Instruments run with Shader Timeline enabled. Offline
XML analysis is archived in `Audit/iphone17pro-instruments-shader-summary.json`,
including source/export hashes, per-prediction timings and interval methodology.
All 41 prediction calls are present (20 encoder, 19 propagator, two initialization).
Excluding each model's first call, median Propagator prediction is 67.497 ms and
app GPU Active interval union is 62.119 ms. Encoder medians are 22.406/20.941 ms.
Two shaders, `sdpa_tile_fwd_8x8x8_noEdgeCheck (230)` and
`sdpa_tile_fwd_8x8x8_doEdgeCheck (234)`, each appear twice per warm Propagator call.
Their combined observed interval median is 43.616 ms; across 18 warm calls,
757.914 ms of 926.212 ms observed shader interval union belongs to this pair.
Shader samples do not cover all 1119.562 ms of GPU Active time. These are sampled
interval observations, not exact MIL-operation timings or additive CPU/GPU costs.

The four expensive attention occurrences are consistent with the source's two
memory-attention layers, each using self-attention and cross-attention. The XML
provides shader identities but no exact mapping to those four MIL operation names.
This redirects the next optimization toward memory attention; the unsuccessful
two-linear convolution rewrite stays closed. Preserve FP32 attention and existing
PyTorch gates until a specific equivalent attention candidate is validated. No
model/fixture changes, tests, builds, conversion or inference were performed by
the assistant. Next work is an isolated attention implementation experiment,
not more generic GPU placement estimates. The trace itself does not embed model
or fixture hashes, so dog-09 association is from the user's capture workflow.

## Latest result: encoder fanout temporal comparison

The user passed both 20-frame runs in `encoder-temporal-01` against original
PyTorch, using the 39-edge `encoder-fanout-01` candidate with the tracker on CPU.
CPU encoder minima: mask IoU 0.995740, pointer cosine 0.999027, memory cosine
0.996570. CPU_AND_NE encoder minima: 0.998153, 0.999490, 0.998537. Both retained
seven spatial entries and sixteen pointers. See
`Audit/mac-encoder-fanout-temporal-summary.json`. This is a Mac diagnostic pass,
not physical-device NE validation or observed hardware utilization.

The normal exporter now reproduces the exact 39 shared-activation replacements
(36 residuals and three saved-feature consumers), verifies their identities and
preserves convolution precision and the RGB image interface. Graph revision is
`dense-points-encoder-fanout.v1`; precision policy is
`mixed-encoder-late-conv33-fanout-and-fp32-attention-iou.v1`. Swift requires these
new identifiers and adds **Encoder NE + CPU tracker**, requesting CPU_AND_NE only
for ImageEncoder and recording each component's requested units. Original PyTorch
reference, learned weights, bounded state and comparison gates remain unchanged.
Historical diagnostics explicitly accept the previous dog-08 policy where needed.

Dog-09 has now passed the normal complete export comparison: referencePassed,
coremlPassed and readyForDeviceValidation are true, with the new graph/precision
identifiers. All 20 CPU frames passed; minimum mask IoU 0.995740, pointer cosine
0.999027, memory cosine 0.996570, final banks 7/16. The RGB-interface export
reproduces the diagnostic's CPU minima. Full supplied status/report and attachment
hash are in `Audit/mac-fanout-normal-validation-report.json`.

The user prepared and installed dog-09, then passed all 20 frames on physical
iPhone15,2 / OS 26.5.2 with both CPU only and Encoder NE + CPU tracker. Complete
reports are `Audit/owned-temporal-fanout-iphone-cpu-report.json` and
`Audit/owned-temporal-fanout-iphone-encoder-ne-report.json`. Both match fixture
80bfbde3da5d8d650a68d60a1f61e2da0050abbe576e0bc848da46d640298455 and source model
manifest 47b7b85eef755ff7a440c0017f232a82ac7213af622d15d30a04f3487e07dd67.
CPU minimum mask IoU/pointer cosine/memory cosine: 0.997512/0.998589/0.996618;
encoder-NE mode: 0.997159/0.998731/0.996138. All timestamps, input hashes and bank
states match; final banks are 7/16, thermal state nominal at start/end.

The encoder-NE mode requests CPU_AND_NE only for ImageEncoder, CPU_ONLY for all
tracker components. Excluding cold frame 0, median encoder call time is 157.89 ms
versus 104.04 ms CPU; median propagator is 208.88 versus 207.45 ms. This establishes
fixture correctness, not a speed improvement or observed hardware utilization.
Debug diagnostic timings exclude several pipeline costs and are not playback FPS.

Full **CPU + Neural Engine** also passed all 20 frames on the same physical phone
and dog-09 fixture. `Audit/owned-temporal-fanout-iphone-all-ne-report.json` preserves
the report. All four components requested CPU_AND_NE. Minimum mask IoU 0.996804,
pointer cosine 0.998240, memory cosine 0.992768; final state 7/16, nominal thermal
state. Source/fixture/frame/timestamp identities match the CPU and encoder-only
NE reports. Full NE-permitted correctness is now established for this fixture;
actual device execution and general segmentation quality remain unmeasured.

Excluding cold frame 0, median encoder call was 168.04 ms and propagator 392.50 ms;
median summed model-call time 564.02 ms versus 310.61 ms CPU only and 367.24 ms
encoder-NE/CPU-tracker. Propagator compile/load was 20.14 seconds. These debug
correctness-probe measurements establish no speed benefit or sustained FPS.

Both remaining dog-09 modes passed all 20 physical-phone frames. GPU minimum
mask IoU/pointer cosine/memory cosine: 0.996449/0.998735/0.996238; Automatic:
0.996802/0.996997/0.992943. Complete reports are the fanout iphone gpu/automatic
files in Audit. All five modes use identical model/fixture/frame/timestamp hashes
and bounded state; all finish nominal thermally. The modes-summary JSON records
metrics and timings, excluding cold frame zero. Median summed model-call times:
CPU 311 ms, encoder-NE 367 ms, full NE 564 ms, GPU 265 ms, Automatic 581 ms.
GPU is fastest in these supplied diagnostic runs, but prediction/state takes
1137 ms versus 265 ms model calls. Do not treat these as playback FPS or measured
NE utilization. Earlier reports do not record build configuration or Metal
validation settings, so repeatable Release measurement is the next concrete need.

Added additive report instrumentation `session-stages.v1`: per-component input
preparation, output copy/validation, pre-prediction checkpoint/logging hook, plus
state pack and commit times. Model timings retain their existing boundaries;
stages exclude model calls and do not claim to cover all session overhead.
`swiftDebugCompilation` records the DEBUG compilation condition; the checked-in
project uses -Onone for Debug and -O for Release. Report decoding keeps the new
frame stage field optional for old reports. No model, state, output-copy logic,
correctness gate or checkpoint behavior is changed. Assistant static review only;
no builds/tests/model execution.

The user clarified that these recent runs are on their **iPhone 14 Pro**, not
their newer 17 Pro; hardware iPhone15,2 is consistent across the owned fixtures.
Preserve the older-device baseline and do not label these as 17 Pro measurements.
The new GPU and CPU + NE reports both pass all 20 frames and explicitly record
swiftDebugCompilation=false and session-stages.v1. Same dog-09 fixture/model/frame
hashes and final banks; nominal thermal state. Reports and attachment hashes are
archived in the iphone14pro-release Audit files and summary. Debugger/Metal
validation settings are not recorded, so do not claim independent verification.

Release medians excluding frame zero: GPU encoder 69.52 ms, propagator 184.78 ms,
summed model calls 254.37 ms, whole session 281.67 ms. NE encoder 157.08 ms,
propagator 384.18 ms, model calls 542.60 ms, session 563.16 ms. Per-frame session
minus model-call medians are 26.98 ms GPU and 20.75 ms NE; instrumentation leaves
only about 0.04/0.02 ms median unaccounted. Release largely removes the earlier
Swift diagnostic overhead. Models, especially propagation, now dominate; do not
optimize tensor handling as though it explains most remaining cost. These are
fixed-fixture measurements, not sustained video playback or observed NE activity.

The user completed both Release modes on **iPhone 17 Pro**, hardware iPhone18,1,
OS 27.0 (24A435). Both pass all 20 frames with DEBUG disabled, matching dog-09
model/fixture/frame/timestamp identities, final banks 7/16 and nominal thermal
state. GPU minimum mask/pointer/memory metrics: 0.996804/0.998976/0.995975;
NE: 0.996449/0.998667/0.992680. Full reports and a summary with attachment hashes
are in the iphone17pro-release Audit files. Hardware AND OS differ from the 14 Pro
(OS 26.5.2); do not attribute the entire change to hardware alone.

17 Pro median encoder/propagator/session times, excluding cold frame zero:
GPU 22.04/66.75/100.09 ms; NE 81.46/344.98/439.47 ms. Median summed model calls
88.82/426.21 ms; non-model session overhead 11.20/12.71 ms. GPU is substantially
faster for this export. Propagation is the largest remaining cost; these are
fixed-fixture timings, not playback FPS or observed Neural Engine utilization.

Added **Inspect GPU/NE plans** on the owned Temporal comparison screen. It checks
the same fixture/model hashes and model metadata, compiles ImageEncoder and
Propagator, and records both CPU_AND_GPU and CPU_AND_NE plans in one action.
It does not run predictions, change model precision, or repeat the correctness
fixture. New schema bytewave.temporal-compute-plan.v1 records all nonconstant
operations with output names, paths, preferred/supported devices and optional
estimated relative costs, per-plan counts/cost totals and errors. Missing costs
stay unreported; estimates are not runtime device usage or measured milliseconds.
Separate atomic temporal-last-plan.json checkpoints preserve load failures/native
abort location, and **Share plan report** exports a unique final report. Existing
prediction function and gates remain unchanged. The user built and completed this inspection on iPhone 17 Pro; all four plans loaded.

The owned 17 Pro plan matches the dog-09 fixture and manifest. Its selected
operations and source attachment hash are archived in
`Audit/owned-temporal-iphone17pro-compute-plan-summary.json`. Encoder GPU prefers
all 489 nonconstant operations; encoder CPU_AND_NE prefers 406 CPU / 83 NE, with
95.82% of estimated relative cost assigned to CPU-preferred operations.
Propagator GPU prefers 704 GPU / 18 CPU operations. The two linear outputs
`linear_9_cast_fp16` and `linear_19_cast_fp16` together account for 46.12% of
estimated relative cost; adding `linear_74_cast_fp16` and `linear_76_cast_fp16`
brings this to 57.65%. Propagator CPU_AND_NE prefers 489 NE / 181 CPU / 52
unreported, but supplies NO cost estimates. Counts and estimates do not measure
runtime hardware usage or establish an optimization speedup.

The user completed the static shape inspection; the parsed terminal summary and
attachment hash are in `Audit/mac-propagator-linear-inspection-summary.json`.
The two largest operations are memory-attention layers 0/1 linear2: FP16 input
[1,4096,2048], weights [256,2048], bias [256], output [1,4096,256]. Each has
2,147,483,648 dense multiply-accumulates. The next two are memory-encoder fuser
layers 0/1 pwconv2, with channels-last input [1,64,64,1024]; they are not spatial
perceiver layers. Their CPU_AND_NE plan prefers CPU.

Next experiment: `Temporal/diagnose_propagator_convs.py` reads the inspected
serialized propagator and creates an unchanged conversion control plus `conv2`.
Only the two memory-attention linear2 operations become FP16 1x1 convolutions:
transpose/reshape to [1,2048,64,64], identical flattened weight bytes and bias,
then restore [1,4096,256]. `propagator_convs.py` guards exact source parameter
names, shapes and precision, unchanged unrelated operation signatures/constants,
replacement weight hashes and external interfaces. No normal exporter, app,
precision policy, reference or correctness gate changes. Different accumulation
and placement can still change numerical outputs or performance.

The user-run diagnostic checks an unchanged reconversion on CPU against the
original CPU propagator on identical inputs (all outputs allclose 1e-3), then
validates unchanged CPU and candidate CPU/GPU/NE runs against the independent
original PyTorch reference for all 20 saved frames. Non-propagator components
stay CPU_ONLY; state is committed only after parity passes, with final banks 7/16.
CPU failure stops accelerated runs. Workers have separate logs, timeouts and
pre-call checkpoints. Mac call times exclude cold propagation and are diagnostic;
no paired GPU/NE performance comparison or iPhone speedup is established.
Packages have a diagnostic contract and readyForDeviceValidation stays false.
Assistant review is AST/JSON parsing, static API/source review and diff checks
only. The user has now passed all four 20-frame comparisons in
propagator-convs-01. Selected supplied fields are archived in
`Audit/mac-propagator-convs-summary.json`. Minimum mask/pointer/memory metrics:
conv2 CPU 0.996448/0.998902/0.996327; GPU 0.996804/0.999025/0.996556;
NE 0.995736/0.998918/0.994489. All final banks 7/16. Unchanged CPU also passes,
and its latest identical-input control has zero absolute error for every output.
Unpaired warm Mac propagator medians: unchanged CPU 96.99 ms, conv2 CPU 101.32 ms,
conv2 GPU 27.49 ms, conv2 NE 179.95 ms. There is no original GPU/NE timing in this
experiment; do not compare the Mac candidate to iPhone baseline timings.

Next: `Temporal/benchmark_propagator_convs.py` reuses the passed packages without
conversion. It verifies source/package/report hashes, drives the saved frames
with the original CPU state, and gives original and conv2 propagators identical
inputs under the same requested compute units (GPU by default). One warm-up per
model is checked and excluded; four pairs at each of 19 propagated frames use
balanced alternating order. All timed outputs pass the unchanged Core ML parity
gates against same-input original CPU. Input hashes are checked after each frame;
state commits only after checks pass. Report includes raw timings/checks, order
strata and a compact summary. This is a co-resident repeated-input Mac benchmark,
not a new temporal-feedback validation, device pass, hardware-utilization reading
or sustained playback benchmark. No model promotion or installed fixture change.
The user completed the paired Mac GPU run: all 76 pairs passed their accuracy
checks. Median original 14.4065 ms, conv2 14.4707 ms; candidate is 0.0642 ms
(0.45%) slower by median, with no useful gain demonstrated. Balanced-order
strata also provide no consistent improvement. These data do not establish a
statistically significant regression or any iPhone effect. Selected supplied
fields are in `Audit/mac-propagator-paired-gpu-summary.json`. Keep the original
normal export and device fixture; do not promote conv2 for GPU performance.

The paired Mac CPU_AND_NE run also completed with all 76 pairs passing accuracy.
Original median 176.5897 ms, conv2 177.0624 ms: no useful latency reduction
(-0.2677%). Both call-order strata remain close. Two ANE compiler failure messages
appeared after the Summary line in the pasted console output; they do not identify
a model or partition, and log ordering does not establish when the failures
occurred. Do not infer successful Neural Engine execution or specific CPU
fallback from numerical success. Selected report fields and exact messages are
in `Audit/mac-propagator-paired-ne-summary.json`.

Close the two-linear convolution experiment: no useful Mac GPU or CPU_AND_NE
benefit demonstrated. Retain the original normal export/dog-09 device fixture.
Do not promote conv2, repeat the same benchmark, or infer device outcomes from
Mac results. Compiler relative-cost estimates did not predict a useful benefit
from this operator substitution; there is no proof of which kernels were used.

Next is an Instruments capture of the ORIGINAL model on physical iPhone 17 Pro,
Release, CPU + GPU. Use the Core ML template and GPU instrument, run the existing
20-frame temporal comparison, then inspect a warm Propagator prediction's Activity,
Data and Compute tracks against GPU activity. Core ML compute requests can be
asynchronous; request intervals alone are not actual GPU occupancy. This is
runtime profiling of the known fixture, not another correctness rerun or a
sustained video benchmark. Native Core ML tracks already identify model calls,
so no app/model/signpost changes are needed for this first capture. Ask for the
model-level aggregation and an expanded warm Propagator timeline screenshot;
keep a saved trace for deeper follow-up. Workflow verified against Apple's
https://developer.apple.com/videos/play/wwdc2022/10027/ .
Assistant runs no builds, benchmarks/tests/inference. Commands and report-reading
steps remain in chat, never README.

## User goal and decisions

Cody Peter is building ByteWave, a local video editor. The goal is general-subject segmentation and temporal tracking as video frames are requested, eventually enabling woven text (text behind selected subjects). Avoid full-video mask precomputation. There is no rush: choose a sound architecture instead of minimizing implementation effort. Neural Engine acceleration is a hypothesis to evaluate, not an established outcome. Keep explanations short. Do not ask for the Metal renderer at this stage. Do not run iOS builds/tests/servers on Cody's behalf; Cody validates on his Mac and, for convenience, his physical iPhone 14 Pro (iPhone15,2); he also has a newer iPhone 17 Pro. Attribute results by report hardware and explicit user confirmation.

## What exists here

This repository contains a standalone iOS 18+ SwiftUI hardware-placement diagnostic and a new first-frame prediction screen, not ByteWave integration. A separate owned 20-frame temporal device comparison has now been built and passed CPU-only on a physical iPhone; accelerated iPhone modes remain pending. `SegmentationProbeApp.swift` compiles and loads four Core ML models sequentially and examines `MLComputePlan` under CPU + Neural Engine, CPU + GPU, or Automatic. It exports a JSON report with device/OS, failures, timings, and preferred/supported devices for model operations.

The user successfully built and ran that original placement screen on iPhone18,1, OS Version 27.0 (Build 24A435), on 2026-09-14. All four components loaded and produced plans in all three modes, without reported errors. `Audit/device-placement-summary.json` preserves counts and input hashes from the three supplied reports. CPU + Neural Engine preferred nonconstant operations: image 278 NE / 2 CPU; initializer 149 NE / 82 CPU / 31 unspecified; memory 249 NE / 1 unspecified; propagator 635 NE / 11 CPU. These are operation counts, not timing shares or measured hardware execution.

`FirstFrameProbe.swift` has now been built and run by the user. Three reports on iPhone15,2 / OS 26.5.2 (23F84) show nonempty masks and successful recorded tensor checks: CPU + Neural Engine mean prediction total 50.3 ms, CPU + GPU 122.7 ms, Automatic 51.0 ms; mask coverage approximately 3.71% in each. These are repeated first-frame predictions, not video FPS. NE used a slightly different point; GPU/Automatic shared the point. This is a different phone from the earlier iPhone18,1 placement reports. See `Audit/device-first-frame-summary.json` for report contents and hashes. The user initially saw an empty mask and an E5RT conv_transpose shape warning, then obtained a mask after moving the point; the warning's cause remains unresolved. Tracking quality, sustained frame rate, and runtime Neural Engine utilization remain unmeasured.

`Temporal/` implements an owned source-controlled export/runtime, separate from the community packages. It pins original EdgeTAM source/checkpoint, keeps one conditioning memory plus six recent memories and sixteen pointers, and puts rotary construction and validity masking inside the graph. The user passed all six state tests and the 20-frame PyTorch comparison on Mac, then exported all four v1 packages. Setup required PyAV 15 wheels; wrapper fixes extracted the memory-position tensor and gave rotary buffers separate identities for tracing. No inference, tests, tracing or conversion have been run by the assistant.

The user's v1 Core ML comparison failed at initialization. A controlled one-frame sweep isolated a successful precision policy: retaining `matmul`, `softmax`, and `scaled_dot_product_attention` in FP32 while converting other eligible operations to FP16. Low/high mask IoU against identical-input PyTorch was 0.999599/0.999925; pointer cosine was 0.999997. FP32 interfaces alone and FP32 normalization did not resolve the failure. The original FP16 model produced non-finite masks on some runs and finite but incorrect masks on another. Full sweep evidence and its attachment hash are in `Audit/mac-initializer-precision-summary.json`. This identifies a working operation group, not a single faulty kernel.

Contract v2 applies that attention policy to all four components with float32 tensor interfaces (labels remain int32), records excluded operations in the export manifest, and checks component outputs before they enter the next model. In the user's `dog-05` run, frames 0 and 1 passed Core ML parity. Frame 2 had mask IoU 1.0 and memory cosine 0.996503, but pointer cosine 0.978729 failed the unchanged 0.99 gate. No non-finite outputs were reported. The full Core ML comparison has not passed. `Temporal/diagnose_temporal.py` now reuses those v2 packages to compare each component against PyTorch on the same actual inputs for three frames, recording original independent-history comparisons and PyTorch mask-selection scores. This diagnostic has not been run by the assistant; it should distinguish local conversion error from accumulated state/input differences before another model change.

The user completed that three-frame same-input diagnostic (`Audit/mac-temporal-pointer-summary.json`). Frame 2 propagator pointer cosine is 0.999997 on identical inputs, versus 0.978729 end-to-end. The encoder raw-feature cosine is approximately 0.99858 on all three frames; same-input initializer and memory components agree much more closely. Same-input and independent-reference PyTorch both select mask index 1 on frame 2. Thus the pointer discrepancy is inherited from input/state differences, with encoder conversion the strongest observed candidate cause; this is not yet causal proof. The next control is `diagnose_temporal.py --torch-image-encoder --frames 20`: substitute only PyTorch image features from the start, retaining Core ML masks, pointers, and memories. This changes no models or parity thresholds and never marks device readiness.

The encoder control initially aborted in native PyTorch GELU with a GIL error. With PyTorch intra-op/inter-op threads set to one, the user completed all 20 frames (`Audit/mac-encoder-control-summary.json`). Seven zero-based frames failed: 2, 3, 4, 7, 8, 13, 19. Encoder replacement alone is therefore insufficient. Local same-input propagator pointer agreement fell sharply at frames 2, 7, 8, 13, with near-tied PyTorch mask-quality scores. Reduced-precision score ranking is a new hypothesis; the Core ML selected index was not directly recorded. Current export policy `fp16-with-fp32-attention-and-iou.v1` retains the attention fix and adds FP32 IoU-head computation plus score slicing/reduction through argmax. It verifies that scope/path matching occurred and records the protected operations. No selection rule or comparison threshold changes. The full validator now uses the single-thread settings too. Next is a fresh complete export/comparison, not another encoder control; no new policy inference/export has been run by the assistant.

The user's `dog-06` full run still failed at frame 2 with essentially unchanged pointer cosine (0.978729). However, repeating the encoder control against these attention-plus-IoU packages passed all 20 frames: minimum mask IoU 0.999355, pointer cosine 0.999816, memory cosine 0.999107. This combined result is archived in `Audit/mac-combined-encoder-control-summary.json`. It validates this fixture's Core ML tracking components when driven by PyTorch encoder features, not a complete Core ML pipeline. The next `diagnose_temporal.py --fp32-image-encoder --frames 20` control exports only a diagnostic FP32 Core ML encoder and uses it with the existing tracker models. It remains explicitly diagnostic, records override package hashes, and does not select FP32 as the final acceleration policy. The assistant has only statically checked that new control.

The user's FP32 Core ML encoder control completed and passed all 20 frames on CPU_ONLY (`Audit/mac-coreml-fp32-encoder-summary.json`): minimum mask IoU 0.999155, pointer cosine 0.999876, memory cosine 0.998751. All candidate components and stored state now come from Core ML in this successful diagnostic. This establishes a complete Core ML accuracy baseline for this fixture, but the regular exported set still contains the failing reduced-precision encoder and device readiness remains false. Next, `diagnose_temporal.py --fp16-conv-image-encoder` exports one diagnostic encoder starting from FP32 and lowering only convolution operations (including their inputs/weights) to FP16, retaining all other operations in FP32. This narrows the precision question by operation group; it does not claim to isolate a specific layer or establish acceleration. It reuses the same dog-06 tracker models, records selected convolution operations and package hashes, and preserves comparison thresholds. No inference/conversion was run by the assistant.

The user's FP16-convolution encoder control completed all 20 frames with 132 convolutions selected. Zero-based frames 2 and 19 failed pointer cosine (0.989646 and 0.987672); minimum mask IoU remained 0.996094 and minimum memory cosine 0.993182. Evidence is archived in `Audit/mac-coreml-fp16-conv-summary.json`. Thus lowering encoder convolutions (including inputs/weights) is sufficient to reintroduce the discrepancy relative to the passing FP32 baseline. This does not establish a single faulty layer. Next are complementary controls retaining convolution indices [0,66) or [66,132) in FP32, with other convolutions in FP16 and all non-convolution operations in FP32. The diagnostic now accepts `--fp32-conv-range START END` with required `--expect-encoder-convs 132`, records indices/module scopes and selected/retained operations, and fails if the expected count or range is not matched. These are diagnostic graph-order ranges, not a final model policy; both controls use the full 20 frames and unchanged gates. No model work or tests were executed by the assistant.

The complementary convolution controls completed (`Audit/mac-coreml-conv-halves-summary.json`). Retaining the later range [66,132) in FP32 passed all 20 frames (minimum mask IoU 0.997782, pointer cosine 0.998420); retaining only [0,66) failed zero-based frames 2,3,4,19. The later half is the next region to examine, without assuming there is a single sensitive layer or monotonic error behavior. The existing diagnostic supports the next two controls: retain [66,99) or [99,132) in FP32, other convolutions FP16, non-convolution operations FP32. Use the same dog-06 models, expected convolution count 132, and full 20-frame comparison. No tooling code changes were needed for this step.

Both quarter-range controls passed all 20 frames (`Audit/mac-coreml-conv-quarters-summary.json`). Retaining [99,132) gave minimum pointer cosine 0.998431 versus 0.993769 for [66,99); its minimum mask IoU was 0.997046 and memory cosine 0.994446. Further bisection on this fixture is deferred. The regular exporter now uses policy `mixed-encoder-late-conv33-and-fp32-attention-iou.v1`: encoder non-convolution operations and the exact 33 tested convolution module paths stay FP32; other encoder convolutions are FP16. Tracker precision remains attention-plus-IoU FP32. `encoder_precision.py` records semantic paths derived from the diagnostic; export verifies the total 132 convolutions and all 33 protected matches, then records per-convolution precision. This implements the successful diagnostic policy as a reproducible complete set, pending a fresh normal validation run. The diagnostic can still read the prior dog-06 policy explicitly for reproducibility. No comparisons, conversion, inference or tests were run by the assistant; only static source/data checks. Next: normal `validate_export.py` into fresh dog-07, then inspect status, Core ML report and saved masks before iPhone work.

The user completed the regular dog-07 validation: referencePassed/coremlPassed/readyForDeviceValidation are all true. The full Core ML report passed all 20 frames; minimum mask IoU 0.997046, pointer cosine 0.998431, memory cosine 0.994446. Evidence and the pasted status are in `Audit/mac-temporal-validation-summary.json`. This passes the Mac export gate for the selected policy; iPhone behavior remains unmeasured.

The next device probe is now implemented in `OwnedTemporalRuntime.swift` and `OwnedTemporalProbe.swift`, with source/resource membership added to the existing Xcode project. It runs a fixed 20-frame fixture through a source-matched Swift state loop, retaining one conditioning + six recent spatial memories and one conditioning + fifteen recent pointers. CMTime timestamps, reset tokens, strict model metadata/schema checks and validation before state commit are included. CPU-only, CPU + Neural Engine, CPU + GPU and Automatic modes compare low-mask/pointer/memory tensors against Mac Core ML CPU replay references, check all outputs for finite values, and show the current frame/mask. High-resolution masks are checked for shape/finiteness but not numerically compared; the report says so. Model-call timings exclude copies/checks/packing and are explicitly not sustained FPS or hardware-utilization measurements. This is a fixed-fixture device correctness probe, not arbitrary-video playback or ByteWave integration.

`Temporal/prepare_device.py` is a user-run Mac preparation step. It requires a passed current-policy run, verifies model hashes and identical saved frame identities, replays only the 20 Core ML CPU frames to save reference tensors, and requires replay mask IoU >= 0.999 against saved candidates. It copies model packages unchanged into a new output directory and publishes fixture.json only on success. Raw BGRA bytes ensure the phone sees the same input pixels, isolating Swift state/device execution from video decoding and color-management differences. The Xcode resource folder `DeviceValidationData` is tracked only as an empty placeholder; generated `baseline` contents are ignored by git. Preparation output is `DeviceValidationData/baseline`. Build/run the new “Test temporal tracking” screen on a physical phone, starting with CPU only, then accelerated modes after that passes. The assistant has not run preparation, tests, builds, inference or conversion; only static review is performed. The next user response may be a preparation traceback, Swift compiler error, or device JSON.

The user built and ran the new Swift CPU-only temporal probe successfully (`Audit/owned-temporal-cpu-report.json`). All 20 frames passed; every compared tensor had maximum absolute error 0 against the Mac replay, all low-mask IoUs were 1, and final bank counts were 7 spatial/16 pointer entries. Source fixture SHA256 is `27bbdc854536a5d33153250bc4bac44940692c0bea76fd8e6e6b9c91fb1792e9`; source model manifest SHA256 is `b8d7f80fd31cc9ae75128379e18a228753254e0a9431d04d3e4fd86914d3b754`. Thermal state stayed nominal. The report identifies hardware as `iPad8,6`, OS Version 26.5.2 (Build 25F84), so the actual test host needs confirmation; do not label this physical-iPhone validation yet. Ask whether this was run on Mac or a physical device. If Mac, obtain physical-iPhone CPU-only first; otherwise confirm the device and proceed to CPU + Neural Engine and CPU + GPU comparisons using the same fixture. No source or threshold change is needed based on this passing result.

The user confirmed that the Swift CPU-only pass was on Mac running the iPad build. The subsequent Mac CPU + Neural Engine run failed during the first image encoder: non-finite raw_vision_features, zero accepted frames. Mac CPU + GPU and Automatic both terminated with a Metal debug assertion that dispatch threadgroup dimensions included zero; the active component is unknown. These results are archived in `Audit/owned-temporal-mac-acceleration-failures.json`; they do not establish a specific defective operation or physical-iPhone behavior. No graph or numerical gate changes were made in response.

The Swift probe now records ProcessInfo.isiOSAppOnMac explicitly and atomically saves `temporal-last-run.json` in app Documents before compile/load and every model prediction, with current frame/component, progress and all completed comparison rows. It also prints/flushed a concise pre-prediction console line. Native assertions cannot be caught as Swift errors, so reopening the temporal screen offers the saved report even when the last run aborted. `completed=false` identifies an interrupted run; `passed` stays false until the full comparison succeeds. Model-call timing excludes checkpoint writes; the broader diagnostic duration now explicitly includes them. The new instrumentation is statically reviewed only and needs the user's rebuild; models and prepared fixture do not need regeneration. Next physical iPhone CPU-only first. The same checkpoint instrumentation can localize any later accelerated-mode abort.

The physical iPhone CPU-only run completed and passed all 20 frames (`Audit/owned-temporal-iphone-cpu-report.json`): hardware iPhone15,2, isIOSAppOnMac=false, OS Version 26.5.2 (Build 23F84). Minimum low-mask IoU against the Mac replay was 0.997465, pointer cosine 0.999296, and memory cosine 0.996183. Final state retained 7 spatial and 16 pointer entries; thermal state remained nominal. Fixture and model-manifest hashes match the previous Mac runs. This validates the fixed-fixture Swift CPU path on a physical phone, not sustained playback or general visual quality. Next assess CPU + Neural Engine and CPU + GPU on the same phone, sharing each report. Existing checkpoint instrumentation is present; no pull or model/fixture regeneration is needed for those runs. If a native abort occurs, relaunch and share the saved temporal report before starting another run.

Physical-iPhone accelerated results are now in `Audit/owned-temporal-iphone-acceleration-failures.json`. CPU + Neural Engine returned non-finite raw_vision_features on frame 1 with ANE compiler errors. CPU + GPU natively aborted at frame 1 Initializer with a zero-threadgroup Metal assertion; the preceding encoder returned finite tensors, without a separate numerical encoder comparison. Both have zero completed comparison frames. The upstream prompt encoder performs boolean-index updates for absent labels, a candidate empty-selection path; the actual failing operation is not proven. `Temporal/inspect_export_graph.py` reads the existing serialized packages and reports zero/unknown output shapes and indexed operations with immediate context. It uses no MLModel, prediction, tracing, or conversion. Only AST/source review has been performed by the assistant. Next obtain this static report from dog-07 before changing graphs; no further device-mode reruns needed yet.

The user completed the static graph inspection. The original initializer contains 10 non_zero operations, 6 scatter_nd and 8 gather_nd, with data-dependent point-selection dimensions; the other components have no non_zero operations. Evidence is summarized in `Audit/owned-temporal-point-indexing-summary.json`. `Temporal/diagnose_point_indexing.py` now prepares a one-frame control: identical CPU encoder inputs, original versus dense-select point encoding with unchanged weights/precision, original-source prompt checks for labels -1/0/1/2/3 and boundary/interior coordinates, full eager/traced initializer agreement, then original/candidate CPU and GPU predictions in isolated subprocesses. It exports only the initializer under a diagnostic contract that the regular device runner rejects; current validated packages/fixture stay unchanged. It requires zero non_zero operations in the candidate. The initializer comparison uses existing mask/pointer cosine and mask-IoU gates; this does not validate temporal memory, NE or physical-device acceleration. Workers preserve logs/return codes on native failures. No tests, conversion or inference were run by the assistant; AST and static review only. Next have the user run this diagnostic on Mac using dog-07 and share report.json before any model promotion.

The user completed both point-indexing controls (`Audit/mac-point-indexing-control-summary.json`). Without explicit Metal validation, original and candidate CPU/GPU predictions passed. With MTL_DEBUG_LAYER=1, original GPU aborted (-6) with the same zero-threadgroup assertion; candidate GPU passed with low-mask IoU 1.0 and pointer cosine 0.999992846. Both GPU logs confirm validation enabled. All 15 prompt cases passed; eager PyTorch candidate exactly matched original on the first frame. This establishes a successful remedy for the reproduced initializer failure on Mac, not full GPU temporal/iPhone validation or the separate NE encoder issue.

The normal Initializer now uses the identical dense-select point implementation. It copies the top-level module registry before replacing the prompt child, sharing learned weights read-only without replacing the independent reference's prompt encoder. Propagator and other component arithmetic are unchanged. Export rejects remaining initializer non_zero operations and records graphRevision=dense-initializer-points.v1 in status, comparison reports, model manifest/metadata, device fixture and device report. Normal validation/preparation and Swift device loading require that revision, preventing stale dog-07 packages being mistaken for the fix. Old precision and point-control diagnostics explicitly retain original prompt handling; diagnostic contracts remain rejected by the device runner. No assistant inference/conversion/build/tests occurred; AST/diff/static review only. Next user runs full normal validator into fresh dog-08, then provides status and Core ML report before fixture preparation/phone rebuild. The current bundled baseline is still dog-07 and must be regenerated after dog-08 passes.

Dog-08 now passed the normal complete validation, with referencePassed/coremlPassed/readyForDeviceValidation=true and graphRevision=dense-initializer-points.v1. All 20 Core ML CPU frames passed: minimum mask IoU 0.997046, pointer cosine 0.998431, memory cosine 0.994446. Evidence is in `Audit/mac-dense-point-temporal-validation-summary.json`. Next prepare a fresh fixture from dog-08, preserving/moving the old bundled dog-07 baseline outside DeviceValidationData, and rebuild on physical iPhone. Run CPU-only then CPU + GPU with Metal API validation still enabled; share temporal reports or saved crash checkpoints. No new export is needed. NE encoder failure is unresolved and is not expected to be fixed by this initializer change.

The revised dog-08 fixture has now passed all 20 frames on physical iPhone15,2 / OS 26.5.2 in both CPU-only and CPU + GPU modes. Both reports identify isIOSAppOnMac=false and graphRevision=dense-initializer-points.v1. CPU minimum mask IoU/pointer cosine/memory cosine: 0.997465/0.999296/0.996183; GPU: 0.997466/0.998979/0.994486. Final banks are 7 spatial/16 pointers, with nominal thermal state throughout. Reports are in `Audit/owned-temporal-dense-iphone-cpu-report.json` and `Audit/owned-temporal-dense-iphone-gpu-report.json`. Fixture SHA256 e0060f7bc8becb84b4ddac5ac425f119c9a4df62228e3afc1b34cbd78a5e6ccc; model manifest SHA256 8e3855c53deda132d4c378d3a93343ac3bc6a678b1a95a8cf5bb7c6bd99e61c1. The user reported both runs successful after being asked to keep Metal API Validation enabled; reports themselves do not record that setting. This confirms the full fixed-fixture GPU path on a phone, not sustained FPS or general tracking quality.

Next narrow the still-unresolved Neural Engine encoder failure. `Temporal/diagnose_encoder_acceleration.py` verifies dog-08 encoder package/frame identity, uses pinned original PyTorch encoder outputs, and compares existing versus diagnostic FP32 encoder under CPU_ONLY and CPU_AND_NE in isolated prediction processes. It reuses the established FP32 encoder export helper and saves each worker log, output metrics and errors, plus incremental report.json. CPU_AND_NE permits CPU fallback; a pass does not prove NE execution, and FP32 changes both numerical precision and eligible placement. No tracker models or phone fixture are replaced. This diagnostic has been AST/static reviewed only; the assistant ran no inference/conversion/tests/builds. User command should use --upstream .upstream-edgetam --run .temporal-runs/dog-08 --output .temporal-runs/encoder-ane-01; include cat for report.json and existing-CPU_AND_NE.log/fp32-CPU_AND_NE.log in chat.

The encoder-ane-01 control is complete (`Audit/mac-encoder-ane-control-summary.json`). Existing mixed encoder passes CPU_ONLY but fails CPU_AND_NE: raw/initial outputs each have 1,048,576 non-finite elements, with high-resolution features finite but approximately 1e37 magnitude. FP32 encoder passes both modes; reported output metrics are identical between CPU_ONLY and CPU_AND_NE, consistent with possible CPU fallback, not proof of ANE correctness. Logs show prediction returned; this isolated run does not reproduce the earlier ANE compiler error. No model promotion or precision change follows from this result.

Next `Temporal/inspect_encoder_placement.py` loads/compiles the verified existing dog-08 and encoder-ane-01 FP32 packages, then inspects MLComputePlan under CPU_AND_NE. It records per-operation preferred/supported devices, nonconstant totals and per-type counts, plus summary.json. These are planned assignments, not actual runtime utilization or timing percentages. No inference/tracing/conversion occurs. It preserves completed/partial reports on ordinary errors. Python API usage was checked against Apple documentation and AST/static reviewed; no plans/models/tests were run by the assistant. User should pass --run .temporal-runs/dog-08 --diagnostic .temporal-runs/encoder-ane-01 --output .temporal-runs/encoder-plan-01 and share summary.json; full operation report is report.json if needed. Passing phone CPU/GPU models stay unchanged.

Encoder-plan-01 confirms the FP32 control has a CPU-only plan: all 277 nonconstant operations support/prefer CPU and none support NE. The existing mixed encoder has 450 nonconstant operations, with 367 preferring CPU and 83 preferring NE; all 83 NE-preferred operations are convolutions (99 convolutions support NE). Evidence summary: `Audit/mac-encoder-placement-summary.json`. This rules out presenting the FP32 pass as an NE fix; the exact corruption location is not established.

`Temporal/diagnose_encoder_taps.py` now prepares a diagnostic copy of the existing serialized encoder using Apple extract_submodel, retaining all four final outputs and exposing input/output tensors at five evenly spaced NE-preferred convolutions from encoder-plan-01 (optional --ne-indices for later targeted samples). It verifies package/frame/plan identity, validates probe CPU final outputs against the untouched original, records the probe compute plan and sampled device preferences, then compares CPU vs CPU_AND_NE tensors in isolated prediction workers. It explicitly reports whether the original non-finite final-output failure is reproduced and whether sampled convolution preferences remain NE; exposing outputs can change partitioning, so passing taps alone cannot exonerate the original graph. The first failing sampled checkpoint is not necessarily the first faulty operation. Existing models and phone fixture remain unchanged. Assistant ran only AST/static/diff checks; user must run --run .temporal-runs/dog-08 --plan .temporal-runs/encoder-plan-01 --output .temporal-runs/encoder-taps-01 and share report.json (plus probe-ne.log on failure).


Encoder-taps-01 failed before any prediction: coremltools 9 extract_submodel exceeded Python's recursion limit inside copy.deepcopy while copying the MIL graph. Evidence: `Audit/mac-encoder-taps-preparation-failure.json`; no new NE result was obtained. The tap diagnostic now temporarily raises the recursion limit to at least 20,000 for extraction only, records both limits before extraction, and restores the original limit in finally. Existing output/placement checks and model bytes are unchanged. Static AST/diff checks only; no assistant model execution. Next user reruns into fresh encoder-taps-02 and shares report.json; include cat directly in chat.


Encoder-taps-02/03/04 now complete, reproduce the NE final failure, preserve all sampled NE preferences and retain exact CPU final agreement. Taps-04 shows healthy shared input to convolutions 6/7 and closely agreeing outputs (cosine 0.999987/0.999974), but convolution 8 input input_39_to_fp16 already has 32,768 non-finite values. Selected supplied facts: `Audit/mac-encoder-tap-localization-summary.json`. No faulty operation is established. The diagnostic now accepts --trace-input for a sampled NE index, follows actual serialized dependencies back to every nearest convolution output, and adds taps for intervening operations. It records input connectivity, original/probe preferred devices and CPU baseline ranges. This includes parallel paths and avoids assuming adjacent convolution indices form a chain. Next user runs indices 6 7 8 with trace-input 8 into fresh encoder-taps-05 and shares report.json. Only AST/data/diff checks by assistant; no model execution or production precision change.


Encoder-taps-05 completed (`Audit/mac-encoder-residual-taps-report.json`). CPU finals remain exact; NE full-graph failure persists. Exposed FP32 residual sums input_33 and input_39 are all zeros although operands agree closely with CPU. input_33_to_fp16 remains healthy, inconsistent with its exposed all-zero FP32 source; input_39_to_fp16 is now finite but wrong (cosine 0.64657). Five exposed FP16-to-FP32 casts move planned preference from CPU to NE (totals 362 CPU/88 NE), despite all sampled convolution preferences remaining NE. Do not identify a faulty add kernel from these inconsistent readouts. New diagnose_encoder_prefix.py extracts two separate prefixes directly from the verified original encoder, ending at input_33 and input_39 with each FP32 output and FP16 cast. It uses the saved taps-05 CPU baseline (checked against reported ranges and cast consistency), requires prefix CPU parity before NE, and records prefix plans, finite/cosine comparisons and within-run cast consistency. Shortening the graph changes downstream consumers and may change placement/lifetimes, so a passing prefix is not a full NE fix. Source packages and phone fixture are unchanged. AST/diff/data checks only by assistant. Next user runs --run dog-08 --taps encoder-taps-05 into fresh encoder-prefix-01 (full paths and cat report.json in chat).


Encoder-prefix-01 completed (`Audit/mac-encoder-prefix-summary.json`): both CPU prefixes exactly match saved full-graph CPU. Under CPU_AND_NE, input_33 prefix passes (cosine 0.999988), but extending through the residual block to input_39 fails (cosine 0.547246), with finite outputs and exact within-run FP32/FP16 cast consistency for both. Terminal sums/casts prefer CPU; prefix plans have 6 and 8 NE-preferred operations respectively. No kernel/lifetime cause is proven. New diagnose_residual_inputs.py uses Apple extract_submodel input cuts on the verified input_39 prefix: shared input_33 versus separate input_33/input_33_to_fp16 boundaries. Both receive identical saved CPU values; separate inputs are independent arrays. It retains the two convolutions and original sums/precision, requires CPU parity against the prefix, and records full tiny-block compute plans and same-input NE metrics in isolated workers. A pass after cutting cannot establish a complete encoder fix or actual hardware usage. Checked Apple extractor input-cut source, Python AST and git diff only; no assistant inference/conversion/tests. User runs --run dog-08 --prefix encoder-prefix-01 into fresh residual-inputs-01 and shares report.json (full paths and cat in chat).


Residual-inputs-01 completed (`Audit/mac-residual-input-controls-report.json`): shared and separate external-input variants both match CPU closely under CPU_AND_NE (all cosine values >0.9999999), with both convolutions still preferring NE. CPU outputs are exact. Sharing an external input alone is not sufficient to reproduce failure; the larger prefix remains the failing context. Next intervention uses diagnose_encoder_prefix.py --round-residual on the original failing input_39 prefix, retaining the upstream encoder. It replaces only the input_39 skip operand input_33 with float32(input_33_to_fp16), leaving convolution inputs/weights and FP32 sums unchanged. This deliberately rounds the skip value and changes possible lifetimes/placement; no production precision change is made. It verifies the exact add/cast topology, records the edit and plan, and requires CPU parity against an explicit saved-branch-sum plus rounded-skip reference before NE. Original CPU metrics also show the rounding impact. Coremltools 9 operation.set_inputs and Builder insertion API were checked against Apple source; only AST/data/diff checks by assistant. User runs --run dog-08 --taps encoder-taps-05 --round-residual into fresh encoder-residual-roundtrip-01, with cat report.json directly in chat.


Encoder-residual-roundtrip-01 passes (`Audit/mac-rounded-residual-prefix-summary.json`): CPU equals the explicit rounded arithmetic exactly; original-prefix CPU max error is 0.001950 for FP32 output. NE cosine improves from 0.547246 to 0.999990 with all outputs finite and consistent casts; plan 35 CPU/8 NE. This is one prefix, not a full encoder fix. Next diagnose_encoder_residuals.py prepares three complete encoders from unchanged serialized source: unchanged (same second conversion), single (tested input_39 skip edit), and matching (same data-dependency pattern across the graph). A match requires an FP32 add operand with an existing FP16 cast and another add operand depending on that cast through convolution; ambiguous matches fail. Edits round only skip operands, retain weights/convolution precision and do not expose intermediate outputs. It requires the known successful edge, records every match/edit and full compute plans, and compares CPU/NE outputs on the saved first frame against untouched CPU. Unchanged CPU must be allclose; altered encoders must meet cosine 0.99 on CPU before NE and NE must agree with original and variant CPU. All stay diagnostic and outside phone fixture. No model execution by assistant, AST/diff checks only. User runs --run dog-08 --prefix-control encoder-residual-roundtrip-01 into fresh encoder-residuals-01; include cat report.json in chat.


Encoder-residuals-01 completed (`Audit/mac-full-encoder-residual-controls-report.json`): unchanged/single/matching CPU comparisons pass. Unchanged second conversion still fails NE, and the single edit is insufficient. Matching all 36 residual edges makes all four outputs finite but badly wrong (original-CPU cosine high0 0.001781, high1 0.547801, raw 0.719460, initial 0.712026). All plans retain 83 NE-preferred convolutions; no promotion. The source FPN retains trunk features for later neck convolutions, a class of consumer outside residual additions. New optional --variants fanout in diagnose_encoder_residuals.py rounds later uses of each FP32 activation that already has an FP16 cast feeding a convolution, while retaining original convolution casts and consumers earlier than the cast. It includes non-add consumers, records exact source/cast/consumer edges, rejects model-output sources, and requires the known residual plus additional consumers. This remains a hypothesis about broader shared-value usage, not a proven buffer bug. Same CPU/NE gates apply; user runs only fanout into fresh encoder-fanout-01 with existing --run dog-08 and --prefix-control encoder-residual-roundtrip-01. Other modes remain reproducible; validated phone models unchanged. Assistant performed source review, AST/data/diff checks only. Include cat report.json in chat.


Encoder-fanout-01 passes the full one-frame CPU and CPU_AND_NE comparison (`Audit/mac-encoder-fanout-report.json`). All outputs finite; minimum NE cosine vs original CPU 0.999308546 and vs candidate CPU 0.999305822. There are 39 edits: 36 residuals plus later consumers of input_49, input_121, input_383 (outputs x, x_97, var_1561). Plan: 406 CPU/83 NE-preferred operations, with all 83 NE preferences still convolutions. No physical-device NE or temporal validation yet. Next validate_encoder_candidate.py reuses the verified fanout package and all 20 saved dog-08 frame PNGs/PTS/prompt, running candidate encoder CPU_ONLY and CPU_AND_NE in separate workers with original tracker packages CPU_ONLY. It uses canonical v.reference_step/v.compare, independent pinned PyTorch history and bounded candidate state, strict frame hashes and original mask/pointer/memory/presence gates. Checks full 7/16 banks, saves masks/per-mode reports/logs and a compact root summary; checkpoints before model calls and uses Torch threads=1. No re-export or readiness promotion. Assistant ran AST/data/diff checks only. User supplies --upstream .upstream-edgetam --run .temporal-runs/dog-08 --candidate .temporal-runs/encoder-fanout-01 --output .temporal-runs/encoder-temporal-01, then cat root report.json. Candidate encoder image interface may be raw NCHW pixels; helper preserves existing 1/255 preprocessing. Current bundled phone models remain dog-08.

The current fixture is `assets/IMG_2771.mov`, point 0.4912/0.4706; current validated Mac models are under `.temporal-runs/dog-09/models`; the prior physical-phone CPU/GPU fixture uses dog-08; dog-07 preserves the earlier initializer; dog-06 and dog-05 retain earlier diagnostic policies. Include `cat` or `open` commands whenever asking for reports (new explicit user preference). Give all setup/rerun commands directly in chat; the user explicitly rejected being sent to a branch README for commands. Keep `diagnose_initializer.py` targeted at the original v1 packages for reproducible diagnostics. Do not integrate the owned models into iPhone until the complete comparison passes.

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

1. The 39-edge encoder candidate passed 20 Mac frames with CPU and CPU_AND_NE encoder, tracker CPU. The normal dog-09 export and physical-phone CPU/Encoder NE + CPU tracker modes now passed. All five physical-phone modes and the Release GPU/NE runs now passed on iPhone 14 Pro. Both 17 Pro Release modes also passed. GPU session is 100 ms, NE 439 ms; next inspect owned encoder/propagator GPU/NE plans to target graph optimization. Preserve old models and parity gates. Include cat/open with report requests. Do not run tests/inference/builds on the user's behalf.
2. Resolve concrete parity/export failures without relaxing thresholds to hide a defect. Current gates are explicit engineering policies, not previously measured results. Obtain representative visual review and retain the fixture results. The community contract remains unknown; the owned implementation does not claim to reconstruct it.
3. The Swift fixed-fixture temporal runner has built and passed CPU-only on both Mac and physical iPhone; accelerated-mode validation remains. Resolve concrete compiler/device failures, then extend to requested-video-frame operation and profile on iPhone. Keep the community packages and existing diagnostic usable until a validated replacement exists. The new export has different input names and no application-supplied rotary tensor; it is not a drop-in replacement.
4. Evaluate mask edges, hair, occlusion/reappearance, fast motion, tracking drift, latency, and sustained performance before integrating woven text.

The user authorized pushing this project to `crpeter/ByteWaveSegmentation`. The initial GitHub permission issue was fixed by granting repository access. No ByteWave production files have been changed. The previous downloadable ZIP failed to expand on macOS; use this repository instead.
