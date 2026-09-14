# Temporal contract review — 2026-09-14

## Decision

Keep the bundled candidate: all four packages loaded and produced compute plans
in the user's CPU + Neural Engine, CPU + GPU, and Automatic runs. See
`device-placement-summary.json`. These observations establish compatibility on
that device/OS, not prediction correctness, Neural Engine execution, or speed.

Implement the documented first-frame path. Do not fabricate the propagator's
application-side tensors. No complete temporal runtime/export source or parity
fixture was located for this community conversion during this review.

## Sources inspected

- [Pinned community export](https://huggingface.co/AhmadYarAI/EdgeTAM-CoreML-Video/tree/6bfdd4765e42508c7707566fff52e65add8b8e3a),
  including its recursive file inventory and bundled `ThirdParty/README.md`.
  The inventory contains four model packages, a demo video, licenses, and the
  model card; it contains no Python/Swift runtime or exporter.
- `Audit/model-inspection.json`: exact input/output names, dimensions and dtypes
  parsed from the bundled model interfaces. Shapes establish representation,
  not the meaning of application-supplied temporal inputs.
- [Original EdgeTAM memory assembly](https://github.com/facebookresearch/EdgeTAM/blob/7711e012a30a2402c4eaab637bdb00a521302c91/sam2/modeling/sam2_base.py),
  `_prepare_memory_conditioned_features`.
- [Original Core ML export](https://github.com/facebookresearch/EdgeTAM/tree/7711e012a30a2402c4eaab637bdb00a521302c91/coreml):
  image encoder / prompt encoder / mask decoder, not this four-package runtime.

The original runtime places conditioning memories before recent nonconditioning
memories, adds temporal positional encodings, and constructs object-pointer
tokens. It concatenates the available entries dynamically. That is useful
reference behavior but does not specify the community conversion's padded,
fixed-size attention and rotary interface.

## Supported first-frame wiring

| Stage | Input | Action |
|---|---|---|
| Image encoder | Display-oriented RGB frame | Resize to 1024 × 1024; the model owns normalization |
| Initializer | Initial vision features, both high-resolution features, positive point | Map normalized display x/y directly to 1024-space, top-left origin; one float16 point and int32 label 1 |
| Initial memory encoder | Raw vision features, initial high-resolution mask and object score | Forward the model outputs unchanged |
| Display | Low-resolution mask logits | Threshold at zero and stretch the overlay back to the displayed frame's aspect ratio |

The Swift probe uses a BGRA pixel buffer as Core ML's RGB image input. The model
schema is checked for all forwarded tensor shapes/types. Finite-value checks
cover the final mask, pointer, score, IoU estimate and initial memory outputs.
The mask's high-resolution logits are never replaced with the display mask.

This is a diagnostic preprocessing choice: extract a display-oriented frame with
long edge at most 1024, then stretch it to the model's square input. It is not a
pixel-parity claim against the exporter's PyTorch preprocessing. Portrait,
landscape and off-center point/overlay alignment still require device checks.

## Propagation blockers

| Input/behavior | Missing authoritative detail |
|---|---|
| `spatial_bank`, `spatial_positions` | Exact slot packing during startup and after eviction; where learned temporal offsets are added |
| `pointer_bank` | Exact pointer ordering, validity, retention and any temporal-position handling in this export |
| `attention_bias` | Token packing and values used for invalid slots, including partially populated banks |
| `rotary_weight` | Construction formula, indexing and dependence on populated memories; cannot be inferred from its 1 × 1792 shape |
| State update | Empty-mask/absent-object policy, initialization versus propagated masks, reset and ordering semantics |

To unblock this export, obtain its bank-building runtime/export code and a small
consecutive-frame parity fixture covering startup and a full memory bank. If
these are unavailable, create an owned export/runtime from the original EdgeTAM
implementation with documented inputs and reference-output parity. Do not
convert undocumented guesses into production state rules.

## Implemented diagnostic boundary

`FirstFrameProbe.swift` performs one untimed warm-up and three timed repetitions
of image encoder → initializer → initial memory encoder on the same frame/point.
No propagator call exists. Compile/load and preprocessing times are separate;
output validation and mask rendering are outside prediction timing intervals.
The report is not a sustained benchmark or a playback-FPS measurement.

Only one preview frame is retained. Model intermediates are released between
repetitions; models and temporary compiled packages are released after the run.
The picker import is removed after extracting the frame. There is no frame-count
pass, whole-video decode loop, or mask cache. Reports overwrite one file per mode.

No Xcode builds, iOS tests, or device runs were performed by the assistant.
Device validation of this new code remains pending. The earlier placement probe
was successfully built and run by the user.
