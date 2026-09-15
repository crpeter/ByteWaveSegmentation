# Owned temporal contract v2

Status: dog-08 passed the normal Mac comparisons and physical-iPhone CPU/GPU
20-frame fixtures. A diagnostic encoder with 39 shared-activation rewrites now
passes 20 frames on Mac with either CPU_ONLY or CPU_AND_NE for the encoder,
with the tracker on CPU_ONLY. The normal exporter now incorporates that rewrite;
its new dog-09 full-set Mac validation passed all 20 frames. Physical-iPhone CPU and encoder-NE/CPU-tracker fixtures also passed all 20
frames. Full-tracker CPU_AND_NE validation remains pending.
These results concern one clip and do not establish sustained performance.

Contract ID: `bytewave.edgetam-temporal-owned.v2`.

This is a new, source-controlled four-model export. It does not claim to recover
the community export's undocumented `attention_bias` / `rotary_weight` contract.
Its model names and inputs are intentionally different. All learned weights come
from the original checkpoint; no training or model-weight edits are involved.

## Pinned reference

- [Meta EdgeTAM source](https://github.com/facebookresearch/EdgeTAM/tree/7711e012a30a2402c4eaab637bdb00a521302c91)
- Checkpoint: `checkpoints/edgetam.pt`, SHA256
  `ed2d4850b8792c239689b043c47046ec239b6e808a3d9b6ae676c803fd8780df`.
  This matches the [official checkpoint revision](https://huggingface.co/facebook/EdgeTAM/blob/f3a09791b2343c0733d456d08d73771d9363b69a/edgetam.pt).
- Original reference functions: `SAM2Base.track_step`,
  `_prepare_memory_conditioned_features`, `_encode_new_memory`, and
  `apply_rotary_enc_v2` in `sam2/modeling/position_encoding.py`.

The loader requires the exact source revision, unmodified tracked files and the
checkpoint hash. It builds the original model with postprocessing disabled and
`binarize_mask_from_pts_for_mem_enc=true`. The only construction substitution is
an equivalent TIMM backbone with `pretrained=False`: strict full-checkpoint loading
supplies every learned weight, avoiding an unrelated ImageNet download.

The contract supports one object, one positive initial point, forward sequential
frames, a memory stride of one and a reset on new selection/video/seek. Corrections,
multiple conditioning frames, reverse tracking and multi-object batching are not
implemented. Original object-presence and pointer behavior is retained, including
memory updates when the subject is predicted absent. No confidence-based eviction
or invented reacquisition rule is added.

## Models

| Package | Inputs | Outputs |
|---|---|---|
| `BWTemporalImageEncoder` | RGB 1024 × 1024 image | Raw and initial 1 × 256 × 64 × 64 features; high-resolution 1 × 32 × 256 × 256 and 1 × 64 × 128 × 128 features |
| `BWTemporalInitializer` | Initial features, both high-resolution features, float32 point 1 × 1 × 2, int32 label 1 × 1 | Low/high mask logits, best IoU estimate, pointer and object score |
| `BWTemporalInitialMemoryEncoder` | Raw features, initial high-resolution logits and object score | Memory features and positions, each 1 × 512 × 64 |
| `BWTemporalPropagator` | Raw/high-resolution features plus the banks below | Mask/score/pointer outputs plus the next memory features and positions |

Image input uses RGB byte values through Core ML's 1/255 image scale; the graph
applies mean `[0.485, 0.456, 0.406]` and std `[0.229, 0.224, 0.225]` exactly once.
Point x/y uses the display-oriented frame's normalized top-left coordinates times
1024, clamped to 1023 at the final edge. Initialization adds `no_mem_embed`; later
frames use memory-conditioned features. Both paths select among three masks using
the original IoU head. Initial memory binarizes the prompt mask; propagated memory
uses sigmoid probabilities. The original 20× scale and −10 bias remain in the graph.

Floating tensor inputs and outputs use float32 in Core ML; labels remain int32.
This interface change distinguishes v2 from the initial, unvalidated v1 export.
Policy `mixed-encoder-late-conv33-fanout-and-fp32-attention-iou.v1` uses component-specific
precision. In the image encoder, 99 convolutions use FP16; 33 named convolution
module paths and all non-convolution operations retain FP32. The protected paths
are in `encoder_precision.py`, taken from the passing [99,132) diagnostic control.
Export checks the expected 132 convolutions, all protected paths, and exactly 33
FP32 matches. The manifest records per-convolution precision and excluded ops.

The encoder then reconstructs exactly 39 later FP32 consumer operands from their
existing FP16 convolution-input activations: 36 residual operands and three saved
feature-map consumers. Arithmetic remains FP32 after this activation rounding.
`encoder_fanout.py` checks all 39 edge identities and preserves convolution
precision. It reloads the serialized graph as in the passing diagnostic and
restores the original RGB interface without adding a second image scale. The
manifest records every replacement. Learned weights remain unchanged.

The other three components use FP16 except `matmul`, `softmax`,
`scaled_dot_product_attention`, the IoU prediction MLP, and its floating score
path through mask selection. Export fails if the initializer/propagator IoU MLP
or argmax path cannot be identified. This retains the earlier tracker policy.

The fanout encoder control passed all 20 frames: CPU minimum mask IoU 0.995740,
pointer cosine 0.999027, memory cosine 0.996570; CPU_AND_NE encoder with CPU tracker
gave 0.998153, 0.999490 and 0.998537 respectively. These diagnostic results do not
replace the normal full-set gate, visual review, or device profiling. No checkpoint
weights or comparison thresholds are changed.
Exact names/shapes are in
`validate_export.py:SHAPES`; no output masks are used to guess memory layouts.

## Application-owned banks

| Input | Shape | Meaning |
|---|---|---|
| `spatial_bank` | 1 × 7 × 512 × 64 | Conditioning memory, then memories at lags 6, 5, 4, 3, 2, 1 |
| `spatial_positions` | 1 × 7 × 512 × 64 | Matching raw position tensors, before learned temporal offsets |
| `pointer_bank` | 1 × 16 × 256 | Conditioning pointer, then nonconditioning pointers at lags 1…15 |
| `valid_slots` | 1 × 23 | Seven spatial flags followed by sixteen pointer flags; exactly 0 or 1 |

Unfilled slots are zeroed and invalid. The conditioning frame is never duplicated
into the recent-memory/pointer slots. A pointer splits into four consecutive
64-channel tokens without reordering. At frame 1 only spatial slot 0 and pointer
slot 0 are valid; at frame 7 all spatial slots are full; at frame 16 all pointer
slots are full and old entries begin to be evicted.

Learned temporal-position rows are added inside the graph in slot order
`[6, 5, 4, 3, 2, 1, 0]`. Pointer positional encodings are zero, as configured by
the pinned model. The application does not compute rotary values or attention
biases. The graph expands validity to 3584 spatial tokens plus 64 pointer tokens,
applying zero attention bias to valid tokens and −10000 to invalid tokens. This
finite mask relies on numerical softmax suppression and must pass parity checks;
it is not asserted to be an exact algebraic identity for arbitrary input values.

`state.py` retains at most seven spatial entries and sixteen pointers, independent
of clip length. Retained bank tensors occupy 1,851,392 bytes at float32, excluding
Python object overhead and temporary packed inputs. All predictions commit only
after validation. Session generation plus next-frame index rejects stale results
after reset; non-increasing presentation timestamps require reset/reselection.
Frame ordering uses accepted decoded-frame indices; timestamps remain exact
rationals rather than frame-index/nominal-FPS estimates.

## Rotary encoding

This follows the original `RoPEAttentionv2` source's token split:

- Queries: 64 × 64 axial positional encoding, theta 10000.
- For each 512-token spatial memory, the first 256 learned tokens bypass rotation;
  the next 256 use 16 × 16 axial positions.
- All pointer tokens bypass rotation.

`owned.py` implements complex multiplication as paired real arithmetic. Constants
are computed from the original formula during model construction and become graph
buffers. They are not application inputs. Self-attention uses the corresponding
64 × 64 real rotary operation. The original model remains untouched, including
its complex rotary implementation, for independent numerical comparison.

## Validation gate

`validate_export.py` reads 20 consecutive frames, enough to exercise startup,
full banks and eviction. The reference uses original `track_step` and a separate
unpruned 20-entry history; the candidate uses the bounded state and owned graphs.

1. Float32 PyTorch: every reported tensor must satisfy atol/rtol 0.001, mask IoU
   at least 0.999 and matching object-presence classification.
2. Export: four iOS 18 Core ML packages, explicit float32 tensor interfaces, new contract
   metadata, pinned checkpoint identity and per-file output hashes.
3. Core ML CPU comparison: repeat the same decoded frames and point, requiring
   mask IoU at least 0.95, cosine similarity at least 0.99 for mask/memory/pointer
   tensors, matching presence, and finite outputs of the expected shapes.

These are explicit engineering gates, not observed results or general model
quality claims. A failed gate records an error and leaves
`readyForDeviceValidation=false`. An empty initial reference mask fails rather
than yielding a meaningless all-background pass. Both absent masks on later
frames may legitimately agree. Candidate and reference masks are saved for visual
inspection; their agreement does not prove correct real-world segmentation.

The fixture reader applies PyAV's quarter-turn display rotation and PIL bilinear
square resizing to both paths. It rejects non-square pixels and mirrored display
matrices. This is a shared diagnostic preprocessing path, not a pixel-parity claim
with the iOS Core Image preprocessing. HDR/color-management parity, longer occlusion
sequences, actual Neural Engine execution, sustained latency and thermal effects
remain separate device-validation work.

## Fixed-fixture device comparison

The Swift probe uses the same bounded banks and exact rational CMTime timestamps.
The Mac preparation step requires the passed normal run and unchanged package
hashes, then replays its 20 saved images with Core ML CPU to save reference tensors.
Raw BGRA inputs bypass device video decoding/resizing. The phone verifies resource
hashes and model metadata/schema before inference, checks all output tensors for
finite values, and compares low masks, pointers, memories and presence against
those Mac Core ML references. High masks are shape/finiteness checked only. The
same cosine 0.99 and low-mask IoU 0.95 gates apply relative to this stated reference.
The report separates compilation/loading, model calls and state/copy overhead.
It does not establish sustained frame rate or runtime Neural Engine utilization.
The "Encoder NE + CPU tracker" mode requests CPU_AND_NE only for ImageEncoder
and CPU_ONLY for the other three components. Reports record requested units per
component; these settings do not measure actual hardware execution.
Only one current input/result and bounded memory are retained; the 20 reference
frames on disk are a correctness fixture, not a production mask cache.

## Graph revision

`dense-points-encoder-fanout.v1` retains the prior dense-point initializer and
adds the encoder rewrite above. Dense point embedding replaces boolean-index
updates with fixed-size broadcast selects, preserving point offsets, padding and
learned embeddings. The independent upstream reference and propagator prompt path
remain unchanged. Export rejects initializer `non_zero` operations. Normal
validation, device preparation and Swift loading require matching graph and
precision metadata; dog-08 fixtures cannot validate the new revision.
