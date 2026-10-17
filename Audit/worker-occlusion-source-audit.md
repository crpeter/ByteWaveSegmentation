# Worker occlusion source audit

Reviewed at ByteWave commit b3c0ed3f2e1c12d72e7fbb77e8bc24581f169a6b.
Pinned upstream: facebookresearch/EdgeTAM commit
7711e012a30a2402c4eaab637bdb00a521302c91, sam2/modeling/sam2_base.py.

## Findings

- OwnedTemporalState and Temporal/state.py keep the initial conditioning
  memory and pointer until explicit reset. The original prompt is not evicted.
- Both pack the conditioning spatial memory plus the previous six decoded
  frames, oldest to newest; pointers are condition plus fifteen recent frames,
  newest to oldest. This matches the pinned forward, stride-one source.
- Both commit every finite validated output, including predicted absence.
  Original track_step also encodes absent predictions; _encode_new_memory
  has explicit no-object handling. Suppressed masks/pointers are deliberate
  model behavior, not a reason to silently skip memory updates.
- owned.Propagator delegates mask-head and new-memory encoding to the pinned
  original implementations. Its fixed attention is separately rewritten and
  therefore remains a comparison target, not assumed correct on arbitrary clips.
- Recent spatial context spans only six frames (about0.2s at30fps), plus the
  retained initial condition. Long absence replaces recent visible entries;
  this alone is not evidence of a bug or proof of this failure's cause.
- Static review found no obvious absence-specific commit/packing discrepancy.
  Prior20-frame dog parity does not validate this longer occlusion case.

## Evidence and next diagnostic

iphone17pro-worker-occlusion-report.json contains selected fields from the
uploaded phone report and its SHA256. One unseeked selection,236 predictions,
124empty masks. All retained120rows (indices116..235;3.8683..7.8367s) are empty.
Sampled screen-recording images show selected bent-over worker, bucket overlap,
then visible shirt without highlight. No precise first-loss timestamp or
post7.84-second recovery conclusion is available from the phone report.

Temporal/diagnose_occlusion.py compares300source frames locally on Mac using:
original PyTorch track_step with its own full prefix history; owned FP32 with
its own bounded state; and exact installed memoryfp16projection package bytes
on CPU_AND_GPU with a third independent bounded state. No model weights,
selection logic or state policies are changed. Numerical mismatches remain
failures under existing gates but tracing continues to inspect later recovery.
Shape/nonfinite/native errors are not bypassed. No automatic promotion.

The original worker video is required, not the screen recording. Source size,
dimensions,duration and firstPTS guard against using another file. These checks
are not cryptographic identity with the phone: the existing phone report lacks
a source-content hash and input pixels. The Mac records source/input SHA256;
all three Mac streams receive identical inputs. PyAV/Pillow versus iPhone
Core Image and Mac versus iPhone execution remain confounders if phone-only
failure cannot be reproduced. Preview initial-selection.png confirms orientation
and point; previews/ panels every15frames and mask-empty transitions aid review.
An automatically returning mask is not automatic proof of correct identity.

Reference helpers now accept optional count/num_frames; original20-frame
callers and gates retain their defaults. Assistant performed AST/static review
and offline uploaded-report analysis only. No builds, tests or inference run.

