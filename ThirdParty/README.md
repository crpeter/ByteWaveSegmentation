---
license: apache-2.0
library_name: coreml
pipeline_tag: mask-generation
tags:
  - coreml
  - ios
  - edge-ml
  - video-segmentation
  - object-tracking
  - edgetam
  - sam2
  - ios-18
---

# EdgeTAM Core ML Video

A community Core ML video implementation of [EdgeTAM](https://github.com/facebookresearch/EdgeTAM), converted from the official Apache 2.0 checkpoint and validated against the original PyTorch video pipeline.

The repository provides four cooperating Core ML packages for prompt initialization, temporal-memory construction, and sequential video mask propagation on iOS 18 and macOS. The conversion preserves EdgeTAM's original learned weights while exposing a mobile-oriented explicit memory-bank interface.

> This community conversion is not affiliated with or endorsed by Meta.

## Demo

The same tracked person mask drives segmentation, background blur, and subject-focus effects:

<video src="assets/EdgeTAM-CoreML-Video-Demo.mp4" controls width="100%"></video>

[Download the H.264 demo](assets/EdgeTAM-CoreML-Video-Demo.mp4)

The demonstration runs the actual four-model Core ML pipeline at 10 tracked frames per second. It does not use simulated masks.

## Model packages

| Package | Responsibility |
|---|---|
| `EdgeTAMVideoImageEncoder.mlpackage` | Encodes a 1024×1024 RGB frame into raw, initial, and high-resolution vision features. |
| `EdgeTAMVideoInitializer.mlpackage` | Converts the first-frame vision features and point prompt into initial masks, object score, IoU estimate, and object pointer. |
| `EdgeTAMVideoMemoryEncoder.mlpackage` | Encodes the initial mask and raw vision features into spatial and temporal memory. |
| `EdgeTAMVideoPropagator.mlpackage` | Attends over the explicit memory bank to propagate the mask and produce the next memory entry. |

## Pipeline

1. Resize each video frame to 1024×1024 and run the image encoder.
2. On the first frame, scale the user's positive or negative point prompt into 1024-space and run the initializer.
3. Run the memory encoder using the initial high-resolution mask.
4. Maintain the conditioning memory, six recent memories, and up to sixteen object pointers in application code.
5. For subsequent frames, build the explicit memory tensors and run the propagator.
6. Threshold `low_res_mask` logits at zero, resize to the source resolution, and composite the desired visual effect.

This split keeps temporal state under the application's control and permits a fresh memory bank for every tracked object/session.

## Core interfaces

### Image encoder

- Input: `image`, RGB image, 1024×1024
- Outputs:
  - `raw_vision_features`: `1×256×64×64`
  - `initial_vision_features`: `1×256×64×64`
  - `high_res_feature_0`: `1×32×256×256`
  - `high_res_feature_1`: `1×64×128×128`

### Initializer

- Inputs: initial and high-resolution vision features, `point_coords`, and `point_labels`
- Outputs:
  - `low_res_mask`: `1×1×256×256`
  - `high_res_mask`: `1×1×1024×1024`
  - `best_iou`: `1`
  - `object_pointer`: `1×256`
  - `object_score`: `1×1`

### Memory encoder

- Inputs: `raw_vision_features`, `high_res_mask`, and `object_score`
- Outputs:
  - `memory_features`: `1×512×64`
  - `memory_positions`: `1×512×64`
  - `temporal_positions`: `7×64`

### Propagator

- Inputs: current-frame features plus `spatial_bank`, `spatial_positions`, `pointer_bank`, `attention_bias`, and `rotary_weight`
- Outputs: current masks, IoU estimate, object pointer/score, and the next memory feature/position tensors

## Requirements

- iOS 18.0+ or a compatible macOS Core ML runtime
- Core ML
- Swift or Objective-C orchestration for preprocessing and explicit memory-bank management
- `.cpuAndGPU` is recommended for the complete pipeline on current iPhones because some graph operations may not compile for the Neural Engine when `.all` is selected.

## Validation

The converted components were compared sequentially with the original EdgeTAM PyTorch video predictor. An eight-frame validation sample produced mask IoU values approximately from `0.9256` to `0.9794`, with cosine similarities approximately from `0.9959` to `0.9999`.

These figures describe one parity-validation sample, not a general benchmark. Device performance depends on hardware, compute-unit selection, thermal state, and application scheduling.

## License and attribution

The original EdgeTAM code and checkpoints are licensed under Apache License 2.0. This repository redistributes converted object-form models under the same license. See [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).

If this work is useful, please cite the original EdgeTAM project:

```bibtex
@article{zhou2025edgetam,
  title={EdgeTAM: On-Device Track Anything Model},
  author={Zhou, Chong and Zhu, Chenchen and Xiong, Yunyang and Suri, Saksham and Xiao, Fanyi and Wu, Lemeng and Krishnamoorthi, Raghuraman and Dai, Bo and Loy, Chen Change and Chandra, Vikas and Soran, Bilge},
  journal={arXiv preprint arXiv:2501.07256},
  year={2025}
}
```

## Conversion credit

Core ML video conversion, explicit-memory runtime design, iOS integration, and parity validation by **Ahmad Yar** (`AhmadYarAI`).
