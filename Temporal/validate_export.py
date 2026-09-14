#!/usr/bin/env python3
"""Run locally on Mac: reference comparison, export, then Core ML comparison.

No package is declared validated unless both comparisons finish. No iOS build,
training, network access, or production-model replacement occurs in this script.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import time

import av
import numpy as np
from PIL import Image
import torch

import owned
from state import TemporalState

COUNT = 20  # Includes startup, seven spatial slots, sixteen pointers, and eviction.
NAMES = {"ImageEncoder": owned.IMAGE_OUTPUTS, "Initializer": owned.MASK_OUTPUTS,
         "InitialMemoryEncoder": owned.MEMORY_OUTPUTS,
         "Propagator": owned.MASK_OUTPUTS + owned.MEMORY_OUTPUTS}
INPUTS = {
    "ImageEncoder": ("image",),
    "Initializer": ("initial_vision_features", "high_res_feature_0", "high_res_feature_1", "point_coords", "point_labels"),
    "InitialMemoryEncoder": ("raw_vision_features", "high_res_mask", "object_score"),
    "Propagator": ("raw_vision_features", "high_res_feature_0", "high_res_feature_1") + owned.BANK_INPUTS,
}
SHAPES = {
    "image": (1, 3, 1024, 1024), "raw_vision_features": (1, 256, 64, 64),
    "initial_vision_features": (1, 256, 64, 64), "high_res_feature_0": (1, 32, 256, 256),
    "high_res_feature_1": (1, 64, 128, 128), "point_coords": (1, 1, 2), "point_labels": (1, 1),
    "low_res_mask": (1, 1, 256, 256), "high_res_mask": (1, 1, 1024, 1024),
    "best_iou": (1,), "object_score": (1, 1), "object_pointer": (1, 256),
    "memory_features": (1, 512, 64), "memory_positions": (1, 512, 64),
    "spatial_bank": (1, 7, 512, 64), "spatial_positions": (1, 7, 512, 64),
    "pointer_bank": (1, 16, 256), "valid_slots": (1, 23),
}


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def video_frames(path):
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        if stream.sample_aspect_ratio not in (None, Fraction(1, 1)):
            raise ValueError("Use a square-pixel video for this comparison.")
        previous = None
        count = 0
        for frame in container.decode(stream):
            if frame.pts is None or frame.time_base is None:
                raise ValueError("Frame has no presentation timestamp.")
            timestamp = Fraction(frame.pts) * frame.time_base
            if previous is not None and timestamp <= previous:
                raise ValueError("Video timestamps are not strictly increasing.")
            previous = timestamp
            rotation = frame.rotation
            if rotation % 90:
                raise ValueError("Only quarter-turn display rotation is supported by this fixture reader.")
            for side in frame.side_data:
                if side.type.name == "DISPLAYMATRIX":
                    matrix = np.frombuffer(side, dtype=np.int32).astype(np.float64).reshape(3, 3)
                    if np.linalg.det(matrix[:2, :2]) <= 0:
                        raise ValueError("Mirrored display matrices need a dedicated fixture reader.")
            # PyAV 14.4 exposes DISPLAYMATRIX rotation in counterclockwise degrees.
            image = frame.to_image().convert("RGB").rotate(rotation, expand=True)
            image = image.resize((1024, 1024), Image.Resampling.BILINEAR)
            pixels = np.asarray(image)
            value = pixels.astype(np.float32).transpose(2, 0, 1)[None] / 255.0
            yield count, timestamp, image, value, hashlib.sha256(pixels.tobytes()).hexdigest()
            count += 1
            if count == COUNT:
                return
        raise ValueError(f"Need at least {COUNT} decoded frames; found {count}.")


class TorchBackend:
    def __init__(self, modules):
        self.modules = modules

    def call(self, component, inputs):
        values = tuple(torch.from_numpy(np.ascontiguousarray(inputs[name])) for name in INPUTS[component])
        outputs = self.modules[component](*values)
        return dict(zip(NAMES[component], (x.detach().cpu().numpy() for x in outputs), strict=True))


class CoreMLBackend:
    def __init__(self, directory):
        import coremltools as ct
        self.models = {name: ct.models.MLModel(str(directory / f"BWTemporal{name}.mlpackage"),
                                             compute_units=ct.ComputeUnit.CPU_ONLY) for name in INPUTS}

    def call(self, component, inputs):
        if component == "ImageEncoder":
            values = {"image": inputs["pil_image"]}
        else:
            values = {name: np.asarray(inputs[name], dtype=np.int32 if name == "point_labels" else np.float16)
                      for name in INPUTS[component]}
        output = self.models[component].predict(values)
        return {name: np.asarray(output[name], dtype=np.float32) for name in NAMES[component]}


def cosine(left, right):
    left, right = left.astype(np.float64).ravel(), right.astype(np.float64).ravel()
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    if denominator == 0:
        return 1.0 if np.array_equal(left, right) else 0.0
    return float(np.dot(left, right) / denominator)


def compare(reference, actual, coreml):
    checks = {}
    passed = True
    for name in owned.MASK_OUTPUTS + owned.MEMORY_OUTPUTS:
        left, right = np.asarray(reference[name]), np.asarray(actual[name])
        if left.shape != SHAPES[name] or right.shape != SHAPES[name]:
            raise ValueError(f"Unexpected output shape for {name}")
        if not np.isfinite(left).all() or not np.isfinite(right).all():
            raise ValueError(f"Non-finite output for {name}")
        close = bool(np.allclose(left, right, atol=0.001, rtol=0.001))
        similarity = cosine(left, right)
        checks[name] = {"maximumAbsoluteError": float(np.max(np.abs(left - right))),
                        "cosineSimilarity": similarity, "float32Close": close}
        if not coreml:
            passed &= close
        elif name in ("low_res_mask", "high_res_mask", "memory_features", "memory_positions", "object_pointer"):
            passed &= similarity >= 0.99
    a, b = reference["low_res_mask"] > 0, actual["low_res_mask"] > 0
    union = int(np.logical_or(a, b).sum())
    mask_iou = float(np.logical_and(a, b).sum() / union) if union else 1.0
    passed &= mask_iou >= (0.95 if coreml else 0.999)
    same_presence = bool(np.array_equal(reference["object_score"] > 0, actual["object_score"] > 0))
    passed &= same_presence
    return {"passed": bool(passed), "maskIoUAgainstReference": mask_iou,
            "sameObjectPresence": same_presence, "outputs": checks}


def reference_step(model, image_tensor, point, index, history):
    output = model.forward_image(owned.normalize(image_tensor))
    _, features, positions, sizes = model._prepare_backbone_features(output)
    prompts = {"point_coords": torch.tensor([[point]], dtype=torch.float32),
               "point_labels": torch.ones((1, 1), dtype=torch.int32)} if index == 0 else None
    # Original dynamic memory selection and original complex-valued RoPE.
    result = model.track_step(index, index == 0, features, positions, sizes,
                              prompts, None, history, num_frames=COUNT)
    # track_step does not expose IoU; obtain the SAM head result independently from
    # its _track_step helper, with the same pre-commit history. No state mutation.
    _, sam, _, _ = model._track_step(index, index == 0, features, positions, sizes,
                                   prompts, None, history, COUNT, False, None)
    reference = {"low_res_mask": result["pred_masks"], "high_res_mask": result["pred_masks_high_res"],
                 "object_pointer": result["obj_ptr"], "object_score": result["object_score_logits"],
                 "best_iou": sam[2].max(dim=-1).values,
                 "memory_features": result["maskmem_features"],
                 "memory_positions": result["maskmem_pos_enc"][0]}
    # Reference keeps all 20 entries so its history selection is independent of
    # the candidate's bounded ring. It does not retain full-resolution masks.
    bucket = "cond_frame_outputs" if index == 0 else "non_cond_frame_outputs"
    history[bucket][index] = {key: result[key] for key in ("maskmem_features", "maskmem_pos_enc", "obj_ptr")}
    return {name: tensor.detach().cpu().numpy() for name, tensor in reference.items()}


def save_mask(path, mask):
    Image.fromarray((mask[0, 0] > 0).astype(np.uint8) * 255).save(path)


def run_comparison(model, backend, args, root, coreml, expected_frames=None):
    state = TemporalState()
    history = {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}}
    stage = "coreml" if coreml else "pytorch"
    directory = root / stage
    directory.mkdir()
    report = {"passed": False, "contract": owned.CONTRACT, "frames": [],
              "scope": "20 consecutive frames; correctness fixture, not a speed or general-quality benchmark",
              "policy": {"maskIoUMinimum": 0.95 if coreml else 0.999,
                         "cosineMinimumCoreML": 0.99, "float32Atol": 0.001, "float32Rtol": 0.001},
              "coremlComputeUnits": "CPU_ONLY" if coreml else None}
    point = [min(max(float(v) * 1024, 0), 1023) for v in args.point]
    try:
        with torch.inference_mode():
            for index, timestamp, image, tensor, digest in video_frames(args.video):
                print(f"{stage}: frame {index + 1}/{COUNT}", flush=True)
                identity = (digest, timestamp.numerator, timestamp.denominator)
                if expected_frames is not None and identity != expected_frames[index]:
                    raise ValueError("Decoded frame pixels or timestamps changed between comparisons.")
                state.validate_time(timestamp)
                token = state.token
                start = time.perf_counter()
                features = backend.call("ImageEncoder", {"image": tensor, "pil_image": image})
                if index == 0:
                    inputs = {**features, "point_coords": np.array([[point]], dtype=np.float32),
                              "point_labels": np.ones((1, 1), dtype=np.int32)}
                    result = backend.call("Initializer", inputs)
                    result.update(backend.call("InitialMemoryEncoder", {**features, **result}))
                else:
                    result = backend.call("Propagator", {**features, **state.pack(timestamp)})
                elapsed = time.perf_counter() - start
                reference = reference_step(model, torch.from_numpy(tensor), point, index, history)
                comparison = compare(reference, result, coreml)
                if index == 0 and not (reference["low_res_mask"] > 0).any():
                    raise ValueError("The selected point gives an empty reference mask. Choose a point inside the subject.")
                state.commit(result, timestamp, token)
                row = {"index": index, "ptsNumerator": timestamp.numerator,
                       "ptsDenominator": timestamp.denominator, "imageSHA256": digest,
                       "candidateWallSecondsDiagnosticOnly": elapsed, "state": state.summary(), **comparison}
                report["frames"].append(row)
                image.save(directory / f"frame-{index:02d}.png")
                save_mask(directory / f"candidate-{index:02d}.png", result["low_res_mask"])
                save_mask(directory / f"reference-{index:02d}.png", reference["low_res_mask"])
                write_json(directory / "report.json", report)
                if not comparison["passed"]:
                    raise ValueError(f"{stage} parity failed at frame {index}; inspect report.json and masks.")
        if len(report["frames"]) != COUNT or state.summary()["spatialEntries"] != 7 or state.summary()["pointerEntries"] != 16:
            raise ValueError("The fixture did not exercise full memory and pointer banks.")
        report["passed"] = True
    except Exception as error:
        report["error"] = str(error)
        raise
    finally:
        write_json(directory / "report.json", report)
    return [(frame["imageSHA256"], frame["ptsNumerator"], frame["ptsDenominator"])
            for frame in report["frames"]]


def export_models(modules, destination):
    import coremltools as ct
    destination.mkdir()
    torch.manual_seed(0)
    manifest = {"contract": owned.CONTRACT, "models": {}}
    for name, module in modules.items():
        print(f"Exporting {name}…", flush=True)
        examples = []
        descriptions = []
        for key in INPUTS[name]:
            shape = SHAPES[key]
            if key == "point_labels":
                example = torch.ones(shape, dtype=torch.int32)
            elif key == "valid_slots":
                example = torch.ones(shape)
            else:
                example = torch.zeros(shape)
            examples.append(example)
            if key == "image":
                descriptions.append(ct.ImageType(name=key, shape=shape, scale=1 / 255.0,
                                                  color_layout=ct.colorlayout.RGB))
            else:
                descriptions.append(ct.TensorType(name=key, shape=shape,
                                                  dtype=np.int32 if key == "point_labels" else np.float16))
        with torch.inference_mode():
            traced = torch.jit.trace(module, tuple(examples), strict=True, check_trace=False)
        # The separate real-frame comparisons are mandatory; tracing alone cannot
        # validate dynamic label/mask choices, startup padding or temporal eviction.
        converted = ct.convert(traced, source="pytorch", convert_to="mlprogram", inputs=descriptions,
                               outputs=[ct.TensorType(name=key, dtype=np.float16) for key in NAMES[name]],
                               compute_precision=ct.precision.FLOAT16,
                               minimum_deployment_target=ct.target.iOS18,
                               skip_model_load=True)
        converted.user_defined_metadata["bytewave.contract"] = owned.CONTRACT
        converted.user_defined_metadata["bytewave.upstream"] = owned.UPSTREAM_REVISION
        converted.user_defined_metadata["bytewave.checkpoint.sha256"] = owned.CHECKPOINT_SHA256
        package = destination / f"BWTemporal{name}.mlpackage"
        converted.save(str(package))
        del converted, traced
        hashes = {}
        for file in sorted(package.rglob("*")):
            if file.is_file():
                with file.open("rb") as stream:
                    hashes[str(file.relative_to(package))] = hashlib.file_digest(stream, "sha256").hexdigest()
        manifest["models"][name] = {"files": hashes, "inputs": list(INPUTS[name]), "outputs": list(NAMES[name])}
        write_json(destination / "manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--point", type=float, nargs=2, metavar=("X", "Y"), required=True,
                        help="Normalized top-left coordinates on the display-oriented first frame")
    parser.add_argument("--output", type=Path, required=True, help="New directory; existing directories are never overwritten")
    parser.add_argument("--reference-only", action="store_true", help="Skip Core ML export; does not establish deployment readiness")
    args = parser.parse_args()
    if not all(np.isfinite(v) and 0 <= v <= 1 for v in args.point):
        parser.error("Point coordinates must be finite and in [0,1].")
    if not args.video.is_file():
        parser.error("Video does not exist.")
    if not args.reference_only and platform.system() != "Darwin":
        parser.error("Core ML prediction comparison requires macOS. Use --reference-only elsewhere.")
    args.output.mkdir(parents=True, exist_ok=False)
    versions = {name: importlib.metadata.version(name) for name in ("torch", "torchvision", "numpy", "av", "timm")}
    if not args.reference_only:
        versions["coremltools"] = importlib.metadata.version("coremltools")
    status = {"contract": owned.CONTRACT, "upstream": owned.UPSTREAM_REVISION,
              "checkpointSHA256": owned.CHECKPOINT_SHA256, "python": sys.version,
              "versions": versions, "pointNormalizedTopLeft": args.point,
              "referencePassed": False, "coremlPassed": False, "readyForDeviceValidation": False}
    write_json(args.output / "status.json", status)
    try:
        model = owned.load_reference(args.upstream)
        modules = owned.components(model)
        frames = run_comparison(model, TorchBackend(modules), args, args.output, coreml=False)
        status["referencePassed"] = True
        write_json(args.output / "status.json", status)
        if not args.reference_only:
            export_models(modules, args.output / "models")
            run_comparison(model, CoreMLBackend(args.output / "models"), args, args.output, coreml=True,
                           expected_frames=frames)
            status["coremlPassed"] = True
            status["readyForDeviceValidation"] = True
    except Exception as error:
        status["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        write_json(args.output / "status.json", status)
    print(f"Finished: {args.output / 'status.json'}", flush=True)


if __name__ == "__main__":
    main()
