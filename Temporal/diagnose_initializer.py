#!/usr/bin/env python3
"""Compare existing FP16 and diagnostic FP32 initializer computation on one frame.

The diagnostic uses FP32 computation and interfaces with the same numerical
input values as the FP16 model. It never establishes device readiness.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import platform

import numpy as np
import torch

import owned
from validate_export import INPUTS, cosine, video_frames, write_json


def summarize(outputs, reference=None):
    checks = {}
    for name, value in outputs.items():
        value = np.asarray(value, dtype=np.float32)
        finite = np.isfinite(value)
        subset = value[finite]
        row = {"shape": list(value.shape), "nonfinite": int(value.size - finite.sum()),
               "minimum": float(subset.min()) if subset.size else None,
               "maximum": float(subset.max()) if subset.size else None}
        if reference is not None and finite.all() and np.isfinite(reference[name]).all():
            expected = reference[name]
            if expected.shape != value.shape:
                raise ValueError(f"Unexpected shape for {name}: {value.shape}")
            row["maximumAbsoluteError"] = float(np.max(np.abs(value - expected)))
            row["cosineSimilarity"] = cosine(expected, value)
            if name in ("low_res_mask", "high_res_mask"):
                a, b = expected > 0, value > 0
                union = int(np.logical_or(a, b).sum())
                row["maskIoUAgainstSameInputTorch"] = float(np.logical_and(a, b).sum() / union) if union else 1.0
        checks[name] = row
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--point", type=float, nargs=2, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if platform.system() != "Darwin":
        parser.error("Core ML prediction requires macOS.")
    if not all(np.isfinite(v) and 0 <= v <= 1 for v in args.point):
        parser.error("Point coordinates must be finite and in [0,1].")
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"contract": owned.CONTRACT, "readyForDeviceValidation": False,
              "scope": "One-frame initializer precision diagnostic; FP32 computation and interfaces, identical numerical inputs",
              "upstream": owned.UPSTREAM_REVISION, "checkpointSHA256": owned.CHECKPOINT_SHA256,
              "pointNormalizedTopLeft": args.point}
    try:
        import coremltools as ct
        report["versions"] = {"coremltools": ct.__version__, "torch": torch.__version__}
        models = {}
        for name in ("ImageEncoder", "Initializer"):
            model = ct.models.MLModel(str(args.models / f"BWTemporal{name}.mlpackage"),
                                     compute_units=ct.ComputeUnit.CPU_ONLY)
            for key, expected in (("bytewave.contract", owned.CONTRACT),
                                  ("bytewave.upstream", owned.UPSTREAM_REVISION),
                                  ("bytewave.checkpoint.sha256", owned.CHECKPOINT_SHA256)):
                if model.user_defined_metadata.get(key) != expected:
                    raise ValueError(f"Unexpected {name} metadata: {key}")
            models[name] = model
        frames = video_frames(args.video)
        try:
            _, timestamp, image, _, digest = next(frames)
        finally:
            frames.close()
        report["frame"] = {"imageSHA256": digest, "ptsNumerator": timestamp.numerator,
                           "ptsDenominator": timestamp.denominator}
        features = models["ImageEncoder"].predict({"image": image})
        report["encoder"] = summarize(features)
        if any(row["nonfinite"] for row in report["encoder"].values()):
            raise ValueError("Encoder output is non-finite; initializer comparison cannot isolate the cause.")
        point = [min(float(v) * 1024, 1023) for v in args.point]
        inputs = {name: np.asarray(features[name], dtype=np.float16)
                  for name in INPUTS["Initializer"][:3]}
        inputs.update(point_coords=np.asarray([[point]], dtype=np.float16),
                      point_labels=np.ones((1, 1), dtype=np.int32))
        module = owned.Initializer(owned.load_reference(args.upstream)).eval()
        # Preserve the exact rounded FP16 input values in the FP32 reference.
        fp32_inputs = {name: value.astype(np.int32 if name == "point_labels" else np.float32)
                       for name, value in inputs.items()}
        examples = tuple(torch.from_numpy(fp32_inputs[name]) for name in INPUTS["Initializer"])
        with torch.inference_mode():
            expected = dict(zip(owned.MASK_OUTPUTS,
                                (v.detach().cpu().numpy() for v in module(*examples)), strict=True))
        report["sameInputTorch"] = summarize(expected)
        if any(row["nonfinite"] for row in report["sameInputTorch"].values()):
            raise ValueError("Same-input PyTorch output is non-finite.")
        report["existingFP16"] = summarize(models["Initializer"].predict(inputs), expected)
        write_json(args.output / "report.json", report)
        print("Exporting diagnostic initializer with FP32 computation and interfaces...", flush=True)
        with torch.inference_mode():
            traced = torch.jit.trace(module, examples, strict=True, check_trace=False)
        converted = ct.convert(
            traced, source="pytorch", convert_to="mlprogram",
            inputs=[ct.TensorType(name=name, shape=tuple(value.shape),
                                  dtype=np.int32 if name == "point_labels" else np.float32)
                    for name, value in zip(INPUTS["Initializer"], examples, strict=True)],
            outputs=[ct.TensorType(name=name, dtype=np.float32) for name in owned.MASK_OUTPUTS],
            compute_precision=ct.precision.FLOAT32, minimum_deployment_target=ct.target.iOS18,
            skip_model_load=True)
        converted.user_defined_metadata["bytewave.diagnostic"] = "initializer-fp32-compute-and-io"
        package = args.output / "DiagnosticInitializerFP32.mlpackage"
        converted.save(str(package))
        diagnostic = ct.models.MLModel(str(package), compute_units=ct.ComputeUnit.CPU_ONLY)
        report["diagnosticFP32"] = summarize(diagnostic.predict(fp32_inputs), expected)
        for variant in ("sameInputTorch", "existingFP16", "diagnosticFP32"):
            print(variant, {name: row["nonfinite"] for name, row in report[variant].items()}, flush=True)
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        write_json(args.output / "report.json", report)
    print(f"Diagnostic saved: {args.output / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()
