#!/usr/bin/env python3
"""Compare existing FP16 and diagnostic FP32 initializer computation on one frame.

The diagnostic uses FP32 computation and interfaces with the same numerical
input values as the FP16 model. It never establishes device readiness.
The optional sweep separates tensor-interface precision from computation and
checks normalization and attention precision on the same traced graph.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import platform

import numpy as np
import torch

import owned
from validate_export import INPUTS, cosine, video_frames, write_json

# This diagnostic deliberately reads the original failing export, including its
# FP16 interfaces. New v2 packages are validated by the full temporal command.
SOURCE_CONTRACT = "bytewave.edgetam-temporal-owned.v1"


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


def convert_variant(ct, traced, examples, inputs, expected, directory, name, dtype,
                    compute_precision, preserve_types=()):
    print(f"Exporting {name}...", flush=True)
    retained_ops = []
    if preserve_types:
        def select(op):
            if op.op_type in preserve_types:
                retained_ops.append({"name": op.name, "type": op.op_type})
                return False
            return True
        compute_precision = ct.transform.FP16ComputePrecision(op_selector=select)
    converted = ct.convert(
        traced, source="pytorch", convert_to="mlprogram",
        inputs=[ct.TensorType(name=key, shape=tuple(value.shape),
                              dtype=np.int32 if key == "point_labels" else dtype)
                for key, value in zip(INPUTS["Initializer"], examples, strict=True)],
        outputs=[ct.TensorType(name=key, dtype=dtype) for key in owned.MASK_OUTPUTS],
        compute_precision=compute_precision, minimum_deployment_target=ct.target.iOS18,
        skip_model_load=True)
    converted.user_defined_metadata["bytewave.diagnostic"] = name
    package = directory / f"{name}.mlpackage"
    converted.save(str(package))
    del converted
    diagnostic = ct.models.MLModel(str(package), compute_units=ct.ComputeUnit.CPU_ONLY)
    values = {key: value.astype(np.int32 if key == "point_labels" else dtype)
              for key, value in inputs.items()}
    outputs = summarize(diagnostic.predict(values), expected)
    print(name, {key: row["nonfinite"] for key, row in outputs.items()}, flush=True)
    return outputs, {"tensorInterface": np.dtype(dtype).name,
                     "requestedFP32OperationTypes": sorted(preserve_types),
                     "operationsExcludedFromFP16Transform": retained_ops}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--point", type=float, nargs=2, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--precision-sweep", action="store_true",
                        help="Compare interface, normalization and attention precision using one trace")
    args = parser.parse_args()
    if platform.system() != "Darwin":
        parser.error("Core ML prediction requires macOS.")
    if not all(np.isfinite(v) and 0 <= v <= 1 for v in args.point):
        parser.error("Point coordinates must be finite and in [0,1].")
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"contract": SOURCE_CONTRACT, "readyForDeviceValidation": False,
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
            for key, expected in (("bytewave.contract", SOURCE_CONTRACT),
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
        # Match the production export's zero-valued examples with positive labels.
        # Predict on the real inputs afterward to check trace specialization.
        trace_examples = tuple(torch.ones_like(value) if name == "point_labels" else torch.zeros_like(value)
                               for name, value in zip(INPUTS["Initializer"], examples, strict=True))
        with torch.inference_mode():
            traced = torch.jit.trace(module, trace_examples, strict=True, check_trace=False)
            traced_outputs = dict(zip(owned.MASK_OUTPUTS,
                                      (v.detach().cpu().numpy() for v in traced(*examples)), strict=True))
        report["sameInputTracedTorch"] = summarize(traced_outputs, expected)
        report["traceExamples"] = "Zeros, except point_labels=1; same trace used for every precision variant"
        if any(not np.allclose(traced_outputs[name], expected[name], atol=0.001, rtol=0.001)
               for name in owned.MASK_OUTPUTS):
            raise ValueError("Traced PyTorch differs from eager PyTorch on the real inputs.")
        variants = [("diagnosticFP32", np.float32, ct.precision.FLOAT32, ())]
        if args.precision_sweep:
            norm_types = ("layer_norm", "instance_norm", "batch_norm", "l2_norm",
                          "reduce_mean", "reduce_sum", "square", "pow", "sqrt", "rsqrt", "real_div")
            attention_types = ("matmul", "softmax", "scaled_dot_product_attention")
            variants += [("sameTraceFP16", np.float16, ct.precision.FLOAT16, ()),
                         ("fp16ComputeFP32IO", np.float32, ct.precision.FLOAT16, ()),
                         ("fp32Normalization", np.float32, None, norm_types),
                         ("fp32Attention", np.float32, None, attention_types)]
        report["precisionPolicies"] = {}
        for name, dtype, precision, preserve in variants:
            try:
                outputs, policy = convert_variant(ct, traced, examples, fp32_inputs, expected,
                                                  args.output, name, dtype, precision, preserve)
                report[name] = outputs
                report["precisionPolicies"][name] = {"compute": "mixed" if preserve else
                                                    ("float32" if precision == ct.precision.FLOAT32 else "float16"),
                                                    **policy}
            except Exception as error:
                report[name] = {"error": f"{type(error).__name__}: {error}"}
                print(f"{name}: {report[name]['error']}", flush=True)
                if not args.precision_sweep:
                    raise
            finally:
                write_json(args.output / "report.json", report)
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        write_json(args.output / "report.json", report)
    print(f"Diagnostic saved: {args.output / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()
