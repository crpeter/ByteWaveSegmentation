#!/usr/bin/env python3
"""Localize temporal drift using existing tracker models and optional encoder controls.

Each encoder precision control exports one diagnostic package. No mode establishes
device readiness or replaces the existing model packages.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import torch

import owned
import validate_export as v
from diagnose_initializer import summarize
from state import TemporalState


def capture_torch_head(model, prediction):
    heads = []

    def capture(_module, _inputs, output):
        scores = output[1].detach().cpu().numpy()
        ordered = np.sort(scores, axis=-1)
        heads.append({"iouEstimates": scores.tolist(),
                      "selectedMaskIndex": np.argmax(scores, axis=-1).tolist(),
                      "topTwoMargin": (ordered[:, -1] - ordered[:, -2]).tolist()
                      if scores.shape[-1] > 1 else None})

    handle = model.sam_mask_decoder.register_forward_hook(capture)
    try:
        result = prediction()
    finally:
        handle.remove()
    return result, heads


def export_encoder_control(module, directory, fp16_conv=False, fp32_conv_range=None,
                           expected_convs=None):
    import coremltools as ct
    from coremltools.converters.mil.mil.scope import ScopeSource
    policy = "fp32-with-fp16-conv.v1" if fp16_conv else "float32"
    if fp32_conv_range is not None:
        policy = "fp32-with-selective-fp16-conv.v1"
    print(f"Exporting diagnostic Core ML image encoder ({policy})...", flush=True)
    converted_convs = []
    retained_convs = []
    conv_count = 0

    def select_conv(op):
        nonlocal conv_count
        # Start from the successful FP32 baseline and lower only convolution
        # operations (including their inputs/weights). Other arithmetic stays
        # FP32. This tests a group; it does not identify a particular layer.
        if op.op_type == "conv":
            index = conv_count
            conv_count += 1
            entry = {"index": index, "name": op.name, "type": op.op_type,
                     "moduleScopes": op.scopes.get(ScopeSource.TORCHSCRIPT_MODULE_NAME, [])}
            if fp32_conv_range is not None and fp32_conv_range[0] <= index < fp32_conv_range[1]:
                retained_convs.append(entry)
                return False
            converted_convs.append(entry)
            return True
        return False

    with torch.inference_mode():
        traced = torch.jit.trace(module, (torch.zeros(v.SHAPES["image"]),),
                                 strict=True, check_trace=False)
    converted = ct.convert(
        traced, source="pytorch", convert_to="mlprogram",
        inputs=[ct.ImageType(name="image", shape=v.SHAPES["image"], scale=1 / 255.0,
                             color_layout=ct.colorlayout.RGB)],
        outputs=[ct.TensorType(name=name, dtype=np.float32) for name in owned.IMAGE_OUTPUTS],
        compute_precision=(ct.transform.FP16ComputePrecision(op_selector=select_conv)
                           if fp16_conv else ct.precision.FLOAT32),
        minimum_deployment_target=ct.target.iOS18,
        skip_model_load=True)
    if fp16_conv and not converted_convs:
        raise ValueError("No encoder convolutions matched; precision control was not applied.")
    if expected_convs is not None and conv_count != expected_convs:
        raise ValueError(f"Expected {expected_convs} encoder convolutions, found {conv_count}; review graph ordering.")
    if fp32_conv_range is not None and len(retained_convs) != fp32_conv_range[1] - fp32_conv_range[0]:
        raise ValueError("Requested FP32 convolution range was not fully matched.")
    converted.user_defined_metadata["bytewave.diagnostic"] = f"image-encoder-{policy}-control"
    converted.user_defined_metadata["bytewave.upstream"] = owned.UPSTREAM_REVISION
    converted.user_defined_metadata["bytewave.checkpoint.sha256"] = owned.CHECKPOINT_SHA256
    package = directory / ("DiagnosticImageEncoderFP16Conv.mlpackage" if fp16_conv
                           else "DiagnosticImageEncoderFP32.mlpackage")
    converted.save(str(package))
    del converted, traced
    hashes = {}
    for file in sorted(package.rglob("*")):
        if file.is_file():
            with file.open("rb") as stream:
                hashes[str(file.relative_to(package))] = hashlib.file_digest(stream, "sha256").hexdigest()
    details = {"computePrecision": policy, "tensorOutputs": "float32", "files": hashes,
               "convolutionsSelectedForFP16Transform": converted_convs,
               "convolutionsRetainedInFP32": retained_convs,
               "fp32ConvolutionRangeStartInclusiveEndExclusive": fp32_conv_range,
               "expectedConvolutionCount": expected_convs}
    return ct.models.MLModel(str(package), compute_units=ct.ComputeUnit.CPU_ONLY), details


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--point", type=float, nargs=2, required=True)
    parser.add_argument("--frames", type=int, default=3)
    encoder_control = parser.add_mutually_exclusive_group()
    encoder_control.add_argument("--torch-image-encoder", action="store_true",
                                 help="Control experiment: feed PyTorch image features to the existing Core ML tracker")
    encoder_control.add_argument("--fp32-image-encoder", action="store_true",
                                 help="Export one diagnostic FP32 Core ML encoder and retain the existing tracker models")
    encoder_control.add_argument("--fp16-conv-image-encoder", action="store_true",
                                 help="Control: FP16 encoder convolutions with other operations in FP32; reuse the tracker")
    parser.add_argument("--fp32-conv-range", type=int, nargs=2, metavar=("START", "END"),
                        help="With --fp16-conv-image-encoder, retain this zero-based convolution range in FP32; END is exclusive")
    parser.add_argument("--expect-encoder-convs", type=int,
                        help="Require this convolution count before accepting an encoder precision control")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if platform.system() != "Darwin":
        parser.error("Core ML predictions require macOS.")
    if not 1 <= args.frames <= v.COUNT:
        parser.error(f"Choose between 1 and {v.COUNT} frames.")
    if not all(np.isfinite(x) and 0 <= x <= 1 for x in args.point):
        parser.error("Point must be finite and in [0,1].")
    if args.fp32_conv_range is not None:
        if not args.fp16_conv_image_encoder or args.expect_encoder_convs is None:
            parser.error("--fp32-conv-range requires --fp16-conv-image-encoder and --expect-encoder-convs.")
        start, end = args.fp32_conv_range
        if not 0 <= start < end <= args.expect_encoder_convs:
            parser.error("Convolution range must satisfy 0 <= START < END <= expected count.")
    if args.expect_encoder_convs is not None:
        if not args.fp16_conv_image_encoder or args.expect_encoder_convs <= 0:
            parser.error("--expect-encoder-convs requires the FP16 convolution control and a positive count.")
    # Diagnostic workaround for a native PyEval_SaveThread abort observed in
    # PyTorch CPU GELU while interleaving Core ML and PyTorch predictions on Mac.
    # Configure before any model work; the exact native root cause is unconfirmed.
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    args.output.mkdir(parents=True, exist_ok=False)
    encoder_source = "pytorch-control" if args.torch_image_encoder else (
        "coreml-fp32-control" if args.fp32_image_encoder else (
            "coreml-fp16-conv-control" if args.fp16_conv_image_encoder else "coreml"))
    report = {"contract": owned.CONTRACT, "readyForDeviceValidation": False,
              "diagnosticComplete": False, "frames": [],
              "scope": "Each Core ML component is compared to PyTorch with the same actual inputs. "
                       "Encoder output source is explicit below; masks and stored state come from Core ML. End-to-end comparison "
                       "uses separate original PyTorch history. No parity gates are changed.",
              "encoderOutputSource": encoder_source,
              "torchThreads": {"intraOp": torch.get_num_threads(),
                               "interOp": torch.get_num_interop_threads()},
              "pointNormalizedTopLeft": args.point,
              "coremlComputeUnits": "CPU_ONLY", "precisionPolicy": v.COREML_PRECISION_POLICY}
    v.write_json(args.output / "report.json", report)
    decoder = None
    try:
        manifest = args.models / "manifest.json"
        report["modelsManifestSHA256"] = hashlib.sha256(manifest.read_bytes()).hexdigest()
        source_policy = json.loads(manifest.read_text())["precisionPolicy"]
        if source_policy not in (v.TRACKER_PRECISION_POLICY, v.PREVIOUS_COREML_PRECISION_POLICY, v.COREML_PRECISION_POLICY):
            raise ValueError("Unsupported source model precision policy for this diagnostic.")
        report["precisionPolicy"] = source_policy
        model = owned.load_reference(args.upstream)
        reference_backend = v.TorchBackend(owned.components(model))
        backend = v.CoreMLBackend(args.models, precision_policy=source_policy)
        if args.fp32_image_encoder or args.fp16_conv_image_encoder:
            encoder, details = export_encoder_control(reference_backend.modules["ImageEncoder"], args.output,
                                                      fp16_conv=args.fp16_conv_image_encoder,
                                                      fp32_conv_range=args.fp32_conv_range,
                                                      expected_convs=args.expect_encoder_convs)
            # Deliberate diagnostic override after validating the original set.
            # All other components continue using the supplied model packages.
            backend.models["ImageEncoder"] = encoder
            report["encoderOverride"] = details
            v.write_json(args.output / "report.json", report)
        state = TemporalState()
        history = {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}}
        point = [min(float(x) * 1024, 1023) for x in args.point]
        decoder = v.video_frames(args.video)
        with torch.inference_mode():
            for index, timestamp, image, tensor, digest in decoder:
                print(f"Diagnostic: frame {index + 1}/{args.frames}", flush=True)
                state.validate_time(timestamp)
                token = state.token
                row = {"index": index, "ptsNumerator": timestamp.numerator,
                       "ptsDenominator": timestamp.denominator, "imageSHA256": digest,
                       "sameInputComponents": {}}
                report["frames"].append(row)

                def predict(component, inputs):
                    actual = backend.call(component, inputs)
                    expected, heads = capture_torch_head(
                        model, lambda: reference_backend.call(component, inputs))
                    row["sameInputComponents"][component] = {
                        "outputs": summarize(actual, expected), "torchMaskHeads": heads,
                        "outputFedToNextComponent": encoder_source if component == "ImageEncoder" else "coreml"}
                    if component == "ImageEncoder" and args.torch_image_encoder:
                        # Only the feature source changes. Keep Core ML masks,
                        # pointers and memories in the candidate state throughout.
                        return expected
                    return actual

                features = predict("ImageEncoder", {"image": tensor, "pil_image": image})
                if index == 0:
                    result = predict("Initializer", {**features,
                        "point_coords": np.asarray([[point]], dtype=np.float32),
                        "point_labels": np.ones((1, 1), dtype=np.int32)})
                    result.update(predict("InitialMemoryEncoder", {**features, **result}))
                else:
                    result = predict("Propagator", {**features, **state.pack(timestamp)})
                reference, heads = capture_torch_head(
                    model, lambda: v.reference_step(model, torch.from_numpy(tensor), point, index, history))
                row["independentReferenceMaskHeads"] = heads
                row["endToEnd"] = v.compare(reference, result, coreml=True)
                state.commit(result, timestamp, token)
                row["state"] = state.summary()
                v.write_json(args.output / "report.json", report)
                if index + 1 == args.frames:
                    break
        if len(report["frames"]) != args.frames:
            raise ValueError("The video did not supply the requested frames.")
        report["diagnosticComplete"] = True
        report["allRequestedFramesPassedComparison"] = all(row["endToEnd"]["passed"] for row in report["frames"])
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        if decoder is not None:
            decoder.close()
        v.write_json(args.output / "report.json", report)
    print(f"Diagnostic saved: {args.output / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()
