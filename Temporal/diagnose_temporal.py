#!/usr/bin/env python3
"""Localize temporal conversion drift using existing models; no export or readiness claim."""
from __future__ import annotations

import argparse
import hashlib
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--point", type=float, nargs=2, required=True)
    parser.add_argument("--frames", type=int, default=3)
    parser.add_argument("--torch-image-encoder", action="store_true",
                        help="Control experiment: feed PyTorch image features to the existing Core ML tracker")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if platform.system() != "Darwin":
        parser.error("Core ML predictions require macOS.")
    if not 1 <= args.frames <= v.COUNT:
        parser.error(f"Choose between 1 and {v.COUNT} frames.")
    if not all(np.isfinite(x) and 0 <= x <= 1 for x in args.point):
        parser.error("Point must be finite and in [0,1].")
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"contract": owned.CONTRACT, "readyForDeviceValidation": False,
              "diagnosticComplete": False, "frames": [],
              "scope": "Each Core ML component is compared to PyTorch with the same actual inputs. "
                       "Encoder output source is explicit below; masks and stored state come from Core ML. End-to-end comparison "
                       "uses separate original PyTorch history. No parity gates are changed.",
              "encoderOutputSource": "pytorch-control" if args.torch_image_encoder else "coreml",
              "pointNormalizedTopLeft": args.point,
              "coremlComputeUnits": "CPU_ONLY", "precisionPolicy": v.COREML_PRECISION_POLICY}
    decoder = None
    try:
        manifest = args.models / "manifest.json"
        report["modelsManifestSHA256"] = hashlib.sha256(manifest.read_bytes()).hexdigest()
        model = owned.load_reference(args.upstream)
        reference_backend = v.TorchBackend(owned.components(model))
        backend = v.CoreMLBackend(args.models)
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
                        "outputFedToNextComponent": "pytorch-control" if
                        component == "ImageEncoder" and args.torch_image_encoder else "coreml"}
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
