#!/usr/bin/env python3
"""Package a passed Mac run for the fixed 20-frame iPhone temporal comparison.

Run on the user's Mac. Replays Core ML CPU predictions to save reference tensors;
does not export models, alter weights, or prepare masks for a full video.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import platform
import shutil

import numpy as np
from PIL import Image
import torch

import owned
import validate_export as v
from state import TemporalState

TENSORS = ("low_res_mask", "best_iou", "object_pointer", "object_score",
           "memory_features", "memory_positions")


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New directory only")
    args = parser.parse_args()
    if platform.system() != "Darwin":
        parser.error("Core ML CPU reference preparation requires macOS.")
    status = json.loads((args.run / "status.json").read_text())
    report = json.loads((args.run / "coreml/report.json").read_text())
    reference = json.loads((args.run / "pytorch/report.json").read_text())
    manifest_path = args.run / "models/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if (status.get("contract") != owned.CONTRACT
            or status.get("precisionPolicy") != v.COREML_PRECISION_POLICY
            or status.get("upstream") != owned.UPSTREAM_REVISION
            or status.get("checkpointSHA256") != owned.CHECKPOINT_SHA256
            or not all(status.get(key) is True for key in
                       ("referencePassed", "coremlPassed", "readyForDeviceValidation"))):
        raise ValueError("A passed complete run with the current contract/policy is required.")
    for item in (reference, report):
        if (item.get("passed") is not True or len(item.get("frames", [])) != v.COUNT
                or not all(row.get("passed") is True for row in item["frames"])):
            raise ValueError("Both complete 20-frame comparison reports must pass.")
    if manifest.get("contract") != owned.CONTRACT or manifest.get("precisionPolicy") != v.COREML_PRECISION_POLICY:
        raise ValueError("Model manifest does not match the passed policy.")
    model_files = {}
    for component in v.INPUTS:
        package_name = f"BWTemporal{component}.mlpackage"
        package = args.run / "models" / package_name
        expected = manifest["models"][component]["files"]
        actual = {str(p.relative_to(package)): sha(p) for p in package.rglob("*") if p.is_file()}
        if actual != expected:
            raise ValueError(f"Model bytes changed after export: {component}")
        model_files.update({f"models/{package_name}/{key}": digest for key, digest in actual.items()})
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    args.output.mkdir(parents=True, exist_ok=False)
    preparation = {"complete": False, "sourceStatus": status, "frames": [],
                   "scope": "20 saved frames replayed with Core ML CPU; references for a device diagnostic only."}
    v.write_json(args.output / "preparation-report.json", preparation)
    try:
        backend = v.CoreMLBackend(args.run / "models")
        state = TemporalState()
        point = [min(float(x) * 1024, 1023) for x in status["pointNormalizedTopLeft"]]
        fixtures = []
        with torch.inference_mode():
            for index, row in enumerate(report["frames"]):
                print(f"Device reference: frame {index + 1}/{v.COUNT}", flush=True)
                original = reference["frames"][index]
                identity = (row["index"], row["ptsNumerator"], row["ptsDenominator"], row["imageSHA256"])
                if identity != tuple(original[key] for key in
                                     ("index", "ptsNumerator", "ptsDenominator", "imageSHA256")) or row["index"] != index:
                    raise ValueError("Source frame identities differ between comparisons.")
                if not 0 < row["ptsDenominator"] <= 2**31 - 1 or not -(2**63) <= row["ptsNumerator"] < 2**63:
                    raise ValueError("Timestamp cannot be represented exactly as CMTime.")
                timestamp = Fraction(row["ptsNumerator"], row["ptsDenominator"])
                image = Image.open(args.run / "coreml" / f"frame-{index:02d}.png").convert("RGB")
                pixels = np.asarray(image)
                if pixels.shape != (1024, 1024, 3) or hashlib.sha256(pixels.tobytes()).hexdigest() != row["imageSHA256"]:
                    raise ValueError("Saved frame pixels do not match the passed run.")
                features = backend.call("ImageEncoder", {"pil_image": image})
                token = state.token
                if index == 0:
                    result = backend.call("Initializer", {**features,
                        "point_coords": np.asarray([[point]], dtype=np.float32),
                        "point_labels": np.ones((1, 1), dtype=np.int32)})
                    result.update(backend.call("InitialMemoryEncoder", {**features, **result}))
                else:
                    result = backend.call("Propagator", {**features, **state.pack(timestamp)})
                saved_mask = np.asarray(Image.open(args.run / "coreml" / f"candidate-{index:02d}.png")) > 0
                current_mask = result["low_res_mask"][0, 0] > 0
                union = np.logical_or(saved_mask, current_mask).sum()
                iou = float(np.logical_and(saved_mask, current_mask).sum() / union) if union else 1.0
                if iou < 0.999:
                    raise ValueError(f"CPU replay changed the saved mask at frame {index}: IoU {iou}")
                state.commit(result, timestamp, token)
                folder = args.output / "frames" / f"{index:02d}"
                folder.mkdir(parents=True)
                image.save(folder / "preview.png")
                bgra = np.empty((1024, 1024, 4), dtype=np.uint8)
                bgra[..., :3] = pixels[..., ::-1]
                bgra[..., 3] = 255
                (folder / "input.bgra").write_bytes(bgra.tobytes())
                files = {}
                for name in TENSORS:
                    file = folder / f"{name}.f32"
                    file.write_bytes(np.asarray(result[name], dtype="<f4").tobytes(order="C"))
                    files[name] = {"path": str(file.relative_to(args.output)), "sha256": sha(file),
                                   "shape": list(v.SHAPES[name])}
                fixtures.append({"index": index, "ptsNumerator": row["ptsNumerator"],
                                 "ptsDenominator": row["ptsDenominator"], "imageSHA256": row["imageSHA256"],
                                 "inputPath": str((folder / "input.bgra").relative_to(args.output)),
                                 "inputSHA256": sha(folder / "input.bgra"),
                                 "previewPath": str((folder / "preview.png").relative_to(args.output)),
                                 "tensors": files})
                preparation["frames"].append({"index": index, "replayMaskIoU": iou, "state": state.summary()})
                v.write_json(args.output / "preparation-report.json", preparation)
        for component in v.INPUTS:
            name = f"BWTemporal{component}.mlpackage"
            shutil.copytree(args.run / "models" / name, args.output / "models" / name)
        if any(sha(args.output / path) != digest for path, digest in model_files.items()):
            raise ValueError("Copied model verification failed.")
        fixture = {"schema": "bytewave.temporal-device-fixture.v1", "contract": owned.CONTRACT,
                   "precisionPolicy": v.COREML_PRECISION_POLICY, "upstream": owned.UPSTREAM_REVISION,
                   "checkpointSHA256": owned.CHECKPOINT_SHA256,
                   "sourceModelsManifestSHA256": sha(manifest_path),
                   "sourceReportSHA256": sha(args.run / "coreml/report.json"),
                   "pointNormalizedTopLeft": status["pointNormalizedTopLeft"],
                   "referenceComputeUnits": "CPU_ONLY", "modelFiles": model_files, "frames": fixtures}
        # Published last: an interrupted preparation cannot look like a ready fixture.
        preparation["complete"] = True
        v.write_json(args.output / "preparation-report.json", preparation)
        v.write_json(args.output / "fixture.json", fixture)
    except Exception as error:
        preparation["complete"] = False
        preparation["error"] = f"{type(error).__name__}: {error}"
        v.write_json(args.output / "preparation-report.json", preparation)
        raise
    print(f"Ready: {args.output / 'preparation-report.json'}", flush=True)


if __name__ == "__main__":
    main()
