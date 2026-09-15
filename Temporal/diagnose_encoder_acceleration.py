#!/usr/bin/env python3
"""Compare existing and FP32 encoders on CPU and CPU + Neural Engine on one frame."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np
from PIL import Image
import torch

import owned
import validate_export as v
from diagnose_initializer import summarize
from diagnose_temporal import export_encoder_control


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def worker():
    import coremltools as ct
    package, image_path, output, units = sys.argv[2:]
    print(f'Loading ImageEncoder with {units}', flush=True)
    model = ct.models.MLModel(package, compute_units=getattr(ct.ComputeUnit, units))
    with Image.open(image_path) as image:
        print('Beginning ImageEncoder prediction', flush=True)
        values = model.predict({'image': image.convert('RGB')})
    np.savez(output, **values)
    print('Prediction returned', flush=True)


def compare_worker(package, frame, output, label, units, reference):
    log = output / f'{label}.log'
    values_path = output / f'{label}.npz'
    with log.open('w') as stream:
        try:
            result = subprocess.run([sys.executable, str(Path(__file__).resolve()), '__predict',
                                     str(package.resolve()), str(frame.resolve()), str(values_path.resolve()), units],
                                    stdout=stream, stderr=subprocess.STDOUT, timeout=300, check=False)
        except subprocess.TimeoutExpired:
            return {'error': 'Worker timed out after 300 seconds', 'log': log.name, 'passed': False}
    if result.returncode != 0:
        return {'error': 'Worker failed or aborted', 'returnCode': result.returncode, 'log': log.name, 'passed': False}
    with np.load(values_path, allow_pickle=False) as data:
        actual = dict(data)
    if set(actual) != set(owned.IMAGE_OUTPUTS) or any(actual[n].shape != v.SHAPES[n] for n in actual):
        return {'error': 'Unexpected encoder output names/shapes', 'log': log.name, 'passed': False}
    checks = summarize(actual, reference)
    finite = all(row['nonfinite'] == 0 for row in checks.values())
    return {'outputs': checks, 'passed': finite and all(row['cosineSimilarity'] >= .99 for row in checks.values()),
            'float32Close': finite and all(np.allclose(actual[n], reference[n], atol=.001, rtol=.001) for n in actual),
            'log': log.name}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if platform.system() != 'Darwin':
        parser.error('Run on Mac')
    import coremltools as ct
    status = json.loads((args.run / 'status.json').read_text())
    manifest_path = args.run / 'models/manifest.json'
    manifest = json.loads(manifest_path.read_text())
    for item in (status, manifest):
        if (item.get('contract') != owned.CONTRACT or item.get('graphRevision') != owned.GRAPH_REVISION
                or item.get('precisionPolicy') != v.COREML_PRECISION_POLICY):
            raise ValueError('Expected current graph/precision revision')
    if not all(status.get(k) is True for k in ('referencePassed', 'coremlPassed', 'readyForDeviceValidation')):
        raise ValueError('Source run must have passed normal validation')
    original = args.run / 'models/BWTemporalImageEncoder.mlpackage'
    if {str(p.relative_to(original)): sha(p) for p in original.rglob('*') if p.is_file()} != manifest['models']['ImageEncoder']['files']:
        raise ValueError('Source encoder package bytes changed')
    frame = args.run / 'coreml/frame-00.png'
    with Image.open(frame) as image:
        pixels = np.asarray(image.convert('RGB')).copy()
    source_report = json.loads((args.run / 'coreml/report.json').read_text())
    digest = hashlib.sha256(pixels.tobytes()).hexdigest()
    if pixels.shape != (1024, 1024, 3) or digest != source_report['frames'][0]['imageSHA256']:
        raise ValueError('Source frame identity changed')
    args.output.mkdir(parents=True, exist_ok=False)
    report_path = args.output / 'report.json'
    report = {'scope': 'One exact-input encoder control. CPU_AND_NE permits CPU fallback; success does not prove ANE execution. '
                       'FP32 changes both precision and eligible placement. No temporal/device readiness or performance claim.',
              'readyForDeviceValidation': False, 'completed': False, 'sourceStatus': status,
              'sourceManifestSHA256': sha(manifest_path), 'imageSHA256': digest,
              'versions': {'torch': torch.__version__, 'coremltools': ct.__version__},
              'host': {'machine': platform.machine(), 'macOS': platform.mac_ver()[0]},
              'metalDebugLayerEnvironment': os.environ.get('MTL_DEBUG_LAYER'),
              'policy': {'encoderCosineMinimum': .99, 'float32Atol': .001, 'float32Rtol': .001}, 'predictions': {}}
    v.write_json(report_path, report)
    try:
        model = owned.load_reference(args.upstream)
        module = owned.ImageEncoder(model).eval()
        tensor = torch.from_numpy(pixels.astype(np.float32).transpose(2, 0, 1)[None] / 255.)
        with torch.inference_mode():
            reference = dict(zip(owned.IMAGE_OUTPUTS, (x.numpy() for x in module(tensor)), strict=True))
        report['torchReference'] = summarize(reference)
        if any(row['nonfinite'] for row in report['torchReference'].values()):
            raise ValueError('PyTorch reference is non-finite')
        # Use the already-established encoder precision control; no tracker export.
        report['stage'] = 'Exporting diagnostic FP32 encoder'
        v.write_json(report_path, report)
        cpu_model, details = export_encoder_control(module, args.output)
        del cpu_model
        report['fp32Encoder'] = details
        candidate = args.output / 'DiagnosticImageEncoderFP32.mlpackage'
        for units in ('CPU_ONLY', 'CPU_AND_NE'):
            for name, package in (('existing', original), ('fp32', candidate)):
                label = f'{name}-{units}'
                report['stage'] = label
                v.write_json(report_path, report)
                print(f'Predicting {label}...', flush=True)
                report['predictions'][label] = compare_worker(package, frame, args.output, label, units, reference)
                v.write_json(report_path, report)
        report['completed'] = True
        report.pop('stage', None)
    except Exception as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        v.write_json(report_path, report)
    print(f'Report: {report_path}', flush=True)


if __name__ == '__main__':
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if len(sys.argv) > 1 and sys.argv[1] == '__predict':
        worker()
    else:
        main()
