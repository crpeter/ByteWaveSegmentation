#!/usr/bin/env python3
"""One-frame fixed-size point embedding control; never a deployment-ready set."""
from __future__ import annotations

import argparse
import hashlib
import json
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


DensePointPromptEncoder = owned.DensePointPromptEncoder


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def worker():
    import coremltools as ct
    package, inputs, output, units = sys.argv[2:]
    model = ct.models.MLModel(package, compute_units=getattr(ct.ComputeUnit, units))
    with np.load(inputs, allow_pickle=False) as data:
        result = model.predict(dict(data))
    np.savez(output, **result)


def predict_isolated(package, inputs, directory, label, units, expected):
    output = directory / f'{label}.npz'
    log = directory / f'{label}.log'
    with log.open('w') as stream:
        try:
            process = subprocess.run([sys.executable, str(Path(__file__).resolve()), '__predict',
                                      str(package.resolve()), str(inputs.resolve()),
                                      str(output.resolve()), units], stdout=stream, stderr=subprocess.STDOUT,
                                     timeout=300, check=False)
        except subprocess.TimeoutExpired:
            return {'error': 'Prediction worker timed out after 300 seconds', 'log': log.name}
    if process.returncode != 0:
        return {'error': 'Prediction worker failed or aborted', 'returnCode': process.returncode, 'log': log.name}
    with np.load(output, allow_pickle=False) as data:
        actual = dict(data)
    if set(actual) != set(expected):
        raise ValueError(f'{label}: unexpected output names')
    checks = summarize(actual, expected)
    finite = all(row['nonfinite'] == 0 for row in checks.values())
    presence = bool(np.array_equal(expected['object_score'] > 0, actual['object_score'] > 0)) if finite else False
    passed = finite and presence and all(checks[n]['cosineSimilarity'] >= .99
                                        for n in ('low_res_mask', 'high_res_mask', 'object_pointer'))
    passed = passed and checks['low_res_mask']['maskIoUAgainstSameInputTorch'] >= .95
    return {'outputs': checks, 'comparison': {'passed': bool(passed), 'sameObjectPresence': presence}, 'log': log.name}


def graph_counts(package):
    from coremltools.proto import Model_pb2
    spec = Model_pb2.Model()
    spec.ParseFromString((package / 'Data/com.apple.CoreML/model.mlmodel').read_bytes())
    counts = {}
    def visit(block):
        for op in block.operations:
            counts[op.type] = counts.get(op.type, 0) + 1
            for nested in op.blocks:
                visit(nested)
    for function in spec.mlProgram.functions.values():
        for block in function.block_specializations.values():
            visit(block)
    return counts


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
    manifest = json.loads((args.run / 'models/manifest.json').read_text())
    if (status.get('contract') != owned.CONTRACT or status.get('precisionPolicy') != v.COREML_PRECISION_POLICY
            or not all(status.get(k) is True for k in ('referencePassed', 'coremlPassed', 'readyForDeviceValidation'))
            or manifest.get('contract') != owned.CONTRACT or manifest.get('precisionPolicy') != v.COREML_PRECISION_POLICY):
        raise ValueError('Expected a passed current-policy source run')
    packages = {n: args.run / 'models' / f'BWTemporal{n}.mlpackage' for n in ('ImageEncoder', 'Initializer')}
    for name, package in packages.items():
        if {str(p.relative_to(package)): sha(p) for p in package.rglob('*') if p.is_file()} != manifest['models'][name]['files']:
            raise ValueError(f'Source package changed: {name}')
    args.output.mkdir(parents=True, exist_ok=False)
    report = {'scope': 'One-frame prompt-indexing control; CPU encoder inputs held identical. GPU workers isolated from native aborts.',
              'readyForDeviceValidation': False, 'sourceStatus': status,
              'sourceManifestSHA256': sha(args.run / 'models/manifest.json'),
              'versions': {'torch': torch.__version__, 'coremltools': ct.__version__}, 'predictions': {}}
    report_path = args.output / 'report.json'
    try:
        image = Image.open(args.run / 'coreml/frame-00.png').convert('RGB')
        digest = hashlib.sha256(np.asarray(image).tobytes()).hexdigest()
        source_report = json.loads((args.run / 'coreml/report.json').read_text())
        if digest != source_report['frames'][0]['imageSHA256']:
            raise ValueError('Source frame identity changed')
        report['imageSHA256'] = digest
        encoder = ct.models.MLModel(str(packages['ImageEncoder']), compute_units=ct.ComputeUnit.CPU_ONLY)
        features = encoder.predict({'image': image})
        if any(not np.isfinite(value).all() for value in features.values()):
            raise ValueError('CPU encoder produced non-finite inputs')
        inputs = {n: np.asarray(features[n], dtype=np.float32) for n in v.INPUTS['Initializer'][:3]}
        inputs.update(point_coords=np.array([[[min(float(x)*1024, 1023) for x in status['pointNormalizedTopLeft']]]], dtype=np.float32),
                      point_labels=np.ones((1, 1), dtype=np.int32))
        np.savez(args.output / 'inputs.npz', **inputs)
        model = owned.load_reference(args.upstream)
        source_prompt = model.sam_prompt_encoder
        candidate_prompt = DensePointPromptEncoder(source_prompt).eval()
        examples = tuple(torch.from_numpy(inputs[n]) for n in v.INPUTS['Initializer'])
        with torch.inference_mode():
            # Cover absent/negative/positive/corner labels and pixel boundaries.
            report['promptChecks'] = []
            for label in (-1, 0, 1, 2, 3):
                for xy in ((0., 0.), (511.25, 713.5), (1023., 1023.)):
                    points = (torch.tensor([[xy]]), torch.tensor([[label]], dtype=torch.int32))
                    original = source_prompt(points=points, boxes=None, masks=None)
                    candidate = candidate_prompt(points=points)
                    close = all(torch.allclose(a, b, atol=1e-6, rtol=1e-6) for a, b in zip(original, candidate, strict=True))
                    report['promptChecks'].append({'label': label, 'point': xy, 'passed': close})
                    if not close:
                        raise ValueError('Point embedding differs from original')
            expected = dict(zip(owned.MASK_OUTPUTS, (x.numpy() for x in owned.Initializer(model, dense_points=False)(*examples)), strict=True))
            if any(not np.isfinite(x).all() for x in expected.values()):
                raise ValueError('Original PyTorch initializer is non-finite')
            model.sam_prompt_encoder = candidate_prompt
            candidate_module = owned.Initializer(model, dense_points=False).eval()
            candidate = dict(zip(owned.MASK_OUTPUTS, (x.numpy() for x in candidate_module(*examples)), strict=True))
            report['sameInputTorch'] = summarize(candidate, expected)
            if not all(np.allclose(candidate[n], expected[n], atol=.001, rtol=.001) for n in expected):
                raise ValueError('Initializer differs from original PyTorch')
            trace_examples = tuple(torch.ones_like(x) if n == 'point_labels' else torch.zeros_like(x)
                                   for n, x in zip(v.INPUTS['Initializer'], examples, strict=True))
            traced = torch.jit.trace(candidate_module, trace_examples, strict=True, check_trace=False)
            traced_outputs = dict(zip(owned.MASK_OUTPUTS, (x.numpy() for x in traced(*examples)), strict=True))
            report['sameInputTracedTorch'] = summarize(traced_outputs, expected)
            if not all(np.allclose(traced_outputs[n], expected[n], atol=.001, rtol=.001) for n in expected):
                raise ValueError('Traced initializer differs from original PyTorch')
            del traced
            v.write_json(report_path, report)
            v.export_models({'Initializer': candidate_module}, args.output / 'models', diagnostic_name='dense-point-selection.v1')
        candidate_package = args.output / 'models/BWTemporalInitializer.mlpackage'
        report['graphCounts'] = {'original': graph_counts(packages['Initializer']), 'candidate': graph_counts(candidate_package)}
        if report['graphCounts']['candidate'].get('non_zero', 0):
            raise ValueError('Candidate still contains dynamic non_zero operations')
        for units in ('CPU_ONLY', 'CPU_AND_GPU'):
            for name, package in (('original', packages['Initializer']), ('candidate', candidate_package)):
                label = f'{name}-{units}'
                report['activePrediction'] = label
                v.write_json(report_path, report)
                print(f'Predicting {label}...', flush=True)
                report['predictions'][label] = predict_isolated(package, args.output / 'inputs.npz', args.output, label, units, expected)
                v.write_json(report_path, report)
        report.pop('activePrediction', None)
        report['completed'] = True
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
