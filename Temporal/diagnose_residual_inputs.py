#!/usr/bin/env python3
"""Compare one shared residual input with separate FP32/FP16 branch inputs.

Uses identical saved CPU tensors and unchanged extracted convolutions/additions.
Cutting the graph changes placement and lifetimes; this is not a full encoder fix.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np

from diagnose_encoder_prefix import file_hashes
from diagnose_initializer import summarize
from inspect_encoder_placement import inspect, sha, write_json

OUTPUTS = ('input_39', 'input_39_to_fp16')


def worker():
    import coremltools as ct
    package, tensors, output, units = sys.argv[2:]
    model = ct.models.MLModel(package, compute_units=getattr(ct.ComputeUnit, units))
    with np.load(tensors, allow_pickle=False) as data:
        inputs = {feature.name: np.array(data[feature.name], copy=True, order='C')
                  for feature in model.get_spec().description.input}
    print('Beginning residual block prediction', flush=True)
    result = model.predict(inputs)
    owned = {name: np.array(value, copy=True) for name, value in result.items()}
    np.savez(output, **owned)
    print('Prediction returned', flush=True)


def predict(package, tensors, root, label, units):
    log = root / f'{label}.log'
    output = root / f'{label}.npz'
    with log.open('w') as stream:
        try:
            process = subprocess.run([sys.executable, str(Path(__file__).resolve()), '__predict',
                                      str(package.resolve()), str(tensors.resolve()), str(output.resolve()), units],
                                     stdout=stream, stderr=subprocess.STDOUT, timeout=300, check=False)
        except subprocess.TimeoutExpired as error:
            raise RuntimeError(f'{label}: timed out; see {log.name}') from error
    if process.returncode:
        raise RuntimeError(f'{label}: worker exited {process.returncode}; see {log.name}')
    with np.load(output, allow_pickle=False) as data:
        return {name: np.array(data[name], dtype=np.float32, copy=True) for name in data.files}


def load_cpu(prefix, name, source):
    path = prefix / f'{name}-cpu.npz'
    with np.load(path, allow_pickle=False) as data:
        values = {key: np.array(data[key], dtype=np.float32, copy=True) for key in data.files}
    expected = source['prefixes'][name]['cpuAgainstFullGraphCPU']
    if set(values) != set(expected):
        raise ValueError('CPU tensor names do not match prefix report')
    for key, row in summarize(values).items():
        if row['nonfinite'] or any(value != expected[key][field] for field, value in row.items()):
            raise ValueError(f'CPU tensor summary does not match prefix report: {key}')
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--prefix', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if platform.system() != 'Darwin':
        parser.error('Run on Mac')
    import coremltools as ct
    from coremltools.converters.mil.debugging_utils import extract_submodel

    source_path = args.prefix / 'report.json'
    source = json.loads(source_path.read_text())
    manifest_path = args.run / 'models/manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if source.get('completed') is not True or source.get('sourceManifestSHA256') != sha(manifest_path):
        raise ValueError('Completed prefix report must match the original model manifest')
    if file_hashes(args.run / 'models/BWTemporalImageEncoder.mlpackage') != manifest['models']['ImageEncoder']['files']:
        raise ValueError('Original encoder package changed')
    for name in ('input_33', 'input_39'):
        if file_hashes(args.prefix / f'{name}.mlpackage') != source['prefixes'][name]['files']:
            raise ValueError(f'Prefix package changed: {name}')
    inputs = load_cpu(args.prefix, 'input_33', source)
    reference = load_cpu(args.prefix, 'input_39', source)
    half = inputs['input_33'].astype(np.float16)
    if not np.array_equal(half.astype(np.float32), inputs['input_33_to_fp16']):
        raise ValueError('CPU FP16 branch input is not the exact cast of FP32 residual input')

    args.output.mkdir(parents=True, exist_ok=False)
    tensors = args.output / 'inputs.npz'
    np.savez(tensors, input_33=inputs['input_33'], input_33_to_fp16=half)
    report_path = args.output / 'report.json'
    report = {'scope': __doc__, 'completed': False, 'readyForDeviceValidation': False,
              'sourceManifestSHA256': sha(manifest_path), 'sourcePrefixReportSHA256': sha(source_path),
              'sourceCPUTensorSHA256s': {name: sha(args.prefix / f'{name}-cpu.npz') for name in ('input_33', 'input_39')},
              'inputFileSHA256': sha(tensors), 'imageSHA256': source['imageSHA256'],
              'coremltoolsVersion': ct.__version__, 'variants': {}}
    try:
        for name, cut_inputs in (('shared', ['input_33']), ('separate', ['input_33', 'input_33_to_fp16'])):
            row = report['variants'][name] = {'inputs': cut_inputs, 'outputs': list(OUTPUTS)}
            report['stage'] = f'{name}: extraction'
            write_json(report_path, report)
            print(f'Extracting {name} input control...', flush=True)
            model = ct.models.MLModel(str(args.prefix / 'input_39.mlpackage'), compute_units=ct.ComputeUnit.CPU_ONLY)
            previous = sys.getrecursionlimit()
            try:
                sys.setrecursionlimit(max(previous, 20_000))
                probe = extract_submodel(model, outputs=list(OUTPUTS), inputs=cut_inputs)
            finally:
                sys.setrecursionlimit(previous)
            if {feature.name for feature in probe.get_spec().description.input} != set(cut_inputs):
                raise ValueError('Extracted block input interface differs from requested cut')
            probe.user_defined_metadata['bytewave.contract'] = 'bytewave.residual-inputs.diagnostic.v1'
            destination = args.output / f'{name}.mlpackage'
            probe.save(str(destination))
            del probe, model
            row['files'] = file_hashes(destination)
            report['stage'] = f'{name}: plan'
            write_json(report_path, report)
            plan = inspect(destination)
            write_json(args.output / f'{name}-plan.json', plan)
            row['preferredCounts'] = plan['preferredCounts']
            row['operations'] = plan['operations']
            convs = [op for op in plan['operations'] if op['type'].split('.')[-1] == 'conv']
            if len(convs) != 2:
                raise ValueError('Expected exactly the two residual-block convolutions')
            row['bothConvolutionsPreferNE'] = all(op['preferred'] == 'MLNeuralEngineComputeDevice' for op in convs)
            values = {}
            for label, units in (('cpu', 'CPU_ONLY'), ('ne', 'CPU_AND_NE')):
                report['stage'] = f'{name}: {label}'
                write_json(report_path, report)
                print(f'Predicting {name}-{label}...', flush=True)
                values[label] = predict(destination, tensors, args.output, f'{name}-{label}', units)
                if set(values[label]) != set(OUTPUTS):
                    raise ValueError('Block output interface changed')
                if label == 'cpu':
                    row['cpuAgainstPrefixCPU'] = summarize(values[label], reference)
                    if not all(np.isfinite(values[label][key]).all() and
                               np.allclose(values[label][key], reference[key], atol=.001, rtol=.001) for key in OUTPUTS):
                        raise ValueError('Isolated block CPU parity failed; do not interpret NE')
            row['neAgainstSameInputCPU'] = summarize(values['ne'], values['cpu'])
            row['neAllFinite'] = all(np.isfinite(v).all() for v in values['ne'].values())
            row['neCosineAtLeast099'] = all(v.get('cosineSimilarity', -1) >= .99 for v in row['neAgainstSameInputCPU'].values())
            write_json(report_path, report)
        report['completed'] = True
        report.pop('stage', None)
    except Exception as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        write_json(report_path, report)
    print(f'Report: {report_path}', flush=True)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '__predict':
        worker()
    else:
        main()
