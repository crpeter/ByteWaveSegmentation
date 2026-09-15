#!/usr/bin/env python3
"""End the encoder at suspect residual outputs, removing later consumers.

Diagnostic only: graph truncation can change partitioning and buffer lifetimes.
A passing prefix does not establish correctness of the complete NE encoder.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys

import numpy as np
from PIL import Image

from diagnose_encoder_taps import predict
from diagnose_initializer import summarize
from inspect_encoder_placement import inspect, sha, write_json


def file_hashes(package):
    return {str(p.relative_to(package)): sha(p) for p in package.rglob('*') if p.is_file()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--taps', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if platform.system() != 'Darwin':
        parser.error('Run on Mac')
    import coremltools as ct
    from coremltools.converters.mil.debugging_utils import extract_submodel

    manifest_path = args.run / 'models/manifest.json'
    manifest = json.loads(manifest_path.read_text())
    source_path = args.taps / 'report.json'
    source = json.loads(source_path.read_text())
    package = args.run / 'models/BWTemporalImageEncoder.mlpackage'
    if source.get('completed') is not True or source.get('sourceManifestSHA256') != sha(manifest_path):
        raise ValueError('Completed tap report must match the source model manifest')
    if file_hashes(package) != manifest['models']['ImageEncoder']['files']:
        raise ValueError('Original encoder bytes changed')
    if file_hashes(args.taps / 'DiagnosticEncoderTaps.mlpackage') != source['probeFiles']:
        raise ValueError('Source tap package bytes changed')
    frame = args.run / 'coreml/frame-00.png'
    with Image.open(frame) as image:
        pixels = np.asarray(image.convert('RGB'))
    if pixels.shape != (1024, 1024, 3) or hashlib.sha256(pixels.tobytes()).hexdigest() != source['imageSHA256']:
        raise ValueError('Source frame changed')

    # The targets come from the supplied input-path trace, not new source-model
    # precision choices. Fail explicitly if used with a different graph.
    targets = ('input_33', 'input_39')
    rows = source['inputPath']['operations']
    operations = {name: row for row in rows for name in row['outputs']}
    for target in targets:
        if operations.get(target, {}).get('type') != 'add':
            raise ValueError(f'Expected traced residual add: {target}')
    baseline_path = args.taps / 'probe-cpu.npz'
    with np.load(baseline_path, allow_pickle=False) as data:
        baseline = {name: np.array(data[name], dtype=np.float32, copy=True)
                    for target in targets for name in (target, target + '_to_fp16')}
    for target in targets:
        if summarize({target: baseline[target]})[target] != source['cpuInputPathOutputs'][target]:
            raise ValueError(f'Saved CPU baseline no longer matches report: {target}')
        if not np.isfinite(baseline[target]).all():
            raise ValueError('CPU baseline must be finite')
        cast = baseline[target].astype(np.float16).astype(np.float32)
        if not np.array_equal(cast, baseline[target + '_to_fp16']):
            raise ValueError(f'CPU baseline cast is inconsistent: {target}')

    args.output.mkdir(parents=True, exist_ok=False)
    report_path = args.output / 'report.json'
    report = {'scope': __doc__, 'completed': False, 'readyForDeviceValidation': False,
              'sourceManifestSHA256': sha(manifest_path), 'sourceTapReportSHA256': sha(source_path),
              'sourceCPUTensorsSHA256': sha(baseline_path), 'imageSHA256': source['imageSHA256'],
              'coremltoolsVersion': ct.__version__, 'prefixes': {}}
    try:
        for target in targets:
            outputs = [target, target + '_to_fp16']
            row = report['prefixes'][target] = {'outputs': outputs}
            report['stage'] = f'{target}: extraction'
            write_json(report_path, report)
            # Start from the untouched original package each time. Expose only
            # the terminal FP32 sum and FP16 cast, not the full diagnostic taps.
            model = ct.models.MLModel(str(package), compute_units=ct.ComputeUnit.CPU_ONLY)
            previous = sys.getrecursionlimit()
            row['extractionRecursionLimit'] = max(previous, 20_000)
            print(f'Extracting prefix ending at {target}...', flush=True)
            try:
                sys.setrecursionlimit(row['extractionRecursionLimit'])
                probe = extract_submodel(model, outputs=outputs)
            finally:
                sys.setrecursionlimit(previous)
            probe.user_defined_metadata['bytewave.contract'] = 'bytewave.encoder-prefix.diagnostic.v1'
            destination = args.output / f'{target}.mlpackage'
            probe.save(str(destination))
            del probe, model
            row['files'] = file_hashes(destination)
            report['stage'] = f'{target}: plan'
            write_json(report_path, report)
            plan = inspect(destination)
            write_json(args.output / f'{target}-plan.json', plan)
            row['preferredCounts'] = plan['preferredCounts']
            row['terminalOperations'] = [op for op in plan['operations'] if set(op['outputs']) & set(outputs)]
            values = {}
            for label, units in (('cpu', 'CPU_ONLY'), ('ne', 'CPU_AND_NE')):
                report['stage'] = f'{target}: {label}'
                write_json(report_path, report)
                print(f'Predicting {target}-{label}...', flush=True)
                values[label] = predict(destination, frame, args.output, f'{target}-{label}', units)
                if set(values[label]) != set(outputs):
                    raise ValueError('Prefix output names differ from requested targets')
                if label == 'cpu':
                    row['cpuAgainstFullGraphCPU'] = summarize(values[label], baseline)
                    if not all(np.isfinite(values[label][n]).all() and
                               np.allclose(values[label][n], baseline[n], atol=.001, rtol=.001) for n in outputs):
                        raise ValueError('Prefix CPU differs from full-graph CPU; do not interpret NE')
            row['neAgainstPrefixCPU'] = summarize(values['ne'], values['cpu'])
            row['neAllFinite'] = all(np.isfinite(v).all() for v in values['ne'].values())
            row['neCosineAtLeast099'] = all(v.get('cosineSimilarity', -1) >= .99 for v in row['neAgainstPrefixCPU'].values())
            # A cast of a finite, in-range FP32 tensor should agree with its
            # separately exposed FP16 value. Record this independently of parity.
            for label in ('cpu', 'ne'):
                value = values[label][target]
                in_range = bool(np.isfinite(value).all() and np.max(np.abs(value)) <= np.finfo(np.float16).max)
                row[label + 'CastConsistency'] = {'inputFiniteAndInRange': in_range}
                if in_range:
                    expected = value.astype(np.float16).astype(np.float32)
                    actual = values[label][target + '_to_fp16']
                    row[label + 'CastConsistency']['exact'] = bool(np.array_equal(expected, actual))
                    if np.isfinite(actual).all():
                        row[label + 'CastConsistency']['maximumAbsoluteError'] = float(np.max(np.abs(expected - actual)))
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
    main()
