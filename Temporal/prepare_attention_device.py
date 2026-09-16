#!/usr/bin/env python3
"""Prepare a separate FP16 attention device fixture from validated existing files.

Copies the original device reference tensors and inputs unchanged; substitutes
only the verified diagnostic Propagator package. No conversion or inference.
The original normal export and baseline device fixture are never modified.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

import owned
import validate_export as v
from benchmark_propagator_convs import candidate_evidence
from diagnose_propagator_convs import file_hashes, verify
from inspect_encoder_placement import sha, write_json
from prepare_device import TENSORS
from propagator_attention import CONTRACT

PRECISIONS = {
    'memoryfp16': 'memory-sdpa-fp16-only.diagnostic.v1',
    'memoryfp16k4096': 'memory-sdpa-fp16-keypad4096.diagnostic.v1',
}


def safe_file(root, relative):
    path = root / relative
    if Path(relative).is_absolute() or '..' in Path(relative).parts or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f'Unsafe fixture path: {relative}')
    if not path.is_file():
        raise ValueError(f'Missing fixture file: {relative}')
    return path


def paired_evidence(args, evidence):
    report = json.loads((args.paired / 'report.json').read_text())
    baseline = 'original' if args.variant == 'memoryfp16' else 'memoryfp16'
    labels = {baseline, args.variant}
    frames = report.get('frames', [])
    if (report.get('passed') is not True or report.get('completed') is not True
            or report.get('candidateVariant') != args.variant or report.get('computeUnits') != 'CPU_AND_GPU'
            or report.get('sourceManifestSHA256') != evidence['sourceManifestSHA256']
            or report.get('sourceFrameReportSHA256') != evidence['sourceFrameReportSHA256']
            or report.get('candidateReportSHA256') != sha(args.candidate / 'report.json')
            or report.get('candidateFiles') != evidence['variants'][args.variant]['files']
            or len(frames) != 20):
        raise ValueError('Expected a complete paired GPU report for these exact candidate bytes')
    if baseline != 'original' and (
            report.get('baselineVariant') != baseline
            or report.get('baselineFiles') != evidence['variants'][baseline]['files']):
        raise ValueError('Paired report does not match the rebuilt FP16 baseline')
    pairs = []
    for index, frame in enumerate(frames):
        current = frame.get('pairs', [])
        if frame.get('index') != index or len(current) != (0 if index == 0 else 4):
            raise ValueError('Paired report frame or pair count changed')
        if index == 1 and (set(frame.get('warmupChecks', {})) != labels
                           or not all(c.get('passed') is True for c in frame['warmupChecks'].values())):
            raise ValueError('Paired warm-up evidence missing')
        for pair in current:
            if (set(pair.get('checks', {})) != labels
                    or not all(c.get('passed') is True for c in pair['checks'].values())):
                raise ValueError('Paired numerical checks did not pass')
        pairs.extend(current)
    if len(pairs) != 76:
        raise ValueError('Incomplete paired GPU evidence')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--paired', type=Path, required=True)
    parser.add_argument('--variant', choices=tuple(PRECISIONS), default='memoryfp16')
    parser.add_argument('--output', type=Path, required=True, help='New directory only')
    args = parser.parse_args()
    args.inspection = None
    status, source = verify(args)
    evidence = candidate_evidence(args)
    if evidence.get('contract') != CONTRACT or evidence.get('candidateVariant') != args.variant:
        raise ValueError('Expected temporal validation for the selected attention variant')
    paired_evidence(args, evidence)
    baseline_path = args.baseline / 'fixture.json'
    baseline = json.loads(baseline_path.read_text())
    preparation = json.loads((args.baseline / 'preparation-report.json').read_text())
    if (preparation.get('complete') is not True or len(preparation.get('frames', [])) != 20
            or baseline.get('schema') != 'bytewave.temporal-device-fixture.v1'
            or baseline.get('contract') != owned.CONTRACT
            or baseline.get('precisionPolicy') != v.COREML_PRECISION_POLICY
            or baseline.get('graphRevision') != owned.GRAPH_REVISION
            or baseline.get('upstream') != owned.UPSTREAM_REVISION
            or baseline.get('checkpointSHA256') != owned.CHECKPOINT_SHA256
            or baseline.get('diagnosticPropagator') is not None
            or baseline.get('referenceComputeUnits') != 'CPU_ONLY'
            or baseline.get('sourceModelsManifestSHA256') != evidence['sourceManifestSHA256']
            or baseline.get('sourceReportSHA256') != evidence['sourceFrameReportSHA256']
            or baseline.get('pointNormalizedTopLeft') != status['pointNormalizedTopLeft']
            or len(baseline.get('frames', [])) != 20):
        raise ValueError('Baseline fixture does not match the validated original run')
    manifest = json.loads((args.run / 'models/manifest.json').read_text())
    expected_files = {f'models/BWTemporal{component}.mlpackage/{name}': digest
                      for component, entry in manifest['models'].items() for name, digest in entry['files'].items()}
    if baseline.get('modelFiles') != expected_files:
        raise ValueError('Baseline model file set differs from original manifest')
    copied_files = {}
    for relative, digest in expected_files.items():
        if sha(safe_file(args.baseline, relative)) != digest:
            raise ValueError(f'Baseline model bytes changed: {relative}')
    for index, frame in enumerate(baseline['frames']):
        original = source['frames'][index]
        keys = ('index', 'imageSHA256', 'ptsNumerator', 'ptsDenominator')
        if any(frame[k] != original[k] for k in keys) or frame['index'] != index:
            raise ValueError('Baseline frame identity changed')
        raw_path = safe_file(args.baseline, frame['inputPath'])
        raw = raw_path.read_bytes()
        if len(raw) != 1024 * 1024 * 4 or hashlib.sha256(raw).hexdigest() != frame['inputSHA256']:
            raise ValueError('Baseline BGRA input changed')
        rgb = np.frombuffer(raw, dtype=np.uint8).reshape(1024, 1024, 4)[..., :3][..., ::-1]
        if hashlib.sha256(rgb.tobytes()).hexdigest() != original['imageSHA256']:
            raise ValueError('Baseline BGRA pixels differ from validated source image')
        copied_files[frame['inputPath']] = frame['inputSHA256']
        preview = safe_file(args.baseline, frame['previewPath'])
        copied_files[frame['previewPath']] = sha(preview)
        if set(frame['tensors']) != set(TENSORS):
            raise ValueError('Baseline reference tensor set changed')
        for name, tensor in frame['tensors'].items():
            path = safe_file(args.baseline, tensor['path'])
            values = path.read_bytes()
            if (tensor['shape'] != list(v.SHAPES[name]) or hashlib.sha256(values).hexdigest() != tensor['sha256']
                    or len(values) != int(np.prod(v.SHAPES[name])) * 4
                    or not np.isfinite(np.frombuffer(values, dtype='<f4')).all()):
                raise ValueError(f'Invalid baseline reference tensor: {name}')
            copied_files[tensor['path']] = tensor['sha256']
    args.output.mkdir(parents=True, exist_ok=False)
    path = args.output / 'preparation-report.json'
    report = {'complete': False, 'deviceValidated': False, 'scope': __doc__,
              'variant': args.variant, 'frames': 20, 'baselineFixtureSHA256': sha(baseline_path),
              'candidateReportSHA256': sha(args.candidate / 'report.json'),
              'pairedReportSHA256': sha(args.paired / 'report.json')}
    write_json(path, report)
    try:
        for relative, digest in copied_files.items():
            destination = args.output / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(safe_file(args.baseline, relative), destination)
            if sha(destination) != digest:
                raise ValueError(f'Fixture copy failed: {relative}')
        fixture = copy.deepcopy(baseline)
        fixture['modelFiles'] = {}
        for component in v.INPUTS:
            name = f'BWTemporal{component}.mlpackage'
            package = (args.candidate / f'{args.variant}.mlpackage' if component == 'Propagator'
                       else args.baseline / 'models' / name)
            expected = (evidence['variants'][args.variant]['files'] if component == 'Propagator'
                        else manifest['models'][component]['files'])
            destination = args.output / 'models' / name
            shutil.copytree(package, destination)
            if file_hashes(destination) != expected:
                raise ValueError(f'Copied {component} package failed verification')
            fixture['modelFiles'].update({f'models/{name}/{key}': digest for key, digest in expected.items()})
        fixture['precisionPolicy'] = PRECISIONS[args.variant]
        fixture['diagnosticPropagator'] = {
            'variant': args.variant, 'contract': CONTRACT, 'precisionPolicy': PRECISIONS[args.variant],
            'baselineFixtureSHA256': report['baselineFixtureSHA256'],
            'candidateReportSHA256': report['candidateReportSHA256'],
            'pairedReportSHA256': report['pairedReportSHA256']}
        report.update(complete=True, originalReferencesUnchanged=True,
                      verifiedModelFileCount=len(fixture['modelFiles']))
        write_json(path, report)
        write_json(args.output / 'fixture.json', fixture)  # Publish only after all copies verify.
    except Exception as error:
        report.update(complete=False, error=f'{type(error).__name__}: {error}')
        write_json(path, report)
        raise
    print(f'Ready for device comparison: {path}')


if __name__ == '__main__':
    main()
