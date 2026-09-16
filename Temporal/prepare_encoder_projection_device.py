#!/usr/bin/env python3
"""Prepare an isolated fused-encoder + existing memoryfp16 device comparison.

Verify Mac accuracy/paired evidence and the prepared original/FP16 fixtures.
Copy identical original references and the existing FP16 tracker; replace only
ImageEncoder. No conversion, inference, reference regeneration or promotion.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np

import owned
import validate_export as v
from benchmark_propagator_convs import candidate_evidence
from diagnose_propagator_convs import file_hashes, verify
from encoder_projection import CONTRACT
from inspect_encoder_placement import sha, write_json
from prepare_attention_device import PRECISIONS, safe_file, verify_baseline
from propagator_attention import CONTRACT as ATTENTION_CONTRACT

PRECISION = 'encoder-projection-memoryfp16.diagnostic.v1'


def read(path):
    return json.loads(path.read_text())


def projection_evidence(args, source):
    report = read(args.candidate / 'report.json')
    variants_path = args.candidate / 'variants.json'
    variants = read(variants_path)
    if (report.get('contract') != CONTRACT or report.get('completed') is not True
            or report.get('passed') is not True
            or report.get('sourceManifestSHA256') != sha(args.run / 'models/manifest.json')
            or report.get('sourceFrameReportSHA256') != sha(args.run / 'coreml/report.json')
            or report.get('variantsSHA256') != sha(variants_path)
            or variants.get('contract') != CONTRACT
            or variants.get('sourceManifestSHA256') != report['sourceManifestSHA256']
            or variants.get('sourceFrameReportSHA256') != report['sourceFrameReportSHA256']
            or variants.get('variants') != report.get('variants')):
        raise ValueError('Expected complete projection evidence for this validated source')
    for variant in ('unchanged', 'projection'):
        if file_hashes(args.candidate / f'{variant}.mlpackage') != report['variants'][variant]['files']:
            raise ValueError(f'Encoder diagnostic bytes changed: {variant}')
    runs = {'unchanged-cpu': ('unchanged', 'CPU_ONLY'),
            'projection-cpu': ('projection', 'CPU_ONLY'), 'projection-gpu': ('projection', 'CPU_AND_GPU')}
    if set(report.get('runs', {})) != set(runs):
        raise ValueError('Incomplete encoder accuracy modes')
    for label, (variant, units) in runs.items():
        row = report['runs'][label]
        path = args.candidate / label / 'report.json'
        child = read(path)
        frames = child.get('frames', [])
        controls = child.get('encoderChecks', [])
        if (row.get('passed') is not True or row.get('framesCompared') != v.COUNT
                or row.get('returnCode') != 0 or row.get('reportSHA256') != sha(path)
                or child.get('completed') is not True or child.get('passed') is not True
                or child.get('candidateContract') != CONTRACT or child.get('candidateVariant') != variant
                or child.get('encoderComputeUnits') != units or child.get('trackerComputeUnits') != 'CPU_ONLY'
                or child.get('sourceManifestSHA256') != report['sourceManifestSHA256']
                or child.get('candidateReportSHA256') != report['variantsSHA256']
                or len(frames) != v.COUNT or len(controls) != v.COUNT):
            raise ValueError(f'Incomplete encoder temporal evidence: {label}')
        for index, (frame, control, original) in enumerate(zip(frames, controls, source['frames'], strict=True)):
            if (frame.get('passed') is not True or control.get('passed') is not True
                    or control.get('index') != index
                    or any(frame.get(k) != original[k] for k in ('index', 'imageSHA256', 'ptsNumerator', 'ptsDenominator'))
                    or set(control.get('outputs', {})) != set(owned.IMAGE_OUTPUTS)):
                raise ValueError(f'Encoder frame/control identity failed: {label}/{index}')
            for name, metric in control['outputs'].items():
                if (metric.get('shape') != list(v.SHAPES[name]) or metric.get('nonfinite') != 0
                        or not metric.get('cosineSimilarity', -1) >= .99
                        or (variant == 'unchanged' and metric.get('float32Close') is not True)):
                    raise ValueError(f'Encoder feature gate failed: {label}/{index}/{name}')
        if frames[-1]['state'] != {'acceptedFrames': 20, 'spatialEntries': 7, 'pointerEntries': 16}:
            raise ValueError('Full temporal memory banks not exercised')
    paired_path = args.candidate / 'paired-summary.json'
    paired = read(paired_path)
    if (report.get('paired', {}).get('reportSHA256') != sha(paired_path)
            or paired.get('passed') is not True or paired.get('completed') is not True
            or paired.get('computeUnits') != 'CPU_AND_GPU' or paired.get('candidateVariant') != 'projection'
            or paired.get('sourceManifestSHA256') != report['sourceManifestSHA256']
            or paired.get('sourceFrameReportSHA256') != report['sourceFrameReportSHA256']
            or paired.get('variantsSHA256') != report['variantsSHA256']
            or paired.get('summary', {}).get('measuredPairs') != 80
            or len(paired.get('frames', [])) != v.COUNT):
        raise ValueError('Incomplete paired encoder GPU evidence')
    labels = {'original', 'projection'}
    for index, frame in enumerate(paired['frames']):
        if (frame.get('index') != index or frame.get('imageSHA256') != source['frames'][index]['imageSHA256']
                or set(frame.get('warmups', {})) != labels or len(frame.get('pairs', [])) != 4):
            raise ValueError('Paired frame/warm-up evidence changed')
        checked = list(frame['warmups'].values())
        for pair_index, pair in enumerate(frame['pairs']):
            if (pair.get('firstModel') != ('original' if pair_index % 2 == 0 else 'projection')
                    or set(pair.get('checks', {})) != labels or set(pair.get('milliseconds', {})) != labels
                    or not all(np.isfinite(t) and t > 0 for t in pair['milliseconds'].values())):
                raise ValueError('Paired order/timings/checks changed')
            checked.extend(pair['checks'].values())
        for check in checked:
            if check.get('passed') is not True or set(check.get('outputs', {})) != set(owned.IMAGE_OUTPUTS):
                raise ValueError('Paired accuracy check failed')
            for name, metric in check['outputs'].items():
                if (metric.get('nonfinite') != 0 or metric.get('shape') != list(v.SHAPES[name])
                        or not metric.get('cosineSimilarity', -1) >= .99):
                    raise ValueError('Paired encoder output gate failed')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True, help='Original prepared device fixture')
    parser.add_argument('--fp16', type=Path, required=True, help='Existing prepared memoryfp16 fixture')
    parser.add_argument('--attention', type=Path, required=True, help='Validated memoryfp16 Mac diagnostic')
    parser.add_argument('--candidate', type=Path, required=True, help='Passed encoder-projection diagnostic')
    parser.add_argument('--output', type=Path, required=True, help='New directory only')
    args = parser.parse_args()
    args.inspection = None
    status, source = verify(args)
    evidence = projection_evidence(args, source)
    baseline, manifest, reference_files = verify_baseline(args, status, source, evidence)
    attention = candidate_evidence(SimpleNamespace(run=args.run, candidate=args.attention,
                                                  inspection=None, variant='memoryfp16'))
    if attention.get('contract') != ATTENTION_CONTRACT or attention.get('candidateVariant') != 'memoryfp16':
        raise ValueError('Expected validated memoryfp16 attention source')
    fp16_path = args.fp16 / 'fixture.json'
    fp16 = read(fp16_path)
    preparation = read(args.fp16 / 'preparation-report.json')
    diagnostic = fp16.get('diagnosticPropagator', {})
    expected_files = dict(baseline['modelFiles'])
    prefix = 'models/BWTemporalPropagator.mlpackage/'
    expected_files = {k: d for k, d in expected_files.items() if not k.startswith(prefix)}
    expected_files.update({prefix + k: d for k, d in attention['variants']['memoryfp16']['files'].items()})
    inherited_keys = ('schema', 'contract', 'graphRevision', 'upstream', 'checkpointSHA256',
                      'sourceModelsManifestSHA256', 'sourceReportSHA256', 'pointNormalizedTopLeft',
                      'referenceComputeUnits', 'frames')
    if (preparation.get('complete') is not True or preparation.get('originalReferencesUnchanged') is not True
            or any(fp16.get(k) != baseline[k] for k in inherited_keys)
            or fp16.get('precisionPolicy') != PRECISIONS['memoryfp16'] or fp16.get('diagnosticEncoder') is not None
            or fp16.get('modelFiles') != expected_files
            or diagnostic.get('contract') != ATTENTION_CONTRACT or diagnostic.get('variant') != 'memoryfp16'
            or diagnostic.get('precisionPolicy') != PRECISIONS['memoryfp16']
            or diagnostic.get('candidateReportSHA256') != sha(args.attention / 'report.json')
            or diagnostic.get('baselineFixtureSHA256') != sha(args.baseline / 'fixture.json')):
        raise ValueError('Prepared FP16 baseline does not match verified original references/attention bytes')
    for relative, digest in {**expected_files, **reference_files}.items():
        if sha(safe_file(args.fp16, relative)) != digest:
            raise ValueError(f'FP16 baseline bytes changed: {relative}')
    for component in v.INPUTS:
        expected = (attention['variants']['memoryfp16']['files'] if component == 'Propagator'
                    else manifest['models'][component]['files'])
        if file_hashes(args.fp16 / f'models/BWTemporal{component}.mlpackage') != expected:
            raise ValueError(f'FP16 baseline package file set changed: {component}')
    args.output.mkdir(parents=True, exist_ok=False)
    report_path = args.output / 'preparation-report.json'
    report = {'scope': __doc__, 'complete': False, 'deviceValidated': False, 'frames': v.COUNT,
              'variant': 'memoryfp16projection', 'baselineFixtureSHA256': sha(fp16_path),
              'candidateReportSHA256': sha(args.candidate / 'report.json'),
              'pairedReportSHA256': sha(args.candidate / 'paired-summary.json')}
    try:
        write_json(report_path, report)
        for relative, digest in reference_files.items():
            destination = args.output / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(safe_file(args.fp16, relative), destination)
            if sha(destination) != digest:
                raise ValueError(f'Copied reference/input changed: {relative}')
        fixture = copy.deepcopy(fp16)
        fixture['modelFiles'] = {}
        for component in v.INPUTS:
            name = f'BWTemporal{component}.mlpackage'
            package = (args.candidate / 'projection.mlpackage' if component == 'ImageEncoder'
                       else args.fp16 / 'models' / name)
            expected = (evidence['variants']['projection']['files'] if component == 'ImageEncoder'
                        else attention['variants']['memoryfp16']['files'] if component == 'Propagator'
                        else manifest['models'][component]['files'])
            destination = args.output / 'models' / name
            shutil.copytree(package, destination)
            if file_hashes(destination) != expected:
                raise ValueError(f'Copied package changed: {component}')
            fixture['modelFiles'].update({f'models/{name}/{k}': d for k, d in expected.items()})
        fixture['precisionPolicy'] = PRECISION
        fixture['diagnosticEncoder'] = {
            'variant': 'projection', 'contract': CONTRACT, 'precisionPolicy': PRECISION,
            **{k: report[k] for k in ('baselineFixtureSHA256', 'candidateReportSHA256', 'pairedReportSHA256')}}
        report.update(complete=True, originalReferencesUnchanged=True, fp16TrackerUnchanged=True,
                      verifiedModelFileCount=len(fixture['modelFiles']))
        write_json(report_path, report)
        write_json(args.output / 'fixture.json', fixture)
    except Exception as error:
        report.update(complete=False, error=f'{type(error).__name__}: {error}')
        write_json(report_path, report)
        raise
    print(f'Ready for device comparison: {report_path}')


if __name__ == '__main__':
    main()
