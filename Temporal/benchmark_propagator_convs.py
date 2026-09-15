#!/usr/bin/env python3
"""Paired Mac call timings for original and validated conv2 propagators.

Both models receive identical inputs from the original CPU temporal state.
Four pairs per propagated frame alternate order equally; warm-up is excluded.
Every measured output must pass existing Core ML gates against same-input CPU.
This compares repeated propagator calls, not sustained playback or iPhone FPS.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import platform
import statistics
import time

import numpy as np
from PIL import Image
import torch

import owned
import validate_export as v
from diagnose_propagator_convs import file_hashes, verify
from inspect_encoder_placement import sha, write_json
from propagator_convs import CONTRACT
from propagator_attention import CONTRACT as ATTENTION_CONTRACT
from state import TemporalState


def checked(output):
    if set(output) != set(v.NAMES['Propagator']):
        raise ValueError('Unexpected propagator output names')
    result = {n: np.array(output[n], dtype=np.float32, copy=True) for n in v.NAMES['Propagator']}
    for name, array in result.items():
        if array.shape != v.SHAPES[name] or not np.isfinite(array).all():
            raise ValueError(f'Invalid output: {name}')
    return result


def candidate_evidence(args):
    path = args.candidate / 'report.json'
    report = json.loads(path.read_text())
    if (report.get('completed') is not True or report.get('passed') is not True
            or report.get('sourceManifestSHA256') != sha(args.run / 'models/manifest.json')
            or report.get('sourceFrameReportSHA256') != sha(args.run / 'coreml/report.json')
            or (args.inspection is not None and
                report.get('sourceInspectionSHA256') != sha(args.inspection / 'report.json'))):
        raise ValueError('Expected a passed candidate comparison matching these source inputs')
    required = ['unchanged-cpu', f'{args.variant}-cpu', f'{args.variant}-gpu']
    if args.variant == 'conv2':
        required.append('conv2-ne')
    for label in required:
        row = report['runs'][label]
        child = json.loads((args.candidate / label / 'report.json').read_text())
        if (row.get('passed') is not True or row.get('framesCompared') != v.COUNT
                or child.get('passed') is not True or child.get('completed') is not True
                or len(child.get('frames', [])) != v.COUNT
                or any(f.get('passed') is not True for f in child['frames'])
                or child.get('sourceManifestSHA256') != report['sourceManifestSHA256']
                or child.get('variantsSHA256') != sha(args.candidate / 'variants.json')):
            raise ValueError(f'Candidate correctness evidence is incomplete: {label}')
    variants = json.loads((args.candidate / 'variants.json').read_text())
    if variants['sourceManifestSHA256'] != report['sourceManifestSHA256']:
        raise ValueError('Candidate manifest differs')
    for variant in ('unchanged', args.variant):
        actual = file_hashes(args.candidate / f'{variant}.mlpackage')
        if actual != report['variants'][variant]['files'] or actual != variants['variants'][variant]['files']:
            raise ValueError(f'Candidate package bytes changed: {variant}')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--inspection', type=Path, help='Required for conv2')
    parser.add_argument('--variant', choices=('conv2', 'chunk256'), default='conv2')
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--units', choices=('CPU_AND_GPU', 'CPU_AND_NE'), default='CPU_AND_GPU')
    args = parser.parse_args()
    if platform.system() != 'Darwin':
        parser.error('Run on Mac')
    if args.variant == 'conv2' and args.inspection is None:
        parser.error('--inspection is required for conv2')
    if args.variant == 'chunk256' and (args.units != 'CPU_AND_GPU' or args.inspection is not None):
        parser.error('chunk256 requires CPU_AND_GPU and no linear inspection')
    candidate_contract = CONTRACT if args.variant == 'conv2' else ATTENTION_CONTRACT
    scope = __doc__.replace('conv2', args.variant)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    args.output.mkdir(parents=True, exist_ok=False)
    path = args.output / 'report.json'
    report = {'scope': scope, 'candidateVariant': args.variant, 'completed': False, 'passed': False, 'readyForDeviceValidation': False,
              'passedMeaning': 'All accuracy checks completed; does not imply a speed improvement.',
              'computeUnits': args.units, 'macOS': platform.mac_ver()[0], 'machine': platform.machine(),
              'pairsPerFrame': 4, 'warmupCallsPerModel': 1, 'frames': [],
              'timingScope': 'perf_counter around predict only; excludes input assembly, output copies/checks, '
                             'reference prediction and checkpoint writes. Models co-resident, calls serialized. '
                             'Warm-up outputs are checked but their times are excluded. '
                             'Thermals and actual hardware execution are not measured.',
              'reference': 'Original CPU propagator on identical inputs; original CPU owns temporal state.',
              'policy': {'maskIoUMinimum': .95, 'cosineMinimum': .99}}
    try:
        report['stage'] = 'Verify source and candidate'
        write_json(path, report)
        status, source = verify(args)
        evidence = candidate_evidence(args)
        report.update(sourceManifestSHA256=evidence['sourceManifestSHA256'],
                      candidateReportSHA256=sha(args.candidate / 'report.json'),
                      sourceFrameReportSHA256=evidence['sourceFrameReportSHA256'],
                      candidateFiles=evidence['variants'][args.variant]['files'],
                      pointNormalizedTopLeft=status['pointNormalizedTopLeft'])
        import coremltools as ct
        report['coremltoolsVersion'] = ct.__version__
        report['stage'] = 'Load CPU reference and paired models'
        write_json(path, report)
        reference = v.CoreMLBackend(args.run / 'models', graph_revision=owned.GRAPH_REVISION)
        models = {}
        report['loadMilliseconds'] = {}
        for label, package in (
                ('original', args.run / 'models/BWTemporalPropagator.mlpackage'),
                (args.variant, args.candidate / f'{args.variant}.mlpackage')):
            start = time.perf_counter()
            models[label] = ct.models.MLModel(str(package), compute_units=getattr(ct.ComputeUnit, args.units))
            report['loadMilliseconds'][label] = 1000 * (time.perf_counter() - start)
            expected_contract = owned.CONTRACT if label == 'original' else candidate_contract
            if models[label].user_defined_metadata.get('bytewave.contract') != expected_contract:
                raise ValueError(f'Unexpected {label} model contract')
            if (label != 'original' and
                    models[label].user_defined_metadata.get('bytewave.diagnostic.variant') != args.variant):
                raise ValueError('Unexpected candidate variant')
        state = TemporalState()
        point = [min(max(float(n) * 1024, 0), 1023) for n in status['pointNormalizedTopLeft']]

        def predict(label, values):
            report['activeModel'] = label
            write_json(path, report)
            start = time.perf_counter()
            output = models[label].predict(values)
            elapsed = 1000 * (time.perf_counter() - start)
            return checked(output), elapsed

        for index, saved in enumerate(source['frames']):
            report['activeFrameIndex'] = index
            report['stage'] = 'CPU reference'
            write_json(path, report)
            print(f'Paired {args.units}: frame {index + 1}/{v.COUNT}', flush=True)
            with Image.open(args.run / f'coreml/frame-{index:02d}.png') as image:
                image = image.convert('RGB')
                pixels = np.asarray(image).copy()
            digest = hashlib.sha256(pixels.tobytes()).hexdigest()
            if saved['index'] != index or pixels.shape != (1024, 1024, 3) or digest != saved['imageSHA256']:
                raise ValueError('Saved input frame changed')
            timestamp = Fraction(saved['ptsNumerator'], saved['ptsDenominator'])
            state.validate_time(timestamp)
            token = state.token
            features = reference.call('ImageEncoder', {'pil_image': image})
            row = {'index': index, 'imageSHA256': digest,
                   'ptsNumerator': timestamp.numerator, 'ptsDenominator': timestamp.denominator, 'pairs': []}
            report['frames'].append(row)
            if index == 0:
                expected = reference.call('Initializer', {**features,
                    'point_coords': np.array([[point]], dtype=np.float32),
                    'point_labels': np.ones((1, 1), dtype=np.int32)})
                expected.update(reference.call('InitialMemoryEncoder', {**features, **expected}))
                if not (expected['low_res_mask'] > 0).any():
                    raise ValueError('CPU initialization is empty')
            else:
                all_inputs = {**features, **state.pack(timestamp)}
                values = {n: np.ascontiguousarray(all_inputs[n], dtype=np.float32) for n in v.INPUTS['Propagator']}
                expected = reference.call('Propagator', values)
                row['inputTensorSHA256'] = {n: hashlib.sha256(a.tobytes()).hexdigest() for n, a in values.items()}
                if index == 1:
                    report['stage'] = 'Warm-up (excluded from timings)'
                    row['warmupChecks'] = {}
                    for label in models:
                        actual, _ = predict(label, values)
                        check = row['warmupChecks'][label] = v.compare(expected, actual, coreml=True)
                        if not check['passed']:
                            raise ValueError(f'{label} warm-up parity failed')
                report['stage'] = 'Measured pairs'
                for repetition in range(4):
                    order = ['original', args.variant] if (index + repetition) % 2 == 0 else [args.variant, 'original']
                    report['activeRepetition'] = repetition
                    pair = {'order': order, 'milliseconds': {}, 'checks': {}}
                    row['pairs'].append(pair)
                    outputs = {}
                    for label in order:
                        outputs[label], pair['milliseconds'][label] = predict(label, values)
                    for label in order:
                        check = pair['checks'][label] = v.compare(expected, outputs[label], coreml=True)
                        if not check['passed']:
                            raise ValueError(f'{label} parity failed at frame {index}, pair {repetition}')
                    write_json(path, report)
                if any(hashlib.sha256(a.tobytes()).hexdigest() != row['inputTensorSHA256'][n] for n, a in values.items()):
                    raise ValueError('Input tensors changed during paired predictions')
            state.commit(expected, timestamp, token)
            row['state'] = state.summary()
            write_json(path, report)
        if state.summary() != {'acceptedFrames': 20, 'spatialEntries': 7, 'pointerEntries': 16}:
            raise ValueError('Full bounded temporal state was not exercised')
        pairs = [p for f in report['frames'] for p in f['pairs']]
        medians = {label: statistics.median(p['milliseconds'][label] for p in pairs) for label in models}
        ratio = medians['original'] / medians[args.variant]
        report['summary'] = {
            'measuredPairs': len(pairs), 'medianMilliseconds': medians,
            'ratioOfMediansOriginalOverCandidate': ratio,
            'medianPairedRatioOriginalOverCandidate': statistics.median(
                p['milliseconds']['original'] / p['milliseconds'][args.variant] for p in pairs),
            'medianLatencyReductionPercent': 100 * (1 - medians[args.variant] / medians['original']),
            'allMeasuredOutputsPassed': True,
            'medianMillisecondsByFirstModel': {
                first: {label: statistics.median(p['milliseconds'][label] for p in pairs if p['order'][0] == first)
                        for label in models} for first in models}}
        if args.variant == 'conv2':
            report['summary']['ratioOfMediansOriginalOverConv2'] = report['summary']['ratioOfMediansOriginalOverCandidate']
            report['summary']['medianPairedRatioOriginalOverConv2'] = report['summary']['medianPairedRatioOriginalOverCandidate']
        report.update(passed=True, completed=True)
        for key in ('stage', 'activeModel', 'activeFrameIndex', 'activeRepetition'):
            report.pop(key, None)
    except Exception as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        write_json(path, report)
        write_json(args.output / 'summary.json', {k: value for k, value in report.items() if k != 'frames'})
        print(f'Summary: {args.output / "summary.json"}', flush=True)


if __name__ == '__main__':
    main()
