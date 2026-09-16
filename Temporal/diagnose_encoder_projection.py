#!/usr/bin/env python3
"""Validate composed FP32 encoder projections, then compare paired Mac GPU calls.

Unchanged reconversion CPU control, candidate CPU/GPU 20-frame PyTorch parity,
then identical-image original/candidate GPU timing. All tracker components stay
on original CPU packages for isolation. No normal export or fixture promotion.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

import numpy as np
from PIL import Image
import torch

import owned
import validate_export as v
from diagnose_initializer import summarize
from diagnose_propagator_convs import file_hashes, verify
from encoder_projection import CONTRACT, make_variant
from inspect_encoder_placement import sha, write_json
from validate_encoder_candidate import CandidateBackend, worker


def checks(actual, reference, *, close=False):
    if set(actual) != set(owned.IMAGE_OUTPUTS) or set(reference) != set(owned.IMAGE_OUTPUTS):
        raise ValueError('Encoder output names changed')
    for values in (actual, reference):
        for name, array in values.items():
            if np.shape(array) != v.SHAPES[name] or not np.isfinite(array).all():
                raise ValueError(f'Invalid encoder tensor: {name}')
    rows = summarize(actual, reference)
    for name, row in rows.items():
        row['float32Close'] = bool(np.allclose(actual[name], reference[name], atol=.001, rtol=.001))
    passed = all(row['cosineSimilarity'] >= .99 and (not close or row['float32Close'])
                 for row in rows.values())
    return {'passed': passed, 'requiresFloat32Close': close, 'outputs': rows}


def load_candidate(args, units):
    import coremltools as ct
    evidence = json.loads((args.candidate / 'variants.json').read_text())
    package = args.candidate / f'{args.variant}.mlpackage'
    if (evidence.get('contract') != CONTRACT
            or evidence['sourceManifestSHA256'] != sha(args.run / 'models/manifest.json')
            or evidence['sourceFrameReportSHA256'] != sha(args.run / 'coreml/report.json')
            or file_hashes(package) != evidence['variants'][args.variant]['files']):
        raise ValueError('Diagnostic provenance or package bytes changed')
    model = ct.models.MLModel(str(package), compute_units=getattr(ct.ComputeUnit, units))
    if (model.user_defined_metadata.get('bytewave.contract') != CONTRACT
            or model.user_defined_metadata.get('bytewave.diagnostic.variant') != args.variant
            or model.get_spec().description.input[0].type.WhichOneof('Type') != 'imageType'):
        raise ValueError('Diagnostic contract, variant or image interface changed')
    return model


class ProjectionBackend(CandidateBackend):
    def __init__(self, args, report, report_path):
        v.CoreMLBackend.__init__(self, args.run / 'models', graph_revision=owned.GRAPH_REVISION)
        self.original = self.models['ImageEncoder']
        self.models['ImageEncoder'] = load_candidate(args, args.encoder_units)
        self.input_kind = 'imageType'
        self.report, self.report_path = report, report_path
        self.unchanged = args.variant == 'unchanged'
        report['encoderChecks'] = []
        report['candidateContract'] = CONTRACT

    def call(self, component, inputs):
        result = super().call(component, inputs)
        if component == 'ImageEncoder':
            self.report['activeComponent'] = 'OriginalCPUEncoderControl'
            write_json(self.report_path, self.report)
            reference = self.original.predict({'image': inputs['pil_image']})
            row = checks(result, reference, close=self.unchanged)
            self.report['encoderChecks'].append({'index': self.report['activeFrameIndex'], **row})
            write_json(self.report_path, self.report)
            if not row['passed']:
                raise ValueError('Encoder differs from original CPU beyond diagnostic gates')
        return result


def run_accuracy(args, label, variant, units):
    log = args.output / f'{label}.log'
    command = [sys.executable, str(Path(__file__).resolve()), '--run', str(args.run.resolve()),
               '--upstream', str(args.upstream.resolve()), '--candidate', str(args.output.resolve()),
               '--variant', variant, '--encoder-units', units,
               '--output', str((args.output / label).resolve())]
    print(f'Comparing {label}: 20 frames against original PyTorch...', flush=True)
    with log.open('w') as stream:
        try:
            process = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, timeout=1200, check=False)
            row = {'returnCode': process.returncode, 'log': log.name}
        except subprocess.TimeoutExpired:
            row = {'error': 'Accuracy worker timed out', 'log': log.name}
    path = args.output / label / 'report.json'
    result = json.loads(path.read_text()) if path.exists() else {}
    frames = result.get('frames', [])
    encoder_checks = result.get('encoderChecks', [])
    row.update(report=f'{label}/report.json', framesCompared=len(frames),
               passed=bool(row.get('returnCode') == 0 and result.get('passed') is True
                           and len(frames) == v.COUNT and all(f.get('passed') is True for f in frames)
                           and len(encoder_checks) == v.COUNT and all(f['passed'] for f in encoder_checks)))
    if path.exists():
        row['reportSHA256'] = sha(path)
    if frames:
        row.update(minimumMaskIoU=min(f['maskIoUAgainstReference'] for f in frames),
                   minimumPointerCosine=min(f['outputs']['object_pointer']['cosineSimilarity'] for f in frames),
                   minimumMemoryCosine=min(f['outputs']['memory_features']['cosineSimilarity'] for f in frames),
                   finalState=frames[-1]['state'])
    for key in ('error', 'activeFrameIndex', 'activeComponent'):
        if key in result:
            row[key] = result[key]
    print(f'{label}: passed={row["passed"]}; report: {path}', flush=True)
    return row


def paired(args, source):
    import coremltools as ct
    path = args.candidate / 'paired-summary.json'
    report = {'scope': 'Paired Mac encoder calls only; no iPhone or sustained playback claim.',
              'completed': False, 'passed': False, 'readyForDeviceValidation': False,
              'computeUnits': 'CPU_AND_GPU', 'candidateVariant': 'projection',
              'sourceManifestSHA256': sha(args.run / 'models/manifest.json'),
              'sourceFrameReportSHA256': sha(args.run / 'coreml/report.json'),
              'variantsSHA256': sha(args.candidate / 'variants.json'),
              'coremltoolsVersion': ct.__version__, 'macOS': platform.mac_ver()[0],
              'machine': platform.machine(), 'pairsPerFrame': 4, 'warmupCallsPerModelPerFrame': 1,
              'timingScope': 'perf_counter around predict only, including image bridge; excludes copies/checks. '
                             'Co-resident models, serialized calls, balanced order. Each frame has checked '
                             'warm-up calls excluded from timings. Thermals/hardware utilization unmeasured.',
              'policy': {'cosineMinimum': .99, 'allOutputsFinite': True}, 'frames': []}
    try:
        write_json(path, report)
        original_path = str(args.run / 'models/BWTemporalImageEncoder.mlpackage')
        cpu = ct.models.MLModel(original_path, compute_units=ct.ComputeUnit.CPU_ONLY)
        models = {'original': ct.models.MLModel(original_path, compute_units=ct.ComputeUnit.CPU_AND_GPU),
                  'projection': load_candidate(args, 'CPU_AND_GPU')}
        times = {label: [] for label in models}
        strata = {label: {model: [] for model in models} for label in models}
        ratios = []
        for index, saved in enumerate(source['frames']):
            print(f'Paired encoder GPU: frame {index + 1}/{v.COUNT}', flush=True)
            with Image.open(args.run / f'coreml/frame-{index:02d}.png') as image:
                image = image.convert('RGB')
                pixels = np.asarray(image)
            if (saved['index'] != index or pixels.shape != (1024, 1024, 3)
                    or hashlib.sha256(pixels.tobytes()).hexdigest() != saved['imageSHA256']):
                raise ValueError('Saved frame changed')
            row = {'index': index, 'imageSHA256': saved['imageSHA256'], 'warmups': {}, 'pairs': []}
            report['frames'].append(row)
            report['activeFrameIndex'] = index
            report['activeComponent'] = 'CPU reference'
            write_json(path, report)
            reference = cpu.predict({'image': image})
            for label, model in models.items():
                report['activeComponent'] = f'{label} warm-up'
                write_json(path, report)
                row['warmups'][label] = checks(model.predict({'image': image}), reference)
                if not row['warmups'][label]['passed']:
                    raise ValueError(f'{label} warm-up encoder parity failed')
            for pair in range(4):
                order = ['original', 'projection'] if pair % 2 == 0 else ['projection', 'original']
                observation = {'firstModel': order[0], 'milliseconds': {}, 'checks': {}}
                row['pairs'].append(observation)
                for label in order:
                    report['activeComponent'] = f'{label} pair {pair}'
                    write_json(path, report)
                    start = time.perf_counter()
                    result = models[label].predict({'image': image})
                    elapsed = 1000 * (time.perf_counter() - start)
                    metrics = checks(result, reference)
                    observation['checks'][label] = metrics
                    observation['milliseconds'][label] = elapsed
                    if not metrics['passed']:
                        raise ValueError(f'{label} measured encoder parity failed')
                    times[label].append(elapsed)
                    strata[order[0]][label].append(elapsed)
                ratios.append(observation['milliseconds']['original'] / observation['milliseconds']['projection'])
            write_json(path, report)
        medians = {label: statistics.median(values) for label, values in times.items()}
        report['summary'] = {'measuredPairs': len(ratios), 'medianMilliseconds': medians,
                             'ratioOfMediansOriginalOverCandidate': medians['original'] / medians['projection'],
                             'medianPairedRatioOriginalOverCandidate': statistics.median(ratios),
                             'medianLatencyReductionPercent': 100 * (1 - medians['projection'] / medians['original']),
                             'allMeasuredOutputsPassed': True,
                             'medianMillisecondsByFirstModel': {
                                 first: {label: statistics.median(values) for label, values in group.items()}
                                 for first, group in strata.items()}}
        report.update(completed=True, passed=True)
        report.pop('activeFrameIndex', None)
        report.pop('activeComponent', None)
    except Exception as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        write_json(path, report)
    return {'report': path.name, 'reportSHA256': sha(path), 'passed': True, **report['summary']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--variant', choices=('unchanged', 'projection'), help=argparse.SUPPRESS)
    parser.add_argument('--encoder-units', choices=('CPU_ONLY', 'CPU_AND_GPU'), help=argparse.SUPPRESS)
    parser.add_argument('--paired-worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if platform.system() != 'Darwin':
        parser.error('Run on Mac')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    args.inspection = None
    status, source = verify(args)
    if args.encoder_units or args.paired_worker:
        if args.candidate is None or args.variant is None:
            parser.error('Worker requires candidate and variant')
        if args.paired_worker:
            if args.variant != 'projection':
                parser.error('Paired worker requires projection')
            evidence = json.loads((args.candidate / 'report.json').read_text())
            if (evidence.get('sourceManifestSHA256') != sha(args.run / 'models/manifest.json')
                    or evidence.get('sourceFrameReportSHA256') != sha(args.run / 'coreml/report.json')
                    or evidence.get('variantsSHA256') != sha(args.candidate / 'variants.json')
                    or set(evidence.get('runs', {})) != {'unchanged-cpu', 'projection-cpu', 'projection-gpu'}
                    or not all(row.get('passed') is True and row['reportSHA256'] == sha(args.candidate / row['report'])
                               for row in evidence['runs'].values())):
                raise ValueError('Paired worker requires all verified temporal comparisons')
            return paired(args, source)
        return worker(args, status, source, backend_class=ProjectionBackend, candidate_evidence='variants.json')
    args.output.mkdir(parents=True, exist_ok=False)
    path = args.output / 'report.json'
    report = {'scope': __doc__, 'contract': CONTRACT, 'completed': False, 'passed': False,
              'readyForDeviceValidation': False,
              'sourceManifestSHA256': sha(args.run / 'models/manifest.json'),
              'sourceFrameReportSHA256': sha(args.run / 'coreml/report.json'), 'variants': {}, 'runs': {}}
    try:
        for variant in ('unchanged', 'projection'):
            report['stage'] = f'Preparing {variant}'
            print(report['stage'], flush=True)
            write_json(path, report)
            package = args.output / f'{variant}.mlpackage'
            row = make_variant(args.run / 'models/BWTemporalImageEncoder.mlpackage', package, variant)
            report['variants'][variant] = {**row, 'files': file_hashes(package)}
        write_json(args.output / 'variants.json', {key: report[key] for key in
                   ('contract', 'sourceManifestSHA256', 'sourceFrameReportSHA256', 'variants')})
        report['variantsSHA256'] = sha(args.output / 'variants.json')
        for label, variant, units in (('unchanged-cpu', 'unchanged', 'CPU_ONLY'),
                                      ('projection-cpu', 'projection', 'CPU_ONLY'),
                                      ('projection-gpu', 'projection', 'CPU_AND_GPU')):
            report['stage'] = label
            write_json(path, report)
            row = report['runs'][label] = run_accuracy(args, label, variant, units)
            write_json(path, report)
            if not row['passed']:
                raise ValueError(f'{label} failed; inspect {label}/report.json and {label}.log')
        report['stage'] = 'Paired GPU timing'
        write_json(path, report)
        command = [sys.executable, str(Path(__file__).resolve()), '--run', str(args.run.resolve()),
                   '--upstream', str(args.upstream.resolve()), '--candidate', str(args.output.resolve()),
                   '--output', str(args.output.resolve()), '--variant', 'projection', '--paired-worker']
        # Separate process releases PyTorch/Core ML allocations from accuracy workers.
        subprocess.run(command, check=True, timeout=1200)
        pair_path = args.output / 'paired-summary.json'
        pair = json.loads(pair_path.read_text())
        if pair.get('passed') is not True or pair.get('completed') is not True:
            raise ValueError('Paired encoder comparison failed')
        report['paired'] = {'report': pair_path.name, 'reportSHA256': sha(pair_path), **pair['summary']}
        report.update(completed=True, passed=True)
        report.pop('stage', None)
    except Exception as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        write_json(path, report)
        print(f'Report: {path}', flush=True)


if __name__ == '__main__':
    main()
