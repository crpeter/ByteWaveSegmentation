#!/usr/bin/env python3
"""Compare an isolated memory-attention candidate on the saved fixture.

Unchanged CPU reconversion control, then candidate CPU/GPU against original
PyTorch over all 20 frames; FP16-derived experiments also rebuild/check the FP16
CPU/GPU baseline. Every query still attends to every original key.
Diagnostic only: no normal exporter, installed model or correctness gate changes.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import platform

import torch

from diagnose_propagator_convs import file_hashes, run_worker, verify, worker
from inspect_encoder_placement import sha, write_json
from propagator_attention import CONTRACT, FP16_BASELINE_VARIANTS, VARIANTS, make_variant


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--experiment', choices=VARIANTS[1:], default='chunk256',
                        help='Explicit FP32 chunks, memory FP16, native 1024-query chunks, or excluded cross-key padding to 4096')
    parser.add_argument('--candidate', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--variant', choices=VARIANTS, help=argparse.SUPPRESS)
    parser.add_argument('--units', choices=('CPU_ONLY', 'CPU_AND_GPU'), help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.inspection = None  # This experiment inspects its own four SDPA inputs.
    if platform.system() != 'Darwin':
        parser.error('Run on Mac')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    status, source = verify(args)
    if args.units:
        if args.candidate is None or args.variant is None:
            parser.error('Worker requires candidate and variant')
        return worker(args, status, source, contract=CONTRACT, scope=__doc__)
    args.output.mkdir(parents=True, exist_ok=False)
    path = args.output / 'report.json'
    report = {'scope': __doc__, 'contract': CONTRACT, 'completed': False, 'passed': False,
              'candidateVariant': args.experiment,
              'readyForDeviceValidation': False,
              'sourceManifestSHA256': sha(args.run / 'models/manifest.json'),
              'sourceFrameReportSHA256': sha(args.run / 'coreml/report.json'),
              'variants': {}, 'runs': {}}
    controls = ('unchanged', 'memoryfp16') if args.experiment in FP16_BASELINE_VARIANTS else ('unchanged',)
    if args.experiment in FP16_BASELINE_VARIANTS:
        report['performanceBaselineVariant'] = 'memoryfp16'
    if args.experiment == 'memoryfp16q1024':
        report['experimentScope'] = ('Four native 1024-query FP16 SDPA calls replace each full-query memory SDPA. '
                                    'All keys/values and broadcast masks remain; not the closed explicit FP32 chunk256 rewrite. '
                                    'Accuracy checks precede a separate paired GPU benchmark against memoryfp16.')
    elif args.experiment == 'memoryfp16k4096':
        report['experimentScope'] = ('Append 448 zero key/value rows with negative-infinity additive mask to each '
                                    '3648-key cross-attention, after rotary/validity construction. Original keys, values, '
                                    'query order and mask entries remain; self-attention stays at baseline FP16. '
                                    'Hypothesis only: a 4096-key shape may change GPU dispatch or boundary handling. '
                                    'Shader names do not prove exact operation attribution or a speed benefit. '
                                    'No query chunking, context reduction, normal export or fixture installation.')
    try:
        for variant in (*controls, args.experiment):
            report['stage'] = f'Preparing {variant}'
            write_json(path, report)
            print(report['stage'], flush=True)
            package = args.output / f'{variant}.mlpackage'
            row = make_variant(args.run / 'models/BWTemporalPropagator.mlpackage', package, variant)
            row['files'] = file_hashes(package)
            report['variants'][variant] = row
        write_json(args.output / 'variants.json', {'sourceManifestSHA256': report['sourceManifestSHA256'],
                                                  'variants': report['variants']})
        runs = [('unchanged-cpu', 'unchanged', 'CPU_ONLY')]
        if args.experiment in FP16_BASELINE_VARIANTS:
            runs.extend([('memoryfp16-cpu', 'memoryfp16', 'CPU_ONLY'),
                         ('memoryfp16-gpu', 'memoryfp16', 'CPU_AND_GPU')])
        runs.extend([(f'{args.experiment}-cpu', args.experiment, 'CPU_ONLY'),
                     (f'{args.experiment}-gpu', args.experiment, 'CPU_AND_GPU')])
        for label, variant, units in runs:
            report['stage'] = label
            write_json(path, report)
            row = report['runs'][label] = run_worker(args, label, variant, units, script_path=__file__)
            write_json(path, report)
            if not row['passed']:
                raise ValueError(f'{label} failed; remaining runs skipped')
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
