#!/usr/bin/env python3
"""Inspect CPU/NE plans for the existing and diagnostic FP32 encoder packages.

Loads/compiles existing packages; no prediction, tracing, conversion or mutation.
Planned device preference is not measured hardware execution or a timing share.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import platform


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path, value):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def inspect(package):
    import coremltools as ct
    from coremltools.models.compute_plan import MLComputePlan
    model = ct.models.MLModel(str(package), compute_units=ct.ComputeUnit.CPU_AND_NE)
    # The compiled path is valid only while its MLModel remains alive.
    plan = MLComputePlan.load_from_path(model.get_compiled_model_path(), compute_units=ct.ComputeUnit.CPU_AND_NE)
    program = plan.model_structure.program
    if program is None:
        raise ValueError('Expected an ML Program compute plan')
    rows = []
    constants = 0
    def visit(block, location):
        nonlocal constants
        for index, op in enumerate(block.operations):
            if op.operator_name == 'const':
                constants += 1
                continue
            usage = plan.get_compute_device_usage_for_mlprogram_operation(op)
            preferred = type(usage.preferred_compute_device).__name__ if usage is not None and usage.preferred_compute_device is not None else 'unspecified'
            supported = sorted({type(device).__name__ for device in usage.supported_compute_devices}) if usage is not None else []
            rows.append({'location': f'{location}/{index}', 'type': op.operator_name,
                         'outputs': [v.name for v in op.outputs],
                         'preferred': preferred, 'supported': supported})
            for child_index, child in enumerate(op.blocks):
                visit(child, f'{location}/{index}/block{child_index}')
    for name, function in program.functions.items():
        visit(function.block, name)
    if not rows:
        raise ValueError('No nonconstant operations found in compute plan')
    preferred = Counter(row['preferred'] for row in rows)
    supported = Counter(device for row in rows for device in row['supported'])
    by_type = {}
    for row in rows:
        by_type.setdefault(row['type'], Counter())[row['preferred']] += 1
    return {'constantOperationsExcluded': constants, 'nonconstantOperationCount': len(rows),
            'preferredCounts': dict(preferred), 'supportedCounts': dict(supported),
            'preferredCountsByOperationType': {k: dict(v) for k, v in sorted(by_type.items())},
            'operations': rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--diagnostic', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if platform.system() != 'Darwin':
        parser.error('Compute plans require macOS')
    import coremltools as ct
    manifest_path = args.run / 'models/manifest.json'
    manifest = json.loads(manifest_path.read_text())
    diagnostic_path = args.diagnostic / 'report.json'
    diagnostic = json.loads(diagnostic_path.read_text())
    if diagnostic.get('completed') is not True or diagnostic.get('sourceManifestSHA256') != sha(manifest_path):
        raise ValueError('Diagnostic and original source manifest do not match')
    packages = {
        'existing': (args.run / 'models/BWTemporalImageEncoder.mlpackage', manifest['models']['ImageEncoder']['files']),
        'fp32': (args.diagnostic / 'DiagnosticImageEncoderFP32.mlpackage', diagnostic['fp32Encoder']['files'])}
    for name, (package, expected) in packages.items():
        if {str(p.relative_to(package)): sha(p) for p in package.rglob('*') if p.is_file()} != expected:
            raise ValueError(f'{name} package bytes changed since encoder comparison')
    args.output.mkdir(parents=True, exist_ok=False)
    report = {'scope': 'Planned CPU/NE device preferences for verified existing encoder packages. '
                       'Nonconstant counts exclude constants; supported counts may overlap. '
                       'Not observed hardware execution, numerical validation or measured performance.',
              'readyForDeviceValidation': False, 'completed': False, 'computeUnits': 'CPU_AND_NE',
              'coremltoolsVersion': ct.__version__, 'macOS': platform.mac_ver()[0], 'machine': platform.machine(),
              'sourceManifestSHA256': sha(manifest_path), 'sourceDiagnosticReportSHA256': sha(diagnostic_path), 'models': {}}
    report_path = args.output / 'report.json'
    for name, (package, hashes) in packages.items():
        report['activeModel'] = name
        write_json(report_path, report)
        print(f'Inspecting {name} encoder plan...', flush=True)
        try:
            report['models'][name] = {'files': hashes, **inspect(package)}
            print(name, report['models'][name]['preferredCounts'], flush=True)
        except Exception as error:
            report['models'][name] = {'files': hashes, 'error': f'{type(error).__name__}: {error}'}
            print(name, report['models'][name]['error'], flush=True)
        finally:
            write_json(report_path, report)
    report.pop('activeModel', None)
    report['completed'] = True
    report['allPlansLoaded'] = all('error' not in model for model in report['models'].values())
    write_json(report_path, report)
    summary = {**report, 'models': {name: {k: value for k, value in model.items() if k != 'operations'}
                                   for name, model in report['models'].items()}}
    write_json(args.output / 'summary.json', summary)
    print(f'Report: {report_path}', flush=True)


if __name__ == '__main__':
    main()
