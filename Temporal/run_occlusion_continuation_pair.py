#!/usr/bin/env python3
"""Run the two local continuation diagnostics and collect one share ZIP."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import subprocess
import sys
from zipfile import ZIP_DEFLATED, ZipFile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', choices=('quality', 'edge'), default='quality')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if platform.system() != 'Darwin':
        parser.error('Run locally on the Mac.')
    root = Path(__file__).resolve().parent.parent
    if Path.cwd().resolve() != root:
        parser.error(f'Run from the project directory: {root}')
    edge = args.experiment == 'edge'
    output = (args.output or Path('.temporal-runs/continuation-edge-01' if edge
                                 else '.temporal-runs/continuation-quality-01')).resolve()
    if not output.is_relative_to((root / '.temporal-runs').resolve()):
        parser.error('Output must be inside .temporal-runs.')
    bundle = root / '.temporal-runs/share' / (output.name + ('-review.zip' if edge else '-reports.zip'))
    if output.exists() or bundle.exists():
        parser.error('Output or share ZIP already exists; choose a fresh --output.')
    common = {
        'upstream': Path('.upstream-edgetam'),
        'phone-report': Path('Audit/iphone17pro-worker-occlusion-report.json'),
        'fixture': Path('DeviceValidationData/memoryfp16projection')}
    cases = {
        'worker': {
            'video': Path('assets/worker-occlusion.mov'),
            'comparison': Path('.temporal-runs/worker-occlusion-trim-01/report.json'),
            'memory-probe': Path('.temporal-runs/worker-memory-probe-01/report.json'),
            'appearance-report': Path('.temporal-runs/worker-identity-signals-01/report.json'),
            'backup-control': Path('.temporal-runs/worker-appearance-backup-01/report.json')},
        'dog': {
            'video': Path('assets/dog-occlusion.mov'),
            'comparison': Path('.temporal-runs/dog-pillar-comparison-01/report.json'),
            'memory-probe': Path('.temporal-runs/dog-pillar-memory-probe-01/report.json'),
            'appearance-report': Path('.temporal-runs/dog-identity-signals-01/report.json'),
            'backup-control': Path('.temporal-runs/dog-appearance-backup-01/report.json')}}
    if edge:
        for name, case in cases.items():
            case['backup-control'] = Path('.temporal-runs/continuation-quality-01') / name / 'report.json'
    missing = sorted({str(p) for case in cases.values()
                      for p in {**common, **case}.values() if not p.exists()})
    if missing:
        parser.error('Missing inputs (no runs started):\n' + '\n'.join(missing))
    output.mkdir(parents=True, exist_ok=False)
    try:
        for name, case in cases.items():
            command = [sys.executable, str(root / 'Temporal/diagnose_occlusion_backup.py'),
                       '--continuation-policy', 'appearance-edge' if edge else 'appearance-only',
                       '--output', str(output / name)]
            for key, value in {**common, **case}.items():
                command.extend(['--' + key, str(value)])
            print(f'Running {name} continuation experiment', flush=True)
            subprocess.run(command, check=True, cwd=root)
    finally:
        reports = [p for name in cases for p in [output / name / 'report.json'] if p.is_file()]
        if reports:
            bundle.parent.mkdir(parents=True, exist_ok=True)
            with ZipFile(bundle, 'x', ZIP_DEFLATED) as archive:
                for p in reports:
                    archive.write(p, p.relative_to(output))
                    if edge:
                        # Include all displayed backup frames and continuation
                        # attempts (including rejection), so no second ZIP is needed.
                        report = json.loads(p.read_text())
                        for row in report.get('frames', []):
                            if not row.get('rowCompleted') or not (
                                    row.get('displaySource') == 'backup' or 'continuation' in row):
                                continue
                            preview = (p.parent / row['preview']).resolve()
                            if not preview.is_relative_to(p.parent.resolve()):
                                raise ValueError('Preview path escapes run directory')
                            archive.write(preview, preview.relative_to(output))
            print(f'Upload this ZIP: {bundle}', flush=True)
            subprocess.run(['open', '-R', str(bundle)], check=False)
    for name in cases:
        report = json.loads((output / name / 'report.json').read_text())
        print(name, json.dumps(report['summary'], indent=2), flush=True)


if __name__ == '__main__':
    main()

