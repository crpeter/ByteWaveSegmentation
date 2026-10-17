#!/usr/bin/env python3
"""Compare first-frame selection points; collect masks and reports in one ZIP.

No temporal propagation or recovery-policy tuning. Defaults are three candidate
points for the reviewed occlusion-new clip, not automatic subject detection.
"""
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
    parser.add_argument('--video', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('.temporal-runs/occlusion-new-mask-selection-01'))
    parser.add_argument('--point', type=float, nargs=2, action='append',
                        help='Repeat to compare candidate points; defaults to head, shoulder and forearm on reviewed clip.')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    if platform.system() != 'Darwin' or Path.cwd().resolve() != root:
        parser.error(f'Run on your Mac from {root}')
    points = args.point or [[0.50, 0.28], [0.36, 0.40], [0.50, 0.45]]
    if not 1 <= len(points) <= 8 or not all(0 <= v <= 1 for p in points for v in p):
        parser.error('Choose one to eight finite normalized points.')
    output = args.output.resolve()
    bundle = root / '.temporal-runs/share' / (output.name + '.zip')
    if not output.is_relative_to((root / '.temporal-runs').resolve()):
        parser.error('Output must be within .temporal-runs.')
    if output.exists() or bundle.exists():
        parser.error('Output or ZIP exists; choose a fresh --output.')
    if not args.video.is_file():
        parser.error(f'Video missing: {args.video.resolve()}')
    output.mkdir(parents=True)
    try:
        for index, point in enumerate(points, 1):
            print(f'Selection {index}/{len(points)}: {point}', flush=True)
            subprocess.run([
                sys.executable, str(root / 'Temporal/diagnose_occlusion.py'),
                '--upstream', '.upstream-edgetam', '--video', str(args.video),
                '--phone-report', 'Audit/iphone17pro-worker-occlusion-report.json',
                '--fixture', 'DeviceValidationData/memoryfp16projection',
                '--new-video', '--initialization-only',
                '--point', str(point[0]), str(point[1]),
                '--output', str(output / f'point-{index:02d}')], cwd=root, check=True)
    finally:
        files = sorted(p for p in output.rglob('*') if p.is_file())
        if files:
            bundle.parent.mkdir(parents=True, exist_ok=True)
            with ZipFile(bundle, 'x', ZIP_DEFLATED) as archive:
                for p in files:
                    archive.write(p, p.relative_to(output))
            print(f'Upload this ZIP: {bundle}', flush=True)
            subprocess.run(['open', '-R', str(bundle)], check=False)
    for index, point in enumerate(points, 1):
        report = json.loads((output / f'point-{index:02d}/report.json').read_text())
        print(f'Point {point}: completed={report["completed"]}; inspect first-frame masks, not just numerical parity.')

if __name__ == '__main__':
    main()
