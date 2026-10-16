#!/usr/bin/env python3
"""Show every original EdgeTAM first-frame mask for previously reviewed points.

Diagnostic only. Original PyTorch on CPU; no Core ML conversion, propagation,
memory encoding, candidate promotion or recovery-policy changes. Predicted IoU
ranks masks but does not establish that a mask covers the intended whole person.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import subprocess
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
from PIL import ImageDraw
import torch

import owned
import validate_export as v
from diagnose_occlusion import sha, save_preview


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', type=Path, required=True)
    parser.add_argument('--upstream', type=Path, default=Path('.upstream-edgetam'))
    parser.add_argument('--selection', type=Path,
                        default=Path('.temporal-runs/occlusion-new-mask-selection-01'))
    parser.add_argument('--output', type=Path,
                        default=Path('.temporal-runs/occlusion-new-mask-alternatives-01'))
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    if platform.system() != 'Darwin' or Path.cwd().resolve() != root:
        parser.error(f'Run on your Mac from {root}')
    output = args.output.resolve()
    bundle = root / '.temporal-runs/share' / (output.name + '.zip')
    if not output.is_relative_to((root / '.temporal-runs').resolve()):
        parser.error('Output must be within .temporal-runs.')
    if output.exists() or bundle.exists():
        parser.error('Output or ZIP exists; choose a fresh --output.')
    if not args.video.is_file():
        parser.error(f'Video missing: {args.video.resolve()}')
    video_hash = sha(args.video)
    paths = sorted(args.selection.glob('point-*/report.json'))
    if not 1 <= len(paths) <= 8:
        parser.error('Expected one to eight completed first-frame selection reports.')
    sources = []
    for path in paths:
        previous = json.loads(path.read_text())
        frames = previous.get('frames', [])
        point = previous.get('pointNormalizedTopLeft', [])
        if (previous.get('contract') != 'bytewave.occlusion-comparison.diagnostic.v1'
                or previous.get('completed') is not True
                or previous.get('initializationOnly') is not True
                or previous.get('sourceVideoSHA256') != video_hash
                or previous.get('sourceFileBytes') != args.video.stat().st_size
                or previous.get('upstream') != owned.UPSTREAM_REVISION
                or previous.get('checkpointSHA256') != owned.CHECKPOINT_SHA256
                or len(frames) != 1 or frames[0].get('index') != 0
                or len(point) != 2
                or not all(np.isfinite(x) and 0 <= x <= 1 for x in point)):
            raise ValueError(f'Invalid or mismatched initialization report: {path}')
        sources.append((path, previous))
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    output.mkdir(parents=True)
    report = {
        'scope': __doc__, 'contract': 'bytewave.initial-mask-alternatives.diagnostic.v1',
        'completed': False, 'readyForDeviceValidation': False,
        'sourceVideoSHA256': video_hash, 'sourceFileBytes': args.video.stat().st_size,
        'upstream': owned.UPSTREAM_REVISION, 'checkpointSHA256': owned.CHECKPOINT_SHA256,
        'scriptSHA256': sha(Path(__file__)), 'torchVersion': torch.__version__,
        'backend': 'Original PyTorch CPU', 'framesDecoded': 0,
        'selectionRule': 'Unchanged highest predicted IoU; all alternatives shown for visual review.',
        'limitation': 'No whole-person quality verdict, alternate-mask pointer, temporal history or Core ML parity claim.',
        'points': []}
    decoder = None
    try:
        v.write_json(output / 'report.json', report)
        decoder = v.video_frames(args.video, count=1)
        _, timestamp, image, tensor, image_hash = next(decoder)
        decoder.close()
        decoder = None
        report.update(framesDecoded=1, imageSHA256=image_hash, ptsSeconds=float(timestamp))
        for path, previous in sources:
            if (previous['frames'][0]['imageSHA256'] != image_hash
                    or previous['frames'][0]['ptsSeconds'] != float(timestamp)):
                raise ValueError(f'Decoded first frame differs from {path}')
        image.save(output / 'source.png')
        model = owned.load_reference(args.upstream)
        with torch.inference_mode():
            features = model.forward_image(owned.normalize(torch.from_numpy(tensor)))
            _, features, positions, sizes = model._prepare_backbone_features(features)
            for number, (path, previous) in enumerate(sources, 1):
                point = previous['pointNormalizedTopLeft']
                coords = [min(float(x) * 1024, 1023) for x in point]
                prompts = {'point_coords': torch.tensor([[coords]], dtype=torch.float32),
                           'point_labels': torch.ones((1, 1), dtype=torch.int32)}
                print(f'Alternatives for point {number}/{len(sources)}: {point}', flush=True)
                report['activePoint'] = number
                v.write_json(output / 'report.json', report)
                _, sam, _, _ = model._track_step(
                    0, True, features, positions, sizes, prompts, None,
                    {'cond_frame_outputs': {}, 'non_cond_frame_outputs': {}},
                    1, False, None)
                expected = [(1, 3, 256, 256), (1, 3, 1024, 1024), (1, 3),
                            (1, 1, 256, 256), (1, 1, 1024, 1024), (1, 256), (1, 1)]
                if len(sam) != len(expected):
                    raise ValueError('Unexpected upstream mask-head outputs.')
                values = []
                for value, shape in zip(sam, expected, strict=True):
                    value = value.detach().cpu().numpy()
                    if value.shape != shape or not np.isfinite(value).all():
                        raise ValueError(f'Invalid mask-head output: expected {shape}, got {value.shape}')
                    values.append(value)
                low, high, scores, selected_low, selected_high, _, object_score = values
                best = int(scores[0].argmax())
                if (not np.array_equal(low[:, best:best + 1], selected_low)
                        or not np.array_equal(high[:, best:best + 1], selected_high)):
                    raise ValueError('Selected mask is not the highest-IoU candidate.')
                old = previous['frames'][0]['streams']['original']
                control = {
                    'lowForegroundFractionMatches': float((selected_low > 0).mean()) == old['foregroundFraction'],
                    'predictedIoUMatches': bool(np.isclose(float(scores[0, best]), old['predictedIoU'], rtol=1e-5, atol=1e-6)),
                    'objectScoreMatches': bool(np.isclose(float(object_score.item()), old['objectScore'], rtol=1e-5, atol=1e-6))}
                row = {'pointNormalizedTopLeft': point, 'sourceReportSHA256': sha(path),
                       'selectedCandidate': best + 1, 'priorSummaryControl': control,
                       'controlLimit': 'Summary values only; prior mask logits were not saved.',
                       'candidates': []}
                panels = {}
                for candidate in range(3):
                    name = f'point-{number:02d}-candidate-{candidate + 1}'
                    score = float(scores[0, candidate])
                    mask = high[:, candidate:candidate + 1]
                    v.save_mask(output / (name + '-mask.png'), mask)
                    label = f'Candidate {candidate + 1} | predicted IoU {score:.4f}'
                    if candidate == best:
                        label += ' | AUTO SELECTED'
                    panels[label] = {'high_res_mask': mask}
                    row['candidates'].append({
                        'candidate': candidate + 1, 'predictedIoU': score,
                        'lowForegroundFraction': float((low[:, candidate:candidate + 1] > 0).mean()),
                        'highForegroundFraction': float((mask > 0).mean()),
                        'mask': name + '-mask.png'})
                marked = image.copy()
                draw = ImageDraw.Draw(marked)
                x, y = coords
                draw.ellipse((x - 7, y - 7, x + 7, y + 7), outline='red', width=3)
                preview = f'point-{number:02d}-alternatives.png'
                save_preview(output / preview, marked, panels, 0, float(timestamp))
                row['preview'] = preview
                report['points'].append(row)
                v.write_json(output / 'report.json', report)
                if not all(control.values()):
                    raise ValueError('Prior first-frame summary did not reproduce; inspect report before interpreting alternatives.')
        report.update(completed=True, activePoint=None)
    except Exception as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        if decoder is not None:
            decoder.close()
        v.write_json(output / 'report.json', report)
        bundle.parent.mkdir(parents=True, exist_ok=True)
        with ZipFile(bundle, 'x', ZIP_DEFLATED) as archive:
            for path in sorted(output.iterdir()):
                if path.is_file():
                    archive.write(path, path.name)
        print(f'Upload this ZIP (completed={report["completed"]}): {bundle}', flush=True)
        subprocess.run(['open', '-R', str(bundle)], check=False)


if __name__ == '__main__':
    main()
