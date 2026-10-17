#!/usr/bin/env python3
"""Measure identity evidence on saved whole-person masks without rerunning tracking.

Original PyTorch CPU image encoder only, on explicitly inspected nonempty masks
and fixed earlier references. No tracker/state mutation, pointer reconstruction,
recovery decision, threshold tuning, model export or app modification.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import platform
import subprocess
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
from PIL import Image
import torch

import owned
import validate_export as v
from diagnose_occlusion import sha, save_preview

DEFAULT_INSPECTIONS = [90, 150, 165, 179, 197, 210, 225, 235, 237,
                       270, 277, 278, 285, 300, 330, 343, 345, 346,
                       355, 358, 360, 375, 386, 387, 542, 545, 549, 550]


def describe(raw, mask, image):
    # Same encoder pooling / RGB definition as diagnose_occlusion_identity.py.
    # Pointer similarity cannot be recovered from saved binary masks; omit it.
    pixels = int(mask.sum())
    coverage = mask.reshape(64, 16, 64, 16).mean(axis=(1, 3), dtype=np.float64)
    pooled = (raw.astype(np.float64) * coverage[None]).sum(axis=(1, 2)) / coverage.sum()
    rgb = np.asarray(image, dtype=np.uint8)[mask].astype(np.int32)
    bins = rgb // 32
    joint = bins[:, 0] * 64 + bins[:, 1] * 8 + bins[:, 2]
    histogram = np.bincount(joint, minlength=512).astype(np.float64) / pixels
    ys, xs = np.nonzero(mask)
    return {'pooledEncoder': pooled, 'rgbHistogram': histogram,
            'highMaskPixels': pixels, 'featureCoverageSum': float(coverage.sum()),
            'boxNormalizedXYXY': [float(xs.min() / 1024), float(ys.min() / 1024),
                                  float((xs.max() + 1) / 1024), float((ys.max() + 1) / 1024)]}


def public(value):
    return {k: x.tolist() if isinstance(x, np.ndarray) else x for k, x in value.items()}


def metrics(value, reference):
    left, right = value['pooledEncoder'], reference['pooledEncoder']
    norm = float(np.linalg.norm(left) * np.linalg.norm(right))
    similarity = float(np.clip(np.dot(left, right) / norm, -1, 1)) if norm > 0 else None
    colour = float(np.minimum(value['rgbHistogram'], reference['rgbHistogram']).sum())
    return {'pooledEncoderCosine': similarity, 'rgbHistogramIntersection': colour,
            'highMaskAreaRatio': value['highMaskPixels'] / reference['highMaskPixels'],
            'existingJointAppearanceRuleMatches': bool(similarity is not None and similarity >= .8 and colour >= .5)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', type=Path, required=True)
    parser.add_argument('--upstream', type=Path, default=Path('.upstream-edgetam'))
    parser.add_argument('--replay', type=Path,
                        default=Path('.temporal-runs/occlusion-new-whole-person-01/report.json'))
    parser.add_argument('--anchor-frames', type=int, nargs='+', default=[0, 30, 60])
    parser.add_argument('--inspect-frames', type=int, nargs='+', default=DEFAULT_INSPECTIONS)
    parser.add_argument('--output', type=Path,
                        default=Path('.temporal-runs/occlusion-new-identity-evidence-01'))
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    if platform.system() != 'Darwin' or Path.cwd().resolve() != root:
        parser.error(f'Run on your Mac from {root}')
    output = args.output.resolve()
    bundle = root / '.temporal-runs/share' / (output.name + '.zip')
    if not output.is_relative_to((root / '.temporal-runs').resolve()):
        parser.error('Output must be within .temporal-runs.')
    if output.exists() or bundle.exists():
        parser.error('Output or ZIP already exists; choose a fresh --output.')
    prior = json.loads(args.replay.read_text())
    source_hash = sha(args.video)
    def all_true(value):
        return all(all_true(v) for v in value.values()) if isinstance(value, dict) and value else value is True
    if (prior.get('contract') != 'bytewave.initial-candidate-replay.diagnostic.v1'
            or prior.get('completed') is not True
            or prior.get('sourceVideoSHA256') != source_hash
            or prior.get('sourceFileBytes') != args.video.stat().st_size
            or prior.get('upstream') != owned.UPSTREAM_REVISION
            or prior.get('checkpointSHA256') != owned.CHECKPOINT_SHA256
            or prior.get('scriptSHA256') != sha(Path(__file__).with_name('diagnose_initial_candidate_replay.py'))
            or not all_true(prior.get('controls', {}))):
        raise ValueError('Expected controlled completed replay of this video with the current replay script.')
    rows = prior['frames']
    total = prior['effectiveFrames']
    anchors_requested = sorted(set(args.anchor_frames))
    inspections = sorted(set(args.inspect_frames))
    if (not isinstance(total, int) or not 20 <= total <= 600
            or [r['index'] for r in rows] != list(range(total))
            or not 1 <= len(anchors_requested) <= 8 or not 1 <= len(inspections) <= 64
            or not 0 <= anchors_requested[0] <= anchors_requested[-1] < inspections[0]
            or inspections[-1] >= total):
        parser.error('Choose 1-8 earlier anchors and 1-64 later inspection frames within completed replay.')
    count = inspections[-1] + 1
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    output.mkdir(parents=True)
    (output / 'previews').mkdir()
    report = {
        'scope': __doc__, 'contract': 'bytewave.saved-mask-identity.diagnostic.v1',
        'completed': False, 'readyForDeviceValidation': False,
        'sourceReplaySHA256': sha(args.replay), 'sourceVideoSHA256': source_hash,
        'sourceFileBytes': args.video.stat().st_size,
        'upstream': owned.UPSTREAM_REVISION, 'checkpointSHA256': owned.CHECKPOINT_SHA256,
        'scriptSHA256': sha(Path(__file__)), 'torchVersion': torch.__version__,
        'stream': 'chosen', 'backend': 'Original PyTorch CPU image encoder',
        'pointNormalizedTopLeft': prior['pointNormalizedTopLeft'],
        'initialCandidate': prior['initialCandidate'],
        'anchorFrames': anchors_requested, 'inspectFrames': inspections,
        'referenceNumFrames': total, 'framesToDecode': count,
        'framesDecodedAndChecked': 0, 'encoderCalls': 0,
        'descriptorDefinition': {
            'encoder': 'Raw backbone_fpn[2], pooled with exact 16x16-tile binary high-mask coverage.',
            'colour': 'Joint RGB 8x8x8 histogram within saved high mask; normalized intersection.',
            'pointer': 'Not measured: replay saved no object-pointer tensors.',
            'missing': 'Nonpositive prior object score or empty prior low/high mask gives null descriptor.',
            'geometry': 'Mask area and normalized box, no minimum-area threshold.'},
        'referenceRule': {'pooledEncoderMinimum': .8, 'rgbIntersectionMinimum': .5,
                          'meaning': 'Same-anchor appearance clause from frozen recovery research, observation only. Not a complete recovery rule or calibrated identity probability; original CPU features here, previous policy used GPU features.'},
        'limits': 'Earlier reference masks require visual review, are fixed and never refreshed. Selected later frames are reviewed examples, not held-out labels. Context, skin/clothes and lighting can confuse descriptors. No thresholds applied to output or history.',
        'anchors': {}, 'measurements': [], 'activeFrameIndex': None}
    decoder = None
    try:
        v.write_json(output / 'report.json', report)
        model = owned.load_reference(args.upstream)
        anchors = {}
        requested = set(anchors_requested + inspections)
        decoder = v.video_frames(args.video, count=count)
        with torch.inference_mode():
            for index, timestamp, image, tensor, image_hash in decoder:
                old = rows[index]
                if image_hash != old['imageSHA256'] or float(timestamp) != old['ptsSeconds']:
                    raise ValueError(f'Decoded pixels or timestamp differ at frame {index}.')
                mask_path = args.replay.parent / 'masks' / 'chosen' / f'frame-{index:04d}.png'
                with Image.open(mask_path) as mask_image:
                    if mask_image.mode != 'L':
                        raise ValueError(f'Expected grayscale binary mask at {index}.')
                    pixels = np.asarray(mask_image)
                if pixels.shape != (1024, 1024) or not np.isin(pixels, [0, 255]).all():
                    raise ValueError(f'Invalid binary mask at {index}.')
                mask = pixels > 0
                obs = old['streams']['chosen']
                if (float(mask.mean()) != obs['highForegroundFraction']
                        or bool(not mask.any()) != obs['highMaskEmpty']):
                    raise ValueError(f'Saved mask disagrees with replay at {index}.')
                report['framesDecodedAndChecked'] += 1
                if index not in requested:
                    continue
                print(f'Appearance evidence: frame {index}/{count - 1}', flush=True)
                report['activeFrameIndex'] = index
                v.write_json(output / 'report.json', report)
                value = None
                if obs['objectScore'] > 0 and not obs['lowMaskEmpty'] and not obs['highMaskEmpty']:
                    raw = model.forward_image(owned.normalize(torch.from_numpy(tensor)))['backbone_fpn'][2]
                    if tuple(raw.shape) != (1, 256, 64, 64) or not torch.isfinite(raw).all().item():
                        raise ValueError('Invalid original raw encoder features.')
                    value = describe(raw[0].numpy(), mask, image)
                    report['encoderCalls'] += 1
                preview = f'previews/{"anchor" if index in anchors_requested else "frame"}-{index:04d}.png'
                # Mask logits are synthetic only for displaying this saved binary mask.
                shown = {'high_res_mask': np.where(mask, 1.0, -1.0)[None, None]}
                save_preview(output / preview, image, {'saved chosen mask; appearance measurement only': shown}, index, float(timestamp))
                entry = {'index': index, 'ptsSeconds': float(timestamp), 'imageSHA256': image_hash,
                         'savedMaskSHA256': sha(mask_path), 'observation': obs,
                         'descriptor': public(value) if value is not None else None, 'preview': preview}
                if index in anchors_requested:
                    if value is None:
                        raise ValueError(f'Anchor {index} is not present.')
                    anchors[index] = value
                    report['anchors'][str(index)] = entry
                else:
                    if len(anchors) != len(anchors_requested):
                        raise ValueError('Missing earlier fixed references.')
                    entry['againstAnchors'] = {str(i): metrics(value, ref) if value is not None else None
                                               for i, ref in anchors.items()}
                    entry['matchingAnchorFrames'] = [int(i) for i, m in entry['againstAnchors'].items()
                                                     if m is not None and m['existingJointAppearanceRuleMatches']]
                    report['measurements'].append(entry)
                v.write_json(output / 'report.json', report)
        report.update(completed=report['framesDecodedAndChecked'] == count
                      and len(report['measurements']) == len(inspections), activeFrameIndex=None)
        report['summary'] = {
            'meaning': 'Signal measurements only; no correct-identity or tracking-quality pass/fail.',
            'measurements': [{'index': r['index'], 'ptsSeconds': r['ptsSeconds'],
                              'observation': r['observation'], 'againstAnchors': r['againstAnchors'],
                              'matchingAnchorFrames': r['matchingAnchorFrames'], 'preview': r['preview']}
                             for r in report['measurements']]}
    except Exception as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        if decoder is not None:
            decoder.close()
        v.write_json(output / 'report.json', report)
        bundle.parent.mkdir(parents=True, exist_ok=True)
        with ZipFile(bundle, 'x', ZIP_DEFLATED) as archive:
            for path in sorted(output.rglob('*')):
                if path.is_file():
                    archive.write(path, path.relative_to(output))
        print(f'Upload this ZIP (completed={report["completed"]}): {bundle}', flush=True)
        subprocess.run(['open', '-R', str(bundle)], check=False)


if __name__ == '__main__':
    main()
