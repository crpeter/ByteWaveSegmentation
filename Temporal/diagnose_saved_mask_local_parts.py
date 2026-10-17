#!/usr/bin/env python3
"""Measure local appearance on saved masks against fixed early subject references.

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


from diagnose_saved_mask_identity import describe, public, metrics


def local_describe(raw, mask, image):
    """Raw grid features are contextual, not semantic body-part embeddings."""
    coverage = mask.reshape(64, 16, 64, 16).mean(axis=(1, 3), dtype=np.float64)
    rgb = np.asarray(image, dtype=np.float64) / 255.0
    sums = (rgb * mask[..., None]).reshape(64, 16, 64, 16, 3).sum(axis=(1, 3))
    counts = coverage * 256
    colour = sums / np.maximum(counts[..., None], 1)
    yy, xx = np.indices((64, 64))
    foreground = coverage > 0
    # Fixed spatial sample, independent of later query features or similarity.
    background = (coverage == 0) & (yy % 4 == 0) & (xx % 4 == 0)
    vectors = raw.transpose(1, 2, 0)
    return {
        'foregroundYX': np.column_stack(np.nonzero(foreground)).astype(np.int16),
        'foregroundCoverage': coverage[foreground],
        'foregroundFeatures': vectors[foreground].astype(np.float32),
        'foregroundRGB': colour[foreground],
        'backgroundYX': np.column_stack(np.nonzero(background)).astype(np.int16),
        'backgroundFeatures': vectors[background].astype(np.float32)}


def unit_rows(vectors):
    vectors = vectors.astype(np.float64)
    norms = np.linalg.norm(vectors, axis=1)
    if not np.isfinite(vectors).all() or np.any(norms <= 0):
        raise ValueError('Nonfinite or zero local feature vector.')
    return vectors / norms[:, None]


def distribution(values, weights):
    order = np.argsort(values, kind='stable')
    cumulative = np.cumsum(weights[order]) / weights.sum()
    return {'weightedMean': float(np.average(values, weights=weights)),
            'weightedP10': float(values[order[min(np.searchsorted(cumulative, .1), len(order)-1)]]),
            'weightedMedian': float(values[order[min(np.searchsorted(cumulative, .5), len(order)-1)]]),
            'weightedP90': float(values[order[min(np.searchsorted(cumulative, .9), len(order)-1)]]),
            'minimum': float(values.min()), 'maximum': float(values.max())}


def local_metrics(query, reference, minimum_reference_coverage):
    eligible = np.flatnonzero(reference['foregroundCoverage'] >= minimum_reference_coverage)
    if not len(eligible):
        return {'available': False, 'reason': 'No eligible reference cells.'}
    q = unit_rows(query['foregroundFeatures'])
    r = unit_rows(reference['foregroundFeatures'][eligible])
    b = unit_rows(reference['backgroundFeatures']) if len(reference['backgroundFeatures']) else None
    # Chunked comparison bounds temporary memory; no approximation of maxima.
    winners, similarities, background_scores, background_indices = [], [], [], []
    reverse_scores = np.full(len(r), -np.inf)
    reverse_indices = np.full(len(r), -1, dtype=np.int64)
    for offset in range(0, len(q), 128):
        block = q[offset:offset+128] @ r.T
        win = block.argmax(axis=1)
        winners.extend(win.tolist())
        similarities.extend(block[np.arange(len(block)), win].tolist())
        local_reverse = block.argmax(axis=0)
        local_scores = block[local_reverse, np.arange(len(r))]
        replace = local_scores > reverse_scores
        reverse_scores[replace] = local_scores[replace]
        reverse_indices[replace] = offset + local_reverse[replace]
        if b is not None:
            scores = q[offset:offset+128] @ b.T
            bg_win = scores.argmax(axis=1)
            background_scores.extend(scores[np.arange(len(scores)), bg_win].tolist())
            background_indices.extend(bg_win.tolist())
    winners = np.asarray(winners)
    similarities = np.clip(np.asarray(similarities), -1, 1)
    original_indices = eligible[winners]
    mutual = reverse_indices[winners] == np.arange(len(q))
    colour = np.abs(query['foregroundRGB'] - reference['foregroundRGB'][original_indices]).mean(axis=1)
    weights = query['foregroundCoverage']
    result = {
        'available': True, 'referenceCoverageMinimum': minimum_reference_coverage,
        'referenceCells': len(eligible), 'backgroundCells': len(reference['backgroundYX']),
        'queryCells': len(q), 'queryCoverageSum': float(weights.sum()),
        'foregroundCosine': distribution(similarities, weights),
        'matchedRGBMeanAbsoluteDistance': distribution(colour, weights),
        'mutualNearestCoverageFraction': float(np.average(mutual, weights=weights)),
        'uniqueReferenceCellsUsed': len(np.unique(original_indices)),
        'cells': {'queryYX': query['foregroundYX'].tolist(), 'coverage': weights.tolist(),
                  'matchedReferenceYX': reference['foregroundYX'][original_indices].tolist(),
                  'foregroundCosine': similarities.tolist(), 'rgbMeanAbsoluteDistance': colour.tolist(),
                  'mutualNearest': mutual.tolist()}}
    if b is not None:
        background_scores = np.clip(np.asarray(background_scores), -1, 1)
        margin = similarities - background_scores
        result['backgroundCosine'] = distribution(background_scores, weights)
        result['foregroundMinusBackground'] = distribution(margin, weights)
        result['cells'].update(backgroundCosine=background_scores.tolist(),
                               foregroundMinusBackground=margin.tolist(),
                               matchedBackgroundYX=reference['backgroundYX'][background_indices].tolist())
    else:
        result['foregroundMinusBackground'] = None
    return result


def save_match_map(path, image, mask, matches, index, anchor):
    # Display fixed scales, never rescale a weak frame into a strong-looking map.
    from PIL import ImageDraw
    canvas = Image.new('RGB', (3072, 1064), 'white')
    rgb = np.asarray(image, dtype=np.float64)
    shown = rgb.copy()
    shown[mask] = .5 * shown[mask] + .5 * np.array([80, 150, 255])
    canvas.paste(Image.fromarray(shown.astype(np.uint8)), (0, 40))
    cells = matches['cells']
    fields = [('foregroundCosine', 1.0), ('foregroundMinusBackground', .25)]
    for panel, (key, scale) in enumerate(fields, 1):
        if key not in cells:
            continue
        grid = np.zeros((64, 64), dtype=np.float64)
        for (y, x), value in zip(cells['queryYX'], cells[key]):
            grid[y, x] = value
        values = np.repeat(np.repeat(grid, 16, axis=0), 16, axis=1)
        positive = np.clip(values / scale, 0, 1)
        negative = np.clip(-values / scale, 0, 1)
        heat = np.stack([255 * (1-positive), 255 * (1-negative),
                         255 * (1-np.maximum(positive, negative))], axis=-1)
        view = rgb.copy()
        view[mask] = .25 * rgb[mask] + .75 * heat[mask]
        canvas.paste(Image.fromarray(view.astype(np.uint8)), (panel*1024, 40))
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 12), f'Frame {index}: saved mask; reference {anchor}', fill='black')
    draw.text((1034, 12), 'Local cosine: -1 red, 0 white, +1 green; reference coverage >= 0.5', fill='black')
    draw.text((2058, 12), 'FG minus sampled BG cosine: -0.25 red, 0 white, +0.25 green (clipped)', fill='black')
    canvas.save(path, quality=85)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', type=Path, required=True)
    parser.add_argument('--upstream', type=Path, default=Path('.upstream-edgetam'))
    parser.add_argument('--replay', type=Path,
                        default=Path('.temporal-runs/occlusion-new-whole-person-01/report.json'))
    parser.add_argument('--identity-report', type=Path,
                        default=Path('.temporal-runs/occlusion-new-identity-evidence-01/report.json'))
    parser.add_argument('--anchor-frames', type=int, nargs='+', default=[0, 30, 60])
    parser.add_argument('--inspect-frames', type=int, nargs='+', default=DEFAULT_INSPECTIONS)
    parser.add_argument('--output', type=Path,
                        default=Path('.temporal-runs/occlusion-new-local-parts-01'))
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
    evidence = json.loads(args.identity_report.read_text())
    if (evidence.get('contract') != 'bytewave.saved-mask-identity.diagnostic.v1'
            or evidence.get('completed') is not True
            or evidence.get('sourceReplaySHA256') != sha(args.replay)
            or evidence.get('sourceVideoSHA256') != source_hash
            or evidence.get('scriptSHA256') != sha(Path(__file__).with_name('diagnose_saved_mask_identity.py'))
            or evidence.get('anchorFrames') != anchors_requested
            or evidence.get('inspectFrames') != inspections):
        raise ValueError('Expected completed matching pooled-identity report with the same fixed frames.')
    evidence_rows = {r['index']: r for r in list(evidence['anchors'].values()) + evidence['measurements']}
    if sorted(evidence_rows) != sorted(anchors_requested + inspections):
        raise ValueError('Prior identity report has incomplete requested frames.')
    count = inspections[-1] + 1
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    output.mkdir(parents=True)
    (output / 'previews').mkdir()
    (output / 'features').mkdir()
    (output / 'match-maps').mkdir()
    report = {
        'scope': __doc__, 'contract': 'bytewave.saved-mask-local-parts.diagnostic.v1',
        'completed': False, 'readyForDeviceValidation': False,
        'sourceIdentityReportSHA256': sha(args.identity_report),
        'pooledDescriptorSourceSHA256': sha(Path(__file__).with_name('diagnose_saved_mask_identity.py')),
        'pooledReproductionControlsPassed': 0,
        'pooledReproductionTolerance': {'atol': 1e-7, 'rtol': 1e-6},
        'localDefinition': {
            'query': 'Every 64x64 grid cell touched by saved foreground; exact mask coverage weights.',
            'reference': 'Per-anchor nearest foreground feature; coverage >= .5 primary, >= 1.0 interior sensitivity check.',
            'background': 'Same anchor: every fourth grid row/column with exactly zero mask coverage. Sparse contextual comparator, not labeled negatives or exhaustive background.',
            'metrics': 'Coverage-weighted cosine distributions, same matched-cell masked mean-RGB distance, reverse-nearest fraction, unique reference cells; per-cell matches saved.',
            'margin': 'Best eligible subject cosine minus best sampled-background cosine. No acceptance cutoff; unequal reference bank sizes can bias maxima.',
            'features': 'Raw contextual backbone_fpn[2] vectors and masked local mean RGB in compressed NPZ; not semantic part labels or identity embeddings. Feature receptive fields extend beyond each 16x16 display tile; partial cells may be context dominated.',
            'maps': 'Primary reference coverage .5 only; fixed scales across frames. All anchors shown independently, never cherry-picked.'},
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
        local_anchors = {}
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
                print(f'Local appearance: frame {index}/{count - 1}', flush=True)
                report['activeFrameIndex'] = index
                v.write_json(output / 'report.json', report)
                previous = evidence_rows[index]
                if (previous['imageSHA256'] != image_hash or previous['ptsSeconds'] != float(timestamp)
                        or previous['savedMaskSHA256'] != sha(mask_path) or previous['observation'] != obs):
                    raise ValueError(f'Prior identity inputs differ at frame {index}.')
                value = None
                local = None
                if obs['objectScore'] > 0 and not obs['lowMaskEmpty'] and not obs['highMaskEmpty']:
                    raw = model.forward_image(owned.normalize(torch.from_numpy(tensor)))['backbone_fpn'][2]
                    if tuple(raw.shape) != (1, 256, 64, 64) or not torch.isfinite(raw).all().item():
                        raise ValueError('Invalid original raw encoder features.')
                    value = describe(raw[0].numpy(), mask, image)
                    local = local_describe(raw[0].numpy(), mask, image)
                    report['encoderCalls'] += 1
                expected = previous['descriptor']
                if (value is None) != (expected is None):
                    raise ValueError(f'Prior descriptor presence differs at {index}.')
                if value is not None:
                    for key, item in public(value).items():
                        if not np.allclose(item, expected[key], atol=1e-7, rtol=1e-6):
                            raise ValueError(f'Prior pooled descriptor differs at {index}: {key}')
                report['pooledReproductionControlsPassed'] += 1
                preview = f'previews/{"anchor" if index in anchors_requested else "frame"}-{index:04d}.png'
                # Mask logits are synthetic only for displaying this saved binary mask.
                shown = {'high_res_mask': np.where(mask, 1.0, -1.0)[None, None]}
                save_preview(output / preview, image, {'saved chosen mask; appearance measurement only': shown}, index, float(timestamp))
                entry = {'index': index, 'ptsSeconds': float(timestamp), 'imageSHA256': image_hash,
                         'savedMaskSHA256': sha(mask_path), 'observation': obs,
                         'descriptor': public(value) if value is not None else None, 'preview': preview}
                entry['localFeatures'] = None
                if local is not None:
                    feature_path = output / 'features' / f'frame-{index:04d}.npz'
                    np.savez_compressed(feature_path, **local)
                    entry['localFeatures'] = {'path': str(feature_path.relative_to(output)),
                                              'sha256': sha(feature_path),
                                              'foregroundCells': len(local['foregroundYX']),
                                              'backgroundCells': len(local['backgroundYX'])}
                if index in anchors_requested:
                    if value is None:
                        raise ValueError(f'Anchor {index} is not present.')
                    anchors[index] = value
                    local_anchors[index] = local
                    report['anchors'][str(index)] = entry
                else:
                    if len(anchors) != len(anchors_requested):
                        raise ValueError('Missing earlier fixed references.')
                    entry['againstAnchors'] = {str(i): metrics(value, ref) if value is not None else None
                                               for i, ref in anchors.items()}
                    entry['matchingAnchorFrames'] = [int(i) for i, m in entry['againstAnchors'].items()
                                                     if m is not None and m['existingJointAppearanceRuleMatches']]
                    entry['localAgainstAnchors'] = {}
                    for anchor, reference in local_anchors.items():
                        if local is None:
                            entry['localAgainstAnchors'][str(anchor)] = None
                            continue
                        checks = {'majorityReference': local_metrics(local, reference, .5),
                                  'interiorReference': local_metrics(local, reference, 1.0)}
                        primary = checks['majorityReference']
                        if primary['available']:
                            map_path = output / 'match-maps' / f'frame-{index:04d}-anchor-{anchor:04d}.jpg'
                            save_match_map(map_path, image, mask, primary, index, anchor)
                            checks['preview'] = str(map_path.relative_to(output))
                        entry['localAgainstAnchors'][str(anchor)] = checks
                    report['measurements'].append(entry)
                v.write_json(output / 'report.json', report)
        report.update(completed=report['framesDecodedAndChecked'] == count
                      and len(report['measurements']) == len(inspections)
                      and report['pooledReproductionControlsPassed'] == len(requested), activeFrameIndex=None)
        report['summary'] = {
            'meaning': 'Signal measurements only; no correct-identity or tracking-quality pass/fail.',
            'measurements': [{'index': r['index'], 'ptsSeconds': r['ptsSeconds'],
                              'observation': r['observation'], 'againstAnchors': r['againstAnchors'],
                              'matchingAnchorFrames': r['matchingAnchorFrames'], 'preview': r['preview'],
                              'localAgainstAnchors': {a: None if checks is None else
                                  {name: {k: val for k, val in result.items() if k != 'cells'}
                                   if isinstance(result, dict) else result
                                   for name, result in checks.items()}
                                  for a, checks in r['localAgainstAnchors'].items()}}
                             for r in report['measurements']]}
        v.write_json(output / 'summary.json', report['summary'])
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
