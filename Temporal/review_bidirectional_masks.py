#!/usr/bin/env python3
"""Review saved forward/backward masks without running models or decoding video.

Validates the reviewed source report and every binary mask, compares descriptive
selection rules, and creates one local HTML report. Disagreement is a review cue,
not identity confidence. No fused masks, automatic acceptance or app changes.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import shutil

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
STREAMS = ('savedForward', 'backward')
THRESHOLDS = (0.5, 0.75, 0.9, 0.95)
# Selected qualitative examples, not pixel labels or an independent test set.
# Drawn from the prior visual review; 224/343 additionally inspected enlarged.
CASES = {
    165: ('savedForward', None, 'Forward retains a genuine left-edge target partial; backward is blank.'),
    191: (None, None, 'Tiny backward edge fragment; identity unresolved.'),
    224: ('savedForward', 'backward', 'Backward adds the foreground man\'s pointing hand/forearm.'),
    225: ('savedForward', 'backward', 'Backward adds the foreground man\'s hand/forearm.'),
    229: ('savedForward', 'backward', 'Backward adds the foreground man\'s hand/forearm.'),
    261: (None, None, 'Narrow backward strip at the occluder boundary; identity unresolved.'),
    277: ('backward', None, 'Backward has more useful target coverage; neither outline is perfect.'),
    300: ('backward', None, 'Backward recovers target head/torso; forearm coverage is incomplete.'),
    310: (None, None, 'Both blank despite a visible target sliver. Neither pass supplies the missing mask.'),
    343: (None, 'savedForward', 'Forward paints the foreground hand; backward is blank. No complete mask asserted.'),
    346: ('backward', 'savedForward', 'Backward follows target head; forward selects the foreground hand.'),
    354: ('backward', 'savedForward', 'Backward avoids much of the foreground woman/hand mixed into forward.'),
    360: ('backward', 'savedForward', 'Backward retains target head/body and excludes much of the wrong foreground.'),
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def child(root, name):
    path = (root / name).resolve()
    if Path(name).is_absolute() or not path.is_relative_to(root.resolve()):
        raise ValueError(f'Invalid source-relative path: {name}')
    return path


def all_true(value):
    if isinstance(value, dict):
        return bool(value) and all(all_true(x) for x in value.values())
    return value is True


def mask(path, expected_sha):
    if sha(path) != expected_sha:
        raise ValueError(f'Mask hash mismatch: {path}')
    with Image.open(path) as image:
        pixels = np.array(image)
        if image.mode != 'L' or pixels.shape != (1024, 1024) or not np.isin(pixels, [0, 255]).all():
            raise ValueError(f'Invalid binary mask: {path}')
    return pixels != 0


def agreement(a, b):
    intersect = int(np.logical_and(a, b).sum())
    union = int(np.logical_or(a, b).sum())
    return dict(intersectionPixels=intersect, unionPixels=union,
                iou=intersect / union if union else None,
                forwardOnlyPixels=int(np.logical_and(a, ~b).sum()),
                backwardOnlyPixels=int(np.logical_and(b, ~a).sum()), bothEmpty=union == 0)


def intervals(indices, rows):
    groups = []
    for index in sorted(indices):
        if groups and index == groups[-1][-1] + 1:
            groups[-1].append(index)
        else:
            groups.append([index])
    return [dict(firstFrame=g[0], lastFrame=g[-1], frames=len(g),
                 firstSeconds=rows[g[0]]['ptsSeconds'], lastSeconds=rows[g[-1]]['ptsSeconds'])
            for g in groups]


def choose(a, b, larger=True):
    if a == b:
        return 'tie'
    return STREAMS[0] if (a > b) == larger else STREAMS[1]


def analyze(report_path, review_path):
    review = json.loads(review_path.read_text())
    report = json.loads(report_path.read_text())
    if (review['contract'] != 'bytewave.backward-reacquisition.review.v1'
            or report['contract'] != 'bytewave.backward-reacquisition.diagnostic.v1'
            or sha(report_path) != review['source']['reportSHA256']):
        raise ValueError('Expected the exact reviewed backward diagnostic report.')
    for key in ('sourceVideoSHA256', 'scriptSHA256', 'seedMaskSHA256'):
        if report[key] != review['source'][key]:
            raise ValueError(f'Source identity mismatch: {key}')
    if report['completed'] is not True or not all_true(report['controls']):
        raise ValueError('Source diagnostic is incomplete or its initialization control failed.')
    source = report_path.parent
    original = report['frames']
    seed, last = report['seedFrame'], report['lastFrame']
    if ([f['index'] for f in original] != list(range(seed, last - 1, -1))
            or [f['processingOrdinal'] for f in original] != list(range(len(original)))
            or report['reverseIndexControlsChecked'] != len(original)):
        raise ValueError('Source frame sequence or control count is incomplete.')
    rows = {}
    previews = {}
    for f in reversed(original):
        index = f['index']
        if not math.isfinite(f['ptsSeconds']) or not all_true(f['directionIndexControl']):
            raise ValueError(f'Invalid time or failed direction control: {index}')
        arrays = []
        for name in STREAMS:
            info = f['maskFiles'][name]
            a = mask(child(source, info['path']), info['sha256'])
            observation = f['streams'][name]
            if (observation['highMaskEmpty'] != (not bool(a.any()))
                    or observation['highForegroundFraction'] != float(a.mean())
                    or not math.isfinite(observation['objectScore'])):
                raise ValueError(f'Mask/observation mismatch: {index}/{name}')
            arrays.append(a)
        overlap = agreement(*arrays)
        if overlap != f['highMaskAgreement']:
            raise ValueError(f'Overlap mismatch: {index}')
        if index == seed and f['maskFiles']['backward']['sha256'] != report['seedMaskSHA256']:
            raise ValueError('Unexpected backward seed mask.')
        preview = child(source, f['preview'])
        with Image.open(preview) as image:
            if image.size != (2048, 1064):
                raise ValueError(f'Unexpected paired preview size: {preview}')
            image.verify()
        previews[index] = preview
        af, ab = [int(a.sum()) for a in arrays]
        sf, sb = [f['streams'][s]['objectScore'] for s in STREAMS]
        category = ('bothEmpty' if overlap['bothEmpty'] else
                    'bothNonempty' if af and ab else 'oneNonempty')
        # These are arithmetic descriptions, not recommended mask selections.
        rules = ({'higherObjectScore': 'bothEmpty', 'largerMask': 'bothEmpty',
                  'smallerNonemptyMask': 'bothEmpty'} if overlap['bothEmpty'] else {
            'higherObjectScore': choose(sf, sb), 'largerMask': choose(af, ab),
            'smallerNonemptyMask': choose(af, ab, larger=False) if af and ab else choose(af, ab)})
        if index == seed:
            rules = {key: 'seedExcluded' for key in rules}
        rows[index] = dict(index=index, ptsSeconds=f['ptsSeconds'], category=category,
                           isSeed=index == seed, iou=overlap['iou'],
                           streams=f['streams'], maskFiles=f['maskFiles'],
                           rules=rules, objectScoreGapBackwardMinusForward=sb - sf,
                           preview=f'previews/frame-{index:04d}.jpg', previewSHA256=sha(preview))
    if any(rows[i]['ptsSeconds'] >= rows[i + 1]['ptsSeconds'] for i in range(last, seed)):
        raise ValueError('Source times do not increase chronologically.')
    return report, review, rows, previews


def summarize(source, rows, threshold):
    usable = [r for r in rows.values() if not r['isSeed']]
    disagreement = [r['index'] for r in usable if r['iou'] is not None and r['iou'] < threshold]
    empty_runs = intervals([r['index'] for r in usable if r['category'] == 'bothEmpty'], rows)
    boundaries = set()
    for run in empty_runs:
        # Inspect both ends of shared absence. Interiors stay accessible in All frames.
        boundaries.update(range(run['firstFrame'], min(run['firstFrame'] + 2, run['lastFrame'] + 1)))
        boundaries.update(range(max(run['firstFrame'], run['lastFrame'] - 1), run['lastFrame'] + 1))
    for row in rows.values():
        row['reviewReasons'] = (['maskDisagreement'] if row['index'] in disagreement else [])
        if row['index'] in boundaries:
            row['reviewReasons'].append('sharedAbsenceBoundary')
    sweep = []
    for t in THRESHOLDS:
        found = [r['index'] for r in usable if r['iou'] is not None and r['iou'] < t]
        sweep.append(dict(iouBelow=t, flaggedFrames=len(found), intervals=intervals(found, rows)))
    cases = []
    for index, (preferred, reject, note) in CASES.items():
        if index not in rows:
            continue
        row = rows[index]
        cases.append(dict(frame=index, ptsSeconds=row['ptsSeconds'], preferred=preferred,
                          reject=reject, note=note, choices=row['rules'],
                          reviewReasons=row['reviewReasons'],
                          higherScoreSelectsKnownWrong=reject is not None and row['rules']['higherObjectScore'] == reject))
    return dict(
        comparedFrames=len(rows), evaluatedFramesExcludingSeed=len(usable),
        sourcePresenceCounts=dict(Counter(r['category'] for r in usable)),
        disagreementFrames=len(disagreement), disagreementIntervals=intervals(disagreement, rows),
        sharedAbsenceIntervals=empty_runs, sharedAbsenceBoundaryFrames=sorted(boundaries),
        queuedFrames=len(set(disagreement) | boundaries),
        unqueuedFramesExcludingSeed=sum(not r['reviewReasons'] for r in usable),
        thresholdSensitivity=sweep, qualitativeCases=cases,
        higherScoreKnownWrongFrames=[c['frame'] for c in cases if c['higherScoreSelectsKnownWrong']],
        note='Cases are selected qualitative comparisons, not complete annotations or a measured accuracy rate. Neither rule ranking nor queue membership validates identity.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, default=ROOT / '.temporal-runs/occlusion-new-backward-01/report.json')
    parser.add_argument('--review', type=Path, default=ROOT / 'Audit/mac-new-backward-reacquisition-review.json')
    parser.add_argument('--output', type=Path, default=ROOT / '.temporal-runs/occlusion-new-bidirectional-review-01')
    parser.add_argument('--review-iou', type=float, default=0.9,
                        help='Display/review cutoff only; never a correctness gate or selection rule.')
    args = parser.parse_args()
    if not math.isfinite(args.review_iou) or not 0 < args.review_iou <= 1:
        parser.error('--review-iou must be in (0, 1].')
    output = args.output.resolve()
    if output.exists():
        parser.error('Choose a fresh output; existing results are preserved.')
    source, review, rows, previews = analyze(args.report.resolve(), args.review.resolve())
    summary = summarize(source, rows, args.review_iou)
    result = dict(contract='bytewave.bidirectional-saved-review.v1', completed=True,
                  readyForDeviceValidation=False, sourceReportSHA256=sha(args.report),
                  sourceReviewSHA256=sha(args.review), scriptSHA256=sha(Path(__file__)),
                  sourceVideoSHA256=source['sourceVideoSHA256'],
                  policy=dict(reviewIoUBelow=args.review_iou, sharedAbsenceBoundaryFramesPerEnd=2,
                              seedExcludedFromRuleComparison=True, automaticMaskSelection=False),
                  sourceChecks=dict(binaryMasksVerified=2 * len(rows), overlapRowsRecomputed=len(rows),
                                    previewFilesVerified=len(previews), allSourceControlsPassed=True),
                  limitations=['No model execution, source video decoding, speed measurement or app change.',
                               'Manual later seed comes from forward prediction; direction evidence is correlated.',
                               'Presence logits are not calibrated identity probabilities or mask quality scores.',
                               'Cutoffs only organize review. Similar masks and shared absence can both be wrong.',
                               'Absence boundaries sample shared misses; interior misses may remain unflagged.',
                               'No mask fusion, chosen-mask export, acceptance threshold or held-out accuracy result.'],
                  summary=summary, frames=list(rows.values()))
    template_path = Path(__file__).with_name('bidirectional_review.html')
    template = template_path.read_text()
    if template.count('__REVIEW_DATA__') != 1:
        raise ValueError('Invalid HTML template.')
    result['templateSHA256'] = sha(template_path)
    payload = json.dumps(result, separators=(',', ':'), allow_nan=False).replace('<', '\\u003c')
    # Validate everything before creating output; copy existing JPEG bytes unchanged.
    (output / 'previews').mkdir(parents=True)
    for index, path in previews.items():
        target = output / rows[index]['preview']
        shutil.copyfile(path, target)
        if sha(target) != rows[index]['previewSHA256']:
            raise ValueError(f'Copied preview mismatch: {target}')
    (output / 'index.html').write_text(template.replace('__REVIEW_DATA__', payload))
    (output / 'report.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    (output / 'summary.json').write_text(json.dumps({k: v for k, v in result.items() if k != 'frames'}, indent=2, allow_nan=False) + '\n')
    print(f'Offline review: {output / "index.html"}')
    print(f'Disagreements: {summary["disagreementFrames"]} frames in {len(summary["disagreementIntervals"])} intervals; shared absence boundaries: {len(summary["sharedAbsenceBoundaryFrames"])} frames.')
    print('No inference or automatic mask selection performed.')


if __name__ == '__main__':
    main()
