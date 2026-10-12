#!/usr/bin/env python3
"""Test appearance-checked recovery as a separate, lower-priority tracker.

Normal tracking always advances its own unchanged history and wins whenever
present. During absence, a separate backup can accept a conditioning-only
candidate after three consistent, appearance-matched frames. Backup predictions
must continue passing the appearance checks; normal return discards the backup.
This is a development experiment on the already reviewed worker and dog clips,
not a calibrated identity method, production change, model export or benchmark.
"""
from __future__ import annotations

import argparse
import copy
import importlib.metadata
import json
from pathlib import Path
import platform

import numpy as np
import torch

import diagnose_occlusion as occlusion
from diagnose_occlusion_identity import describe, metrics, public_descriptor
from diagnose_occlusion_inputs import tensor_hashes
from diagnose_occlusion_memory import BANK_NAMES, clone, conditioning_only
from diagnose_occlusion_recovery import blank_preview, mask_iou, present
import owned
import validate_export as v
from state import TemporalState


# Fixed BEFORE the full replay, informed by both development clips' sparse
# measurements. They are not held-out validation or universal identity cutoffs.
RULE = {
    'version': 'baseline-first-appearance-backup.v1',
    'missingFrames': 3,
    'confirmationFrames': 3,
    'minimumPredictedIoU': 0.8,
    'minimumConsecutiveMaskIoU': 0.5,
    'minimumPooledEncoderCosine': 0.8,
    'minimumRGBHistogramIntersection': 0.5,
    'sameAnchorMustPassBothAppearanceChecks': True,
    'commonAnchorRequiredAcrossConfirmationStreak': True,
    'continuationMustMatchAnAcceptanceAnchor': True,
    'pointerCosineUsedAsGate': False,
    'normalPresentAlwaysWins': True}


def assess(features, output, image, anchors):
    value = describe(features, output, image)
    comparisons = {str(index): metrics(value, anchor) for index, anchor in anchors.items()}
    matches = {index for index in anchors if comparisons[str(index)] is not None
        and comparisons[str(index)]['pooledEncoderCosine'] is not None
        and comparisons[str(index)]['pooledEncoderCosine'] >= RULE['minimumPooledEncoderCosine']
        and comparisons[str(index)]['rgbHistogramIntersection'] >= RULE['minimumRGBHistogramIntersection']}
    qualified = present(output) and output['best_iou'].item() >= RULE['minimumPredictedIoU']
    return {'present': present(output), 'predictedIoU': float(output['best_iou'].item()),
            'againstAnchors': comparisons, 'appearanceMatchingAnchors': sorted(matches),
            'qualifyingAnchors': sorted(matches) if qualified else []}


def condition_hashes(state):
    values = {'memory_features': state.condition.features,
              'memory_positions': state.condition.positions,
              'object_pointer': state.condition_pointer}
    return tensor_hashes(values, tuple(values))


def intervals(rows, predicate):
    result = []
    for row in rows:
        if not predicate(row):
            continue
        if result and row['index'] == result[-1]['lastFrame'] + 1:
            result[-1].update(lastFrame=row['index'], lastPTSSeconds=row['ptsSeconds'])
            result[-1]['frames'] += 1
        else:
            result.append({'firstFrame': row['index'], 'lastFrame': row['index'],
                           'firstPTSSeconds': row['ptsSeconds'], 'lastPTSSeconds': row['ptsSeconds'], 'frames': 1})
    return result


def summarize(report):
    rows = [row for row in report['frames'] if row.get('rowCompleted')]
    return {
        'framesReplayed': len(rows), 'events': report['events'],
        'baselineMismatchFrames': [row['index'] for row in report['frames'] if row.get('baselineReproduced') is False],
        'probeMismatchFrames': [row['index'] for row in report['frames'] if row.get('probeReproduced') is False],
        'normalPresentFrames': sum(row['normalPresent'] for row in rows),
        'normalPresentOutputsPreserved': all(row['normalOutputPreserved'] for row in rows),
        'backupDisplayIntervals': intervals(rows, lambda row: row['displaySource'] == 'backup'),
        'emptyDisplayIntervals': intervals(rows, lambda row: row['displaySource'] == 'empty'),
        'rejectedNonemptySearchFrames': [row['index'] for row in rows
            if row.get('search', {}).get('present') and not row['recoveryAccepted']],
        'interpretationAllowed': bool(report['completed'] and report.get('conditioningUnchangedAtEnd')
            and report.get('anchorDescriptorsReproduced')
            and all(row.get('baselineReproduced') is True for row in report['frames'])
            and all(row.get('probeReproduced') is True for row in report['frames'] if 'probeReproduced' in row)
            and all(row['normalOutputPreserved'] for row in rows)),
        'qualityMeaning': 'Controls and normal-output preservation are not identity validation. Inspect every backup interval and rejection around visible return. Thresholds were informed by both clips; an independent clip is still required.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('upstream', 'video', 'phone-report', 'fixture', 'comparison', 'memory-probe', 'appearance-report', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if platform.system() != 'Darwin':
        parser.error('Run locally on the Mac.')
    prior = json.loads(args.comparison.read_text())
    probe = json.loads(args.memory_probe.read_text())
    appearance = json.loads(args.appearance_report.read_text())
    if (prior.get('contract') != 'bytewave.occlusion-comparison.diagnostic.v1'
            or prior.get('completed') is not True or prior.get('previewOnly') is not False
            or prior.get('inferencePerformed') is not True
            or probe.get('contract') != 'bytewave.occlusion-memory-probe.diagnostic.v1'
            or probe.get('completed') is not True
            or probe.get('summary', {}).get('interpretationAllowed') is not True
            or probe.get('sourceComparisonSHA256') != occlusion.sha(args.comparison)
            or appearance.get('contract') != 'bytewave.occlusion-identity-signals.diagnostic.v1'
            or appearance.get('completed') is not True
            or appearance.get('summary', {}).get('interpretationAllowed') is not True
            or appearance.get('sourceComparisonSHA256') != occlusion.sha(args.comparison)
            or appearance.get('sourceMemoryProbeSHA256') != occlusion.sha(args.memory_probe)):
        parser.error('Expected completed, controlled and linked comparison, memory and appearance reports.')
    total = prior['effectiveFrames']
    previous_rows, probe_rows = prior['frames'], probe['frames']
    anchor_frames = appearance['anchorFrames']
    if (not 2 <= total <= 600 or len(previous_rows) != total or len(probe_rows) < total
            or [row['index'] for row in previous_rows] != list(range(total))
            or [row['index'] for row in probe_rows] != list(range(len(probe_rows)))
            or not all(row.get('rowCompleted') for row in probe_rows)
            or not 1 <= len(anchor_frames) <= 8 or anchor_frames != sorted(set(anchor_frames))
            or not 0 <= anchor_frames[0] <= anchor_frames[-1] < probe['snapshotInputFrame'] < total
            or probe.get('referenceNumFrames') != total or appearance.get('referenceNumFrames') != total
            or any(str(index) not in appearance['anchors'] for index in anchor_frames)
            or appearance.get('diagnosticScriptSHA256') != occlusion.sha(Path(__file__).with_name('diagnose_occlusion_identity.py'))):
        parser.error('Incomplete full-clip controls, invalid earlier references, or changed appearance diagnostic.')
    args.new_video, args.point = True, prior['pointNormalizedTopLeft']
    _, fixture, _ = occlusion.verify_inputs(args)
    identity = {'sourceVideoSHA256': occlusion.sha(args.video),
                'sourceFileBytes': args.video.stat().st_size,
                'phoneReportSHA256': occlusion.sha(args.phone_report),
                'fixtureSHA256': occlusion.sha(args.fixture / 'fixture.json'),
                'modelFiles': fixture['modelFiles'], 'upstream': owned.UPSTREAM_REVISION,
                'checkpointSHA256': owned.CHECKPOINT_SHA256}
    for key, value in {**identity, 'pointNormalizedTopLeft': args.point}.items():
        if any(source.get(key) != value for source in (prior, probe, appearance)):
            raise ValueError(f'Report identity mismatch: {key}')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'previews').mkdir()
    path = args.output / 'report.json'
    report = {
        'scope': __doc__, 'contract': 'bytewave.occlusion-backup.diagnostic.v1',
        'completed': False, 'readyForDeviceValidation': False, **identity,
        'diagnosticScriptSHA256': occlusion.sha(Path(__file__)),
        'sourceComparisonSHA256': occlusion.sha(args.comparison),
        'sourceMemoryProbeSHA256': occlusion.sha(args.memory_probe),
        'sourceAppearanceReportSHA256': occlusion.sha(args.appearance_report),
        'pointNormalizedTopLeft': args.point, 'referenceNumFrames': total,
        'framesToReplay': total, 'anchorFrames': anchor_frames, 'computeUnits': 'CPU_AND_GPU',
        'rule': RULE,
        'ruleLimit': 'Exploratory thresholds chosen after sparse worker AND dog appearance review. These clips are development evidence. Stable wrong regions can still pass; small correct subjects can fail. Baseline priority protects only baseline-present frames and does not prove identity.',
        'referencePolicy': 'Recompute fixed manually selected earlier normal descriptors and require exact reproduction of reviewed appearance report. No future image features or later updates enter references. Disable backup until all references have been observed.',
        'memoryPolicy': 'Normal state commits normal output every frame, never backup. A confirmed candidate gets a deep copy of the current pre-commit normal clock/conditioning with recent banks cleared, then current candidate committed at actual PTS. Backup propagates only while normal is absent; failed backup is discarded. Normal return immediately discards backup and pending streak.',
        'continuationPolicy': 'Each backup propagation must be present, predictedIoU>=.8, and meet encoder>=.8 AND colour>=.5 against one of its acceptance anchors. Failure discards it before display/commit and restarts search on that frame. No overlap gate on continuation.',
        'controlMeaning': 'Every baseline reproduces prior observations/comparison. Every search reproduces archived conditioning-only probe. Recomputed anchors reproduce appearance vectors exactly. Search candidates never alter normal history. Old reports are controls/reference definitions, not current-frame identity labels.',
        'policy': {'maskIoUMinimum': .95, 'cosineMinimum': .99, 'float32Atol': .001, 'float32Rtol': .001},
        'versions': {name: importlib.metadata.version(name) for name in ('torch', 'coremltools', 'numpy', 'av', 'Pillow')},
        'torchThreads': {'intraOp': 1, 'interOp': 1},
        'anchors': {}, 'events': [], 'frames': [], 'activeFrameIndex': None, 'activeOperation': None}
    decoder = None

    def checkpoint(label):
        report['activeOperation'] = label
        v.write_json(path, report)

    def call(component, inputs, label):
        checkpoint(label)
        result = backend.call(component, clone(inputs))
        tensor_hashes(result, v.NAMES[component])
        return result

    def event(kind, index, timestamp, **extra):
        report['events'].append({'type': kind, 'frame': index, 'ptsSeconds': float(timestamp), **extra})

    try:
        checkpoint('load reference and exact installed GPU models')
        model = owned.load_reference(args.upstream)
        backend = occlusion.FixtureBackend(args.fixture)
        normal_state, backup_state = TemporalState(), None
        anchors, locked_anchors = {}, set()
        history = {'cond_frame_outputs': {}, 'non_cond_frame_outputs': {}}
        point = [min(float(x) * 1024, 1023) for x in args.point]
        missing, streak = 0, 0
        previous_mask, confirmation_start = None, None
        streak_anchors = set()
        searching = False
        initial_hashes, original_condition = None, None
        previous_display = None
        decoder = v.video_frames(args.video, count=total)
        with torch.inference_mode():
            for index, timestamp, image, tensor, digest in decoder:
                old = previous_rows[index]
                if (digest != old['imageSHA256'] or timestamp.numerator != old['ptsNumerator']
                        or timestamp.denominator != old['ptsDenominator']):
                    raise ValueError(f'Decoded pixels/PTS mismatch at frame {index}')
                report['activeFrameIndex'] = index
                print(f'Appearance backup: {index + 1}/{total}', flush=True)
                checkpoint('original reference')
                original = v.reference_step(model, torch.from_numpy(tensor), point, index, history, num_frames=total)
                tensor_hashes(original, v.NAMES['Propagator'])
                features = call('ImageEncoder', {'image': tensor, 'pil_image': image}, 'shared GPU encoder')
                normal_state.validate_time(timestamp)
                if normal_state.next_index != index:
                    raise ValueError('Normal frame clock mismatch')
                token = normal_state.token
                if index == 0:
                    normal = call('Initializer', {**features,
                        'point_coords': np.asarray([[point]], dtype=np.float32),
                        'point_labels': np.ones((1, 1), dtype=np.int32)}, 'original selection')
                    normal.update(call('InitialMemoryEncoder', {**features, **normal}, 'initial memory'))
                    if not present(normal):
                        raise ValueError('Original selection is absent')
                else:
                    packed = normal_state.pack(timestamp)
                    packed_hashes = tensor_hashes(packed, BANK_NAMES)
                    normal = call('Propagator', {**features, **packed}, 'unchanged normal GPU')
                    search_bank = conditioning_only(packed)
                    current_hashes = tensor_hashes(search_bank, BANK_NAMES)
                    if initial_hashes is None:
                        initial_hashes = current_hashes
                        report['conditioningOnlyBankHashes'] = current_hashes
                    elif current_hashes != initial_hashes:
                        raise ValueError('Normal conditioning changed')
                baseline = v.compare(original, normal, coreml=True)
                normal_present = present(normal)
                row = {'index': index, 'ptsSeconds': float(timestamp),
                       'ptsNumerator': timestamp.numerator, 'ptsDenominator': timestamp.denominator,
                       'imageSHA256': digest, 'normalPresent': normal_present,
                       'streams': {'normalGPU': occlusion.observation(normal)},
                       'comparisons': {'normalGPUAgainstOriginal': baseline},
                       'baselineReproduced': baseline == old['comparisons']['coremlGPUAgainstOriginal']
                           and occlusion.observation(normal) == old['streams']['coremlGPU']
                           and occlusion.observation(original) == old['streams']['original'],
                       'recoveryAccepted': False}
                report['frames'].append(row)
                if not row['baselineReproduced']:
                    raise ValueError(f'Baseline control failed at {index}')
                if index in anchor_frames:
                    descriptor = describe(features, normal, image)
                    archived = appearance['anchors'][str(index)]
                    reproduced = (descriptor is not None and public_descriptor(descriptor) == archived['descriptor']
                        and digest == archived['imageSHA256'] and float(timestamp) == archived['ptsSeconds']
                        and occlusion.observation(normal) == archived['observation'])
                    if not reproduced:
                        raise ValueError(f'Appearance reference mismatch at {index}')
                    anchors[index] = descriptor
                    report['anchors'][str(index)] = {'reproduced': True, 'imageSHA256': digest}
                selected = normal if normal_present else None
                display_source = 'normal' if normal_present else 'empty'
                candidate, backup_output = None, None
                if normal_present:
                    if backup_state is not None or searching:
                        event('normalResumed', index, timestamp, discardedBackup=backup_state is not None)
                    backup_state, searching = None, False
                    locked_anchors, streak_anchors = set(), set()
                    missing, streak, previous_mask, confirmation_start = 0, 0, None, None
                else:
                    missing += 1
                    if backup_state is not None:
                        if backup_state.next_index != index:
                            raise ValueError('Backup frame clock mismatch')
                        backup_output = call('Propagator', {**features, **backup_state.pack(timestamp)}, 'backup propagation')
                        row['streams']['backupOrdinary'] = occlusion.observation(backup_output)
                        row['continuation'] = assess(features, backup_output, image, anchors)
                        matches = set(row['continuation']['qualifyingAnchors']) & locked_anchors
                        row['continuation']['passed'] = bool(matches)
                        if matches:
                            backup_state.commit(backup_output, timestamp, backup_state.token)
                            selected, display_source = backup_output, 'backup'
                        else:
                            event('backupRejected', index, timestamp)
                            backup_state, searching = None, False
                            locked_anchors, streak_anchors = set(), set()
                            streak, previous_mask, confirmation_start = 0, None, None
                    if backup_state is None and missing >= RULE['missingFrames'] and len(anchors) == len(anchor_frames):
                        if not searching:
                            event('searchStarted', index, timestamp)
                            searching = True
                        candidate = call('Propagator', {**features, **search_bank}, 'conditioning-only search')
                        check = v.compare(original, candidate, coreml=True)
                        probe_row = probe_rows[index]
                        row['probeReproduced'] = (check == probe_row.get('comparisons', {}).get('conditioningOnlyAgainstOriginal')
                            and occlusion.observation(candidate) == probe_row.get('streams', {}).get('conditioningOnly'))
                        if not row['probeReproduced']:
                            raise ValueError(f'Archived search control missing or mismatched at {index}')
                        row['streams']['searchCandidate'] = occlusion.observation(candidate)
                        row['comparisons']['searchCandidateAgainstOriginal'] = check
                        row['search'] = assess(features, candidate, image, anchors)
                        matches = set(row['search']['qualifyingAnchors'])
                        mask = candidate['low_res_mask'] > 0
                        overlap = mask_iou(previous_mask, mask) if previous_mask is not None else None
                        common = matches & streak_anchors
                        if not matches:
                            streak, previous_mask, confirmation_start, streak_anchors = 0, None, None, set()
                        else:
                            if previous_mask is None or overlap < RULE['minimumConsecutiveMaskIoU'] or not common:
                                streak, confirmation_start, streak_anchors = 1, index, matches
                            else:
                                streak += 1
                                streak_anchors = common
                            previous_mask = mask.copy()
                        row['search'].update(consecutiveMaskIoU=overlap, confirmationStreak=streak,
                            confirmationStartFrame=confirmation_start, commonAnchors=sorted(streak_anchors))
                        if streak >= RULE['confirmationFrames']:
                            # Deep copy BEFORE normal commit preserves actual clock/PTS.
                            # Original condition stays; no absent recent entries or
                            # pending candidates are inserted into backup history.
                            backup_state = copy.deepcopy(normal_state)
                            backup_state.recent.clear()
                            backup_state.pointers.clear()
                            backup_state.commit(candidate, timestamp, backup_state.token)
                            locked_anchors = set(streak_anchors)
                            selected, display_source = candidate, 'backup'
                            row['recoveryAccepted'] = True
                            event('backupAccepted', index, timestamp, confirmationStartFrame=confirmation_start,
                                  matchingAnchors=sorted(locked_anchors))
                            searching = False
                            streak, previous_mask, confirmation_start, streak_anchors = 0, None, None, set()
                if index > 0:
                    if tensor_hashes(normal_state.pack(timestamp), BANK_NAMES) != packed_hashes:
                        raise ValueError('Backup modified normal state')
                    if tensor_hashes(search_bank, BANK_NAMES) != initial_hashes:
                        raise ValueError('Search conditioning mutated')
                normal_state.commit(normal, timestamp, token)
                hashes = condition_hashes(normal_state)
                if original_condition is None:
                    original_condition = hashes
                if hashes != original_condition or (backup_state is not None and condition_hashes(backup_state) != original_condition):
                    raise ValueError('Original condition tensors changed')
                row['normalOutputPreserved'] = not normal_present or (display_source == 'normal'
                    and all(np.array_equal(normal[name], selected[name]) for name in v.NAMES['Propagator']))
                if not row['normalOutputPreserved']:
                    raise ValueError('Present normal output was replaced')
                row['displaySource'] = display_source
                row['displayMaskNonempty'] = selected is not None
                row['missingStreak'] = missing
                row['normalStateAfter'] = normal_state.summary()
                row['backupStateAfter'] = backup_state.summary() if backup_state is not None else None
                row['streams']['displayed'] = occlusion.observation(selected) if selected is not None else None
                if selected is not None:
                    row['comparisons']['displayedAgainstOriginal'] = v.compare(original, selected, coreml=True)
                # Save all backup/search frames, transitions and earlier references.
                if candidate is not None or backup_output is not None or display_source != previous_display or index in anchor_frames:
                    relative = f'previews/frame-{index:04d}.png'
                    panels = {'unchanged normal GPU': normal}
                    if backup_output is not None:
                        panels['raw backup propagation'] = backup_output
                    if candidate is not None:
                        panels['raw search'] = candidate
                    if candidate is None and backup_output is None:
                        panels['no backup call'] = blank_preview(normal)
                    panels['display: ' + display_source] = selected if selected is not None else blank_preview(normal)
                    occlusion.save_preview(args.output / relative, image, panels, index, float(timestamp))
                    row['preview'] = relative
                previous_display = display_source
                row['rowCompleted'] = True
                v.write_json(path, report)
        report['conditioningUnchangedAtEnd'] = initial_hashes is not None
        report['anchorDescriptorsReproduced'] = len(anchors) == len(anchor_frames)
        report['completed'] = len(report['frames']) == total
        report['activeOperation'] = None
    except Exception as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        if decoder is not None:
            decoder.close()
        report['summary'] = summarize(report)
        v.write_json(path, report)
        v.write_json(args.output / 'summary.json', {key: value for key, value in report.items() if key != 'frames'})
        print(f'Report: {path}', flush=True)


if __name__ == '__main__':
    main()
