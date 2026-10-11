#!/usr/bin/env python3
"""Test causal, confirmed recovery with feedback using installed GPU packages.

After consecutive absent ordinary predictions, search with original conditioning
memory only. Require consecutive nonempty, sufficiently confident, overlapping
search masks before accepting one. Clear recent banks on acceptance, keep the
original conditioning and actual frame clock, then resume ordinary propagation.
Pending search masks are not displayed or committed. This is a diagnostic
heuristic, not identity proof or an app change. No conversion or export.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import platform

import numpy as np
import torch

import diagnose_occlusion as occlusion
from diagnose_occlusion_inputs import tensor_hashes
from diagnose_occlusion_memory import BANK_NAMES, clone, conditioning_only
import owned
import validate_export as v
from state import TemporalState


def present(output):
    return bool(output['object_score'].item() > 0
                and (output['low_res_mask'] > 0).any()
                and (output['high_res_mask'] > 0).any())


def mask_iou(left, right):
    union = int(np.logical_or(left, right).sum())
    return float(np.logical_and(left, right).sum() / union) if union else 0.0


def blank_preview(output):
    # Display-only suppression. These modified arrays NEVER enter tracking state
    # or raw numerical comparisons. Other values are irrelevant to save_preview.
    return {**output, 'low_res_mask': np.full_like(output['low_res_mask'], -1024),
            'high_res_mask': np.full_like(output['high_res_mask'], -1024)}


def summarize(report):
    rows = [row for row in report['frames'] if row.get('rowCompleted')]
    lo, hi = report['visualReviewRange']
    visible = [row for row in rows if lo <= row['index'] <= hi]
    return {
        'framesReplayed': len(rows), 'events': report['events'],
        'recoveryAcceptanceFrames': [event['frame'] for event in report['events']
                                     if event['type'] == 'recoveryAccepted'],
        'normalNonemptyInVisualReviewRange': [row['index'] for row in visible
            if not row['streams']['normalGPU']['lowMaskEmpty']],
        'policyDisplayedInVisualReviewRange': [row['index'] for row in visible
            if row['displayMaskNonempty']],
        'baselineMismatchFrames': [row['index'] for row in report['frames']
            if row.get('baselineReproduced') is False],
        'searchProbeMismatchFrames': [row['index'] for row in report['frames']
            if row.get('search', {}).get('priorProbeReproduced') is False],
        'heldBackNonemptySearchFrames': [row['index'] for row in rows
            if row.get('search', {}).get('candidatePresent') and not row['recoveryAccepted']],
        'interpretationAllowed': bool(report['completed'] and report.get('conditioningUnchangedAtEnd')),
        'qualityMeaning': 'Acceptance and nonempty masks are heuristic decisions, not ground-truth identity. Inspect first acceptance, subsequent propagation, obstruction, and exit. No generalization or device-readiness claim.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('upstream', 'video', 'phone-report', 'fixture', 'comparison', 'memory-probe', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--frames', type=int, default=276)
    parser.add_argument('--missing-frames', type=int, default=3)
    parser.add_argument('--confirmation-frames', type=int, default=3)
    parser.add_argument('--minimum-predicted-iou', type=float, default=.8)
    parser.add_argument('--minimum-temporal-iou', type=float, default=.5)
    args = parser.parse_args()
    if platform.system() != 'Darwin':
        parser.error('Run this diagnostic locally on the Mac.')
    if (not 1 <= args.missing_frames <= 30 or not 2 <= args.confirmation_frames <= 30
            or not np.isfinite(args.minimum_predicted_iou) or not 0 <= args.minimum_predicted_iou <= 1
            or not np.isfinite(args.minimum_temporal_iou) or not 0 < args.minimum_temporal_iou <= 1):
        parser.error('Invalid diagnostic thresholds.')
    prior = json.loads(args.comparison.read_text())
    probe = json.loads(args.memory_probe.read_text())
    if (prior.get('contract') != 'bytewave.occlusion-comparison.diagnostic.v1'
            or prior.get('completed') is not True or prior.get('previewOnly') is not False
            or prior.get('inferencePerformed') is not True
            or probe.get('contract') != 'bytewave.occlusion-memory-probe.diagnostic.v1'
            or probe.get('completed') is not True
            or probe.get('summary', {}).get('interpretationAllowed') is not True
            or probe.get('sourceComparisonSHA256') != occlusion.sha(args.comparison)):
        parser.error('Expected completed, linked occlusion and controlled memory-probe reports.')
    total = prior['effectiveFrames']
    previous_rows, probe_rows = prior['frames'], probe['frames']
    if (len(previous_rows) != total
            or [row['index'] for row in previous_rows] != list(range(total))
            or [row['index'] for row in probe_rows] != list(range(len(probe_rows)))
            or not all(row.get('rowCompleted') for row in probe_rows)
            or not 2 <= args.frames <= min(total, len(probe_rows), 600)
            or not 0 <= probe['visualReviewRange'][0] <= probe['visualReviewRange'][1] < args.frames):
        parser.error('Incomplete reports or invalid replay range.')
    args.new_video, args.point = True, prior['pointNormalizedTopLeft']
    _, fixture, _ = occlusion.verify_inputs(args)
    identity = {'sourceVideoSHA256': occlusion.sha(args.video),
                'sourceFileBytes': args.video.stat().st_size,
                'phoneReportSHA256': occlusion.sha(args.phone_report),
                'fixtureSHA256': occlusion.sha(args.fixture / 'fixture.json'),
                'modelFiles': fixture['modelFiles'], 'upstream': owned.UPSTREAM_REVISION,
                'checkpointSHA256': owned.CHECKPOINT_SHA256}
    for key, value in {**identity, 'pointNormalizedTopLeft': args.point}.items():
        if prior.get(key) != value or probe.get(key) != value:
            raise ValueError(f'Report identity mismatch: {key}')
    if probe.get('referenceNumFrames') != total:
        raise ValueError('Reference length differs from probe.')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'previews').mkdir()
    path = args.output / 'report.json'
    report = {
        'scope': __doc__, 'contract': 'bytewave.occlusion-recovery.diagnostic.v1',
        'completed': False, 'readyForDeviceValidation': False, **identity,
        'sourceComparisonSHA256': occlusion.sha(args.comparison),
        'sourceMemoryProbeSHA256': occlusion.sha(args.memory_probe),
        'pointNormalizedTopLeft': args.point, 'referenceNumFrames': total,
        'requestedFrames': args.frames, 'computeUnits': 'CPU_AND_GPU',
        'visualReviewRange': probe['visualReviewRange'],
        'recoveryRule': {'missingFrames': args.missing_frames,
            'confirmationFrames': args.confirmation_frames,
            'minimumPredictedIoU': args.minimum_predicted_iou,
            'minimumConsecutiveMaskIoU': args.minimum_temporal_iou,
            'positiveObjectScoreRequired': True,
            'nonemptyLowAndHighMaskRequired': True},
        'ruleLimit': 'Exploratory thresholds informed by this development clip, not calibrated identity confidence. Consecutive-mask IoU has no motion compensation. Stable wrong masks may pass; fast correct subjects may fail. Validate on independent clips before promotion.',
        'policy': {'maskIoUMinimum': .95, 'cosineMinimum': .99,
                   'float32Atol': .001, 'float32Rtol': .001},
        'memoryPolicy': 'Two bounded states retain original conditioning. Normal state always commits its own prediction. Recovery state commits ordinary output while searching, but does not display it. Only upon confirmation clear recent spatial/pointer entries and commit current search output. Pending candidates never enter memory. Absolute frame indices and source PTS advance once per frame, without reset or relabeling.',
        'controlMeaning': 'Baseline comparison and observations must exactly reproduce prior report. Conditioning-only search calls must reproduce prior probe where available. Prior reports and review interval are control/summary data, never recovery-decision inputs. Recovery/reference comparisons are descriptive; original reference loses the worker.',
        'torchThreads': {'intraOp': 1, 'interOp': 1},
        'versions': {name: importlib.metadata.version(name)
                     for name in ('torch', 'coremltools', 'numpy', 'av', 'Pillow')},
        'events': [], 'frames': [], 'activeFrameIndex': None, 'activeOperation': None}
    decoder = None

    def checkpoint(label):
        report['activeOperation'] = label
        v.write_json(path, report)

    def call(component, inputs, label):
        checkpoint(label)
        result = backend.call(component, clone(inputs))
        tensor_hashes(result, v.NAMES[component])
        return result

    try:
        checkpoint('load reference and exact installed GPU packages')
        model = owned.load_reference(args.upstream)
        backend = occlusion.FixtureBackend(args.fixture)
        normal_state, recovery_state = TemporalState(), TemporalState()
        history = {'cond_frame_outputs': {}, 'non_cond_frame_outputs': {}}
        point = [min(float(x) * 1024, 1023) for x in args.point]
        searching, diverged = False, False
        missing, streak = 0, 0
        previous_mask, confirmation_start = None, None
        initial_hashes = None
        decoder = v.video_frames(args.video, count=args.frames)
        with torch.inference_mode():
            for index, timestamp, image, tensor, digest in decoder:
                previous = previous_rows[index]
                if (digest != previous['imageSHA256'] or timestamp.numerator != previous['ptsNumerator']
                        or timestamp.denominator != previous['ptsDenominator']):
                    raise ValueError(f'Decoded pixels/PTS differ at frame {index}.')
                report['activeFrameIndex'] = index
                print(f'Confirmed recovery: {index + 1}/{args.frames}', flush=True)
                checkpoint('original reference')
                original = v.reference_step(model, torch.from_numpy(tensor), point, index,
                                            history, num_frames=total)
                tensor_hashes(original, v.NAMES['Propagator'])
                features = call('ImageEncoder', {'image': tensor, 'pil_image': image}, 'shared GPU encoder')
                for state in (normal_state, recovery_state):
                    state.validate_time(timestamp)
                    if state.next_index != index:
                        raise ValueError('State frame clock differs from decoded frame.')
                normal_token, recovery_token = normal_state.token, recovery_state.token
                if index == 0:
                    normal = call('Initializer', {**features,
                        'point_coords': np.asarray([[point]], dtype=np.float32),
                        'point_labels': np.ones((1, 1), dtype=np.int32)}, 'original GPU selection')
                    normal.update(call('InitialMemoryEncoder', {**features, **normal}, 'initial memory'))
                    ordinary = normal
                else:
                    normal_bank, recovery_bank = normal_state.pack(timestamp), recovery_state.pack(timestamp)
                    if index == 1:
                        initial_hashes = tensor_hashes(conditioning_only(recovery_bank), BANK_NAMES)
                        report['conditioningOnlyBankHashes'] = initial_hashes
                    normal = call('Propagator', {**features, **normal_bank}, 'normal GPU')
                    if not diverged:
                        if tensor_hashes(normal_bank, BANK_NAMES) != tensor_hashes(recovery_bank, BANK_NAMES):
                            raise ValueError('Histories diverged before first recovery acceptance.')
                        ordinary = normal
                    else:
                        ordinary = call('Propagator', {**features, **recovery_bank}, 'recovery ordinary propagation')
                baseline = v.compare(original, normal, coreml=True)
                row = {'index': index, 'ptsSeconds': float(timestamp),
                       'ptsNumerator': timestamp.numerator, 'ptsDenominator': timestamp.denominator,
                       'imageSHA256': digest, 'modeBefore': 'searching' if searching else 'tracking',
                       'ordinarySharedWithBaseline': not diverged,
                       'comparisons': {'normalGPUAgainstOriginal': baseline},
                       'streams': {'original': occlusion.observation(original),
                                   'normalGPU': occlusion.observation(normal),
                                   'recoveryOrdinary': occlusion.observation(ordinary)},
                       'baselineReproduced': baseline == previous['comparisons']['coremlGPUAgainstOriginal']
                           and occlusion.observation(original) == previous['streams']['original']
                           and occlusion.observation(normal) == previous['streams']['coremlGPU'],
                       'recoveryAccepted': False}
                report['frames'].append(row)
                if not row['baselineReproduced']:
                    raise ValueError(f'Baseline no longer reproduces at frame {index}.')
                if index == 0 and not present(normal):
                    raise ValueError('Recovery requires a nonempty original selection.')
                if index > 0 and not searching:
                    missing = 0 if present(ordinary) else missing + 1
                    if missing >= args.missing_frames:
                        searching = True
                        report['events'].append({'type': 'searchStarted', 'frame': index,
                                                 'ptsSeconds': float(timestamp)})
                selected, candidate = ordinary, None
                show = present(ordinary) and not searching
                if searching:
                    search_bank = conditioning_only(recovery_state.pack(timestamp))
                    if tensor_hashes(search_bank, BANK_NAMES) != initial_hashes:
                        raise ValueError('Original conditioning changed during recovery.')
                    candidate = call('Propagator', {**features, **search_bank}, 'conditioning-only search')
                    candidate_present = present(candidate)
                    qualified = candidate_present and float(candidate['best_iou'].item()) >= args.minimum_predicted_iou
                    current_mask = candidate['low_res_mask'] > 0
                    overlap = mask_iou(previous_mask, current_mask) if previous_mask is not None else None
                    if not qualified:
                        streak, previous_mask, confirmation_start = 0, None, None
                    else:
                        if previous_mask is None or overlap < args.minimum_temporal_iou:
                            streak, confirmation_start = 1, index
                        else:
                            streak += 1
                        previous_mask = current_mask.copy()
                    search_check = v.compare(original, candidate, coreml=True)
                    old = probe_rows[index]
                    available = 'conditioningOnly' in old['streams']
                    reproduced = (search_check == old['comparisons']['conditioningOnlyAgainstOriginal']
                                  and occlusion.observation(candidate) == old['streams']['conditioningOnly']) if available else None
                    row['search'] = {'candidatePresent': candidate_present, 'qualifiedThisFrame': qualified,
                        'consecutiveMaskIoU': overlap, 'confirmationStreak': streak,
                        'confirmationStartFrame': confirmation_start,
                        'priorProbeReproduced': reproduced, 'priorProbeAvailable': available}
                    row['streams']['searchCandidate'] = occlusion.observation(candidate)
                    row['comparisons']['searchCandidateAgainstOriginal'] = search_check
                    if available and not reproduced:
                        raise ValueError(f'Conditioning-only search differs from controlled probe at frame {index}.')
                    if streak >= args.confirmation_frames:
                        # Outputs have already passed shape/finite validation. Preserve
                        # conditioning, generation, next_index and last_timestamp.
                        before = recovery_state.summary()
                        recovery_state.recent.clear()
                        recovery_state.pointers.clear()
                        selected, show = candidate, True
                        row['recoveryAccepted'] = True
                        report['events'].append({'type': 'recoveryAccepted', 'frame': index,
                            'ptsSeconds': float(timestamp), 'confirmationStartFrame': confirmation_start,
                            'confirmedFrames': streak, 'stateBeforeClear': before})
                        searching, diverged = False, True
                        missing, streak, previous_mask, confirmation_start = 0, 0, None, None
                # Every frame advances once. Only the accepted CURRENT candidate
                # enters history; no backfilled masks or repeated timestamps.
                normal_state.commit(normal, timestamp, normal_token)
                recovery_state.commit(selected, timestamp, recovery_token)
                row['streams']['committedRecovery'] = occlusion.observation(selected)
                row['comparisons']['committedRecoveryAgainstOriginal'] = v.compare(original, selected, coreml=True)
                row['displayMaskNonempty'] = bool(show)
                row['modeAfter'] = 'searching' if searching else 'tracking'
                row['missingStreak'] = missing
                row['normalStateAfter'] = normal_state.summary()
                row['recoveryStateAfter'] = recovery_state.summary()
                # Hash conditioning directly after commit; pack() requires a future PTS.
                anchors = {'memory_features': recovery_state.condition.features,
                           'memory_positions': recovery_state.condition.positions,
                           'object_pointer': recovery_state.condition_pointer}
                hashes = tensor_hashes(anchors, tuple(anchors))
                if index == 0:
                    anchor_hashes = hashes
                elif hashes != anchor_hashes:
                    raise ValueError('Original conditioning tensors changed.')
                if index == 0 or candidate is not None or diverged:
                    relative = f'previews/frame-{index:04d}.png'
                    occlusion.save_preview(args.output / relative, image,
                        {'normal GPU': normal,
                         'search candidate (unfiltered)' if candidate is not None else 'search not queried': candidate if candidate is not None else blank_preview(normal),
                         'confirmed recovery output': selected if show else blank_preview(selected)}, index, float(timestamp))
                    row['preview'] = relative
                row['rowCompleted'] = True
                v.write_json(path, report)
        report['conditioningUnchangedAtEnd'] = initial_hashes is not None
        report['completed'] = len(report['frames']) == args.frames
        report['activeOperation'] = None
    except Exception as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        if decoder is not None:
            decoder.close()
        report['summary'] = summarize(report)
        v.write_json(path, report)
        v.write_json(args.output / 'summary.json', {k: value for k, value in report.items() if k != 'frames'})
        print(f'Report: {path}', flush=True)


if __name__ == '__main__':
    main()
