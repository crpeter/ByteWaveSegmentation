#!/usr/bin/env python3
"""Probe older tracking memory during occlusion without a new selection.

Normal GPU tracking advances unchanged. From its first previously observed
empty-mask frame, side calls use either the original conditioning slots only,
or a frozen copy of the normal input bank immediately before that frame.
Side outputs NEVER enter any history. All calls share current encoder features.
Frozen slots keep their capture-time relative-age semantics; this deliberately
tests stale banks, not an age-correct retention policy or automatic recovery.
No conversion, export, installed model, app, or correctness-gate changes.
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
import owned
import validate_export as v
from state import TemporalState


BANK_NAMES = ('spatial_bank', 'spatial_positions', 'pointer_bank', 'valid_slots')
PROBES = ('conditioningOnly', 'retainedBank')


def clone(values):
    return {name: value.copy() if isinstance(value, np.ndarray) else value
            for name, value in values.items()}


def conditioning_only(bank):
    result = {name: np.zeros_like(value) for name, value in bank.items()}
    for name in ('spatial_bank', 'spatial_positions', 'pointer_bank'):
        result[name][:, 0] = bank[name][:, 0]
    result['valid_slots'][:, [0, 7]] = bank['valid_slots'][:, [0, 7]]
    return result


def summarize(report):
    rows = [row for row in report['frames'] if row.get('rowCompleted')]
    post = [row for row in rows if row['index'] >= report['snapshotInputFrame']]
    lo, hi = report['visualReviewRange']
    mismatches = [row['index'] for row in report['frames']
                  if row.get('baselineReproduced') is False]
    result = {
        'framesReplayed': len(rows), 'framesProbed': len(post),
        'baselineMismatchFrames': mismatches,
        'forkControlAllOutputsExact': report.get('forkControlAllOutputsExact', False),
        'interpretationAllowed': bool(report['completed'] and not mismatches
            and report.get('forkControlAllOutputsExact')
            and report.get('frozenBanksUnchangedAtEnd')),
        'streams': {},
        'qualityMeaning': 'Nonempty masks and agreement are not identity truth. Inspect previews during obstruction and visible return. This is a frozen-bank probe, not a deployed recovery policy.'}
    for label in ('original', 'normalGPU', *PROBES):
        visible = [row for row in post if lo <= row['index'] <= hi]
        result['streams'][label] = {
            'nonemptyFramesAfterSnapshot': [row['index'] for row in post
                if not row['streams'][label]['lowMaskEmpty']],
            'nonemptyFramesInVisualReviewRange': [row['index'] for row in visible
                if not row['streams'][label]['lowMaskEmpty']],
            'visualReviewFramesCompared': len(visible)}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('upstream', 'video', 'phone-report', 'fixture', 'comparison', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--frames', type=int, default=276)
    parser.add_argument('--review-range', type=int, nargs=2, default=(175, 205),
                        help='Previously visually located return interval; used only for reporting.')
    args = parser.parse_args()
    if platform.system() != 'Darwin':
        parser.error('Run locally on the Mac.')
    prior = json.loads(args.comparison.read_text())
    if (prior.get('contract') != 'bytewave.occlusion-comparison.diagnostic.v1'
            or prior.get('completed') is not True or prior.get('previewOnly') is not False
            or prior.get('inferencePerformed') is not True):
        parser.error('Expected completed original occlusion comparison.')
    total, previous_rows = prior['effectiveFrames'], prior['frames']
    if (len(previous_rows) != total
            or [row['index'] for row in previous_rows] != list(range(total))
            or not 2 <= args.frames <= min(total, 600)
            or not 0 <= args.review_range[0] <= args.review_range[1] < args.frames):
        parser.error('Replay and review ranges must fit the completed comparison.')
    empty = [row['index'] for row in previous_rows
             if row['streams']['coremlGPU']['lowMaskEmpty']]
    if not empty or not 1 <= empty[0] < args.review_range[0]:
        parser.error('Expected a nonempty initial selection followed by absence before review.')
    fork = empty[0]
    args.new_video, args.point = True, prior['pointNormalizedTopLeft']
    _, fixture, _ = occlusion.verify_inputs(args)
    identity = {'sourceVideoSHA256': occlusion.sha(args.video),
                'sourceFileBytes': args.video.stat().st_size,
                'phoneReportSHA256': occlusion.sha(args.phone_report),
                'fixtureSHA256': occlusion.sha(args.fixture / 'fixture.json'),
                'modelFiles': fixture['modelFiles'], 'upstream': owned.UPSTREAM_REVISION,
                'checkpointSHA256': owned.CHECKPOINT_SHA256}
    for name, value in {**identity, 'pointNormalizedTopLeft': args.point}.items():
        if prior.get(name) != value:
            raise ValueError(f'Prior comparison identity mismatch: {name}')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'previews').mkdir()
    path = args.output / 'report.json'
    report = {
        'scope': __doc__, 'contract': 'bytewave.occlusion-memory-probe.diagnostic.v1',
        'completed': False, 'readyForDeviceValidation': False, **identity,
        'sourceComparisonSHA256': occlusion.sha(args.comparison),
        'pointNormalizedTopLeft': args.point, 'referenceNumFrames': total,
        'requestedFrames': args.frames, 'snapshotInputFrame': fork,
        'snapshotChoice': 'First empty normal GPU mask in completed prior report. Retrospective diagnostic boundary, not a new online absence detector. Capture input bank before this frame; latest entry may already be partially occluded.',
        'visualReviewRange': list(args.review_range), 'computeUnits': 'CPU_AND_GPU',
        'policy': {'maskIoUMinimum': .95, 'cosineMinimum': .99,
                   'float32Atol': .001, 'float32Rtol': .001},
        'comparisonMeaning': 'Normal GPU/reference must reproduce prior comparison dictionaries exactly. Probe/reference metrics are descriptive under the same gates; changing memory intentionally changes outputs. No promotion or identity verdict follows.',
        'memoryPolicy': 'Normal GPU alone commits to bounded state every decoded frame, including absence. Both side banks are fixed, retain frame-0 conditioning, and receive no new outputs, prompts or reference tensors. Original reference advances its independent history.',
        'timeLimit': 'Normal frame clock and exact source PTS advance. Frozen bank slots retain capture-time relative-age meanings; original source indices and actual ages are reported separately. No claim that this models correct temporal aging or isolates tensor content from slot age.',
        'torchThreads': {'intraOp': 1, 'interOp': 1},
        'versions': {name: importlib.metadata.version(name)
                     for name in ('torch', 'coremltools', 'numpy', 'av', 'Pillow')},
        'frames': [], 'activeFrameIndex': None, 'activeOperation': None}
    decoder = None
    banks = None

    def checkpoint(label):
        report['activeOperation'] = label
        v.write_json(path, report)

    def call(component, inputs, label):
        checkpoint(label)
        result = backend.call(component, clone(inputs))
        tensor_hashes(result, v.NAMES[component])
        return result

    try:
        checkpoint('load reference and exact installed packages')
        model = owned.load_reference(args.upstream)
        backend = occlusion.FixtureBackend(args.fixture)
        state = TemporalState()
        history = {'cond_frame_outputs': {}, 'non_cond_frame_outputs': {}}
        point = [min(float(x) * 1024, 1023) for x in args.point]
        decoder = v.video_frames(args.video, count=args.frames)
        with torch.inference_mode():
            for index, timestamp, image, tensor, digest in decoder:
                previous = previous_rows[index]
                if (digest != previous['imageSHA256'] or timestamp.numerator != previous['ptsNumerator']
                        or timestamp.denominator != previous['ptsDenominator']):
                    raise ValueError(f'Decoded pixels/PTS differ at frame {index}.')
                report['activeFrameIndex'] = index
                print(f'Memory probe: {index + 1}/{args.frames}', flush=True)
                checkpoint('original reference')
                original = v.reference_step(model, torch.from_numpy(tensor), point, index,
                                            history, num_frames=total)
                tensor_hashes(original, v.NAMES['Propagator'])
                features = call('ImageEncoder', {'image': tensor, 'pil_image': image}, 'shared GPU encoder')
                state.validate_time(timestamp)
                if state.next_index != index:
                    raise ValueError('Normal state frame clock changed.')
                token = state.token
                if index == 0:
                    normal = call('Initializer', {**features,
                        'point_coords': np.asarray([[point]], dtype=np.float32),
                        'point_labels': np.ones((1, 1), dtype=np.int32)}, 'original GPU selection')
                    normal.update(call('InitialMemoryEncoder', {**features, **normal}, 'initial memory'))
                else:
                    packed = state.pack(timestamp)
                    normal = call('Propagator', {**features, **packed}, 'normal GPU')
                baseline = v.compare(original, normal, coreml=True)
                row = {'index': index, 'ptsSeconds': float(timestamp),
                       'ptsNumerator': timestamp.numerator, 'ptsDenominator': timestamp.denominator,
                       'imageSHA256': digest,
                       'comparisons': {'normalGPUAgainstOriginal': baseline},
                       'streams': {'original': occlusion.observation(original),
                                   'normalGPU': occlusion.observation(normal)},
                       'baselineReproduced': baseline == previous['comparisons']['coremlGPUAgainstOriginal']
                           and occlusion.observation(original) == previous['streams']['original']
                           and occlusion.observation(normal) == previous['streams']['coremlGPU']}
                report['frames'].append(row)
                if not row['baselineReproduced']:
                    raise ValueError(f'Normal replay does not reproduce prior report at frame {index}; stop before interpreting probes.')
                if index == fork:
                    banks = {'retainedBank': clone(packed), 'conditioningOnly': conditioning_only(packed)}
                    bank_hashes = {label: tensor_hashes(bank, BANK_NAMES) for label, bank in banks.items()}
                    recent = {entry.index for entry in state.recent}
                    pointers = dict(state.pointers)
                    spatial_sources = [state.condition.index] + [index - lag if index - lag in recent else None
                        for lag in range(6, 0, -1)]
                    pointer_sources = [state.condition.index] + [index - lag if index - lag in pointers else None
                        for lag in range(1, 16)]
                    report['snapshot'] = {'inputFrame': index, 'ptsSeconds': float(timestamp),
                        'spatialSourceFrames': spatial_sources, 'pointerSourceFrames': pointer_sources,
                        'banks': bank_hashes, 'normalStateBefore': state.summary()}
                    control = call('Propagator', {**features, **banks['retainedBank']}, 'same-bank fork control')
                    exact = {name: bool(np.array_equal(normal[name], control[name]))
                             for name in v.NAMES['Propagator']}
                    row['sameBankControlOutputsExact'] = exact
                    report['forkControlAllOutputsExact'] = all(exact.values())
                    if not report['forkControlAllOutputsExact']:
                        raise ValueError('Same-input bank control failed; no probe interpretation.')
                outputs = {'normalGPU': normal}
                if index >= fork:
                    row['elapsedFramesSinceSnapshot'] = index - fork
                    row['retainedSpatialActualAges'] = [index - source if source is not None else None
                        for source in spatial_sources]
                    row['retainedPointerActualAges'] = [index - source if source is not None else None
                        for source in pointer_sources]
                    row['probeOrder'] = list(PROBES if index % 2 else reversed(PROBES))
                    for label in row['probeOrder']:
                        outputs[label] = control if index == fork and label == 'retainedBank' else call(
                            'Propagator', {**features, **banks[label]}, label)
                        row['streams'][label] = occlusion.observation(outputs[label])
                        row['comparisons'][label + 'AgainstOriginal'] = v.compare(original, outputs[label], coreml=True)
                    if any(tensor_hashes(bank, BANK_NAMES) != bank_hashes[label] for label, bank in banks.items()):
                        raise ValueError('Frozen input bank was mutated.')
                    relative = f'previews/frame-{index:04d}.png'
                    occlusion.save_preview(args.output / relative, image,
                        {'normal GPU': normal, 'initial memory only': outputs['conditioningOnly'],
                         'retained pre-absence bank': outputs['retainedBank']}, index, float(timestamp))
                    row['preview'] = relative
                # Experimental outputs are deliberately never committed.
                state.commit(normal, timestamp, token)
                row['normalStateAfter'] = state.summary()
                row['rowCompleted'] = True
                v.write_json(path, report)
        report['frozenBanksUnchangedAtEnd'] = banks is not None and all(
            tensor_hashes(bank, BANK_NAMES) == bank_hashes[label] for label, bank in banks.items())
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
