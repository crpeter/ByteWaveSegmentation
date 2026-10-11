#!/usr/bin/env python3
"""Inspect candidate-mask scores without feeding diagnostic predictions into memory.

Original PyTorch and installed GPU retain independent histories. GPU side calls
use identical installed-GPU features and history: unchanged reconversion and a
copy exposing existing score/argmax outputs. Both must reproduce every installed
output exactly before tapped selection is considered interpretable. Numerical
controls and failures are recorded; no runtime, selection or gate changes.
No speed, phone parity, ground-truth identity or recovery claim.
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
from mask_choice_taps import prepare
import owned
import validate_export as v
from state import TemporalState


def choice(scores, selected):
    scores = np.asarray(scores, dtype=np.float32)
    index_array = np.asarray(selected)
    if (scores.shape != (1, 3) or not np.isfinite(scores).all()
            or index_array.size != 1 or not np.isfinite(index_array).all()):
        raise ValueError('Expected finite three-candidate scores and one index.')
    index_value = float(index_array.item())
    if index_value != int(index_value) or not 0 <= index_value < 3:
        raise ValueError('Invalid selected index.')
    index = int(index_value)
    order = np.sort(scores[0])
    return {'scores': scores[0].tolist(), 'selectedIndex': index,
            'topTwoMargin': float(order[-1] - order[-2]),
            'exactTopTie': bool(order[-1] == order[-2]),
            'indexSelectsMaximum': bool(scores[0, index] == order[-1]),
            'numpyFirstArgmax': int(np.argmax(scores[0]))}


def control(original, actual):
    metrics = v.compare(original, actual, coreml=True)
    exact = {name: bool(np.array_equal(original[name], actual[name]))
             for name in v.NAMES['Propagator']}
    return {'allOutputsExact': all(exact.values()), 'outputsExact': exact,
            'existingGateComparison': metrics}


def summary(report):
    rows = [row for row in report['frames'] if 'taps' in row]
    usable = [row for row in rows if row['taps']['interpretationAllowed']]
    return {'framesReplayed': len(report['frames']), 'framesInspected': len(rows),
            'interpretableFrames': [row['index'] for row in usable],
            'controlFailureFrames': [row['index'] for row in rows if not row['taps']['interpretationAllowed']],
            'differentChoiceFrames': [row['index'] for row in usable
                                     if row['taps']['choiceDiffersFromOriginal']],
            'differentChoiceAndParityFailureFrames': [row['index'] for row in usable
                if row['taps']['choiceDiffersFromOriginal'] and not row['gpuAgainstOriginal']['passed']],
            'sameChoiceButParityFailureFrames': [row['index'] for row in usable
                if not row['taps']['choiceDiffersFromOriginal'] and not row['gpuAgainstOriginal']['passed']],
            'originalMinimumScoreMargin': min((row['originalChoice']['topTwoMargin'] for row in rows), default=None),
            'interpretableGPUMinimumScoreMargin': min((row['taps']['choice']['topTwoMargin'] for row in usable), default=None),
            'normalGPUPriorComparisonMismatchFrames': [row['index'] for row in report['frames']
                                                       if not row['priorGPUComparisonExact']],
            'originalPriorObservationMismatchFrames': [row['index'] for row in report['frames']
                                                       if not row['priorOriginalObservationExact']]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('upstream', 'video', 'phone-report', 'fixture', 'comparison', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--first-frame', type=int, default=24)
    parser.add_argument('--last-frame', type=int, default=48)
    args = parser.parse_args()
    if platform.system() != 'Darwin':
        parser.error('Run locally on the Mac.')
    prior = json.loads(args.comparison.read_text())
    if (prior.get('contract') != 'bytewave.occlusion-comparison.diagnostic.v1'
            or prior.get('completed') is not True or prior.get('previewOnly') is not False
            or prior.get('inferencePerformed') is not True):
        parser.error('Expected completed original occlusion comparison report.')
    total = prior['effectiveFrames']
    prior_rows = prior['frames']
    if (len(prior_rows) != total or [row['index'] for row in prior_rows] != list(range(total))
            or not 1 <= args.first_frame <= args.last_frame < min(total, 600)):
        parser.error('Invalid inspection range or prior report.')
    args.new_video, args.point = True, prior['pointNormalizedTopLeft']
    _, fixture, _ = occlusion.verify_inputs(args)
    identity = {'sourceVideoSHA256': occlusion.sha(args.video),
                'sourceFileBytes': args.video.stat().st_size,
                'phoneReportSHA256': occlusion.sha(args.phone_report),
                'fixtureSHA256': occlusion.sha(args.fixture / 'fixture.json'),
                'modelFiles': fixture['modelFiles'], 'upstream': owned.UPSTREAM_REVISION,
                'checkpointSHA256': owned.CHECKPOINT_SHA256}
    for name, value in identity.items():
        if prior.get(name) != value:
            raise ValueError(f'Source comparison mismatch: {name}')
    import coremltools as ct
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    args.output.mkdir(parents=True, exist_ok=False)
    path = args.output / 'report.json'
    selected = sorted(set(range(args.first_frame, args.last_frame + 1)) | {1, 5})
    selected = [i for i in selected if i <= args.last_frame]
    report = {'scope': __doc__, 'contract': 'bytewave.mask-choice.diagnostic.v1',
              'completed': False, 'readyForDeviceValidation': False, **identity,
              'sourceComparisonSHA256': occlusion.sha(args.comparison),
              'pointNormalizedTopLeft': args.point, 'referenceNumFrames': total,
              'torchThreads': {'intraOp': 1, 'interOp': 1},
              'computeUnits': 'CPU_AND_GPU',
              'policy': {'maskIoUMinimum': .95, 'cosineMinimum': .99,
                         'float32Atol': .001, 'float32Rtol': .001,
                         'tapInterpretationRequiresExactOriginalOutputs': True},
              'selectedFrames': selected, 'frames': [], 'variants': {},
              'versions': {name: importlib.metadata.version(name)
                           for name in ('torch', 'coremltools', 'numpy', 'av', 'Pillow')},
              'interpretation': 'Different candidate choices may explain abrupt pointer changes but do not prove the source of earlier drift. Histories differ intentionally. Exact side-call controls are required; matching final outputs still cannot prove identical internal execution. Tied scores are reported explicitly.',
              'memoryOwner': 'Only unmodified installed GPU outputs advance GPU state; original PyTorch has its own independent history.',
              'activeFrameIndex': None, 'activeOperation': None}
    decoder, hook = None, None
    captured = []

    def checkpoint(label):
        report['activeOperation'] = label
        v.write_json(path, report)

    def capture(_module, _inputs, output):
        # Read-only hook: no replacement return, output mutation or extra head run.
        scores = output[1].detach().cpu().numpy().copy()
        captured.append((scores, int(output[2].shape[1])))

    def gpu_call(component, inputs):
        checkpoint('installed GPU ' + component)
        return backend.call(component, {k: x.copy() if isinstance(x, np.ndarray) else x
                                        for k, x in inputs.items()})

    try:
        checkpoint('prepare unchanged and tapped propagators')
        package = args.fixture / 'models/BWTemporalPropagator.mlpackage'
        side_models = {}
        for label, expose in (('unchanged', False), ('tapped', True)):
            checkpoint('prepare ' + label)
            destination = args.output / (label + '.mlpackage')
            report['variants'][label] = prepare(package, destination, expose)
            checkpoint('load ' + label)
            side_models[label] = ct.models.MLModel(str(destination), compute_units=ct.ComputeUnit.CPU_AND_GPU)
        mapping = report['variants']['tapped']['mapping']
        checkpoint('load original reference and installed packages')
        model = owned.load_reference(args.upstream)
        hook = model.sam_mask_decoder.register_forward_hook(capture)
        backend = occlusion.FixtureBackend(args.fixture)
        state, history = TemporalState(), {'cond_frame_outputs': {}, 'non_cond_frame_outputs': {}}
        point = [min(float(x) * 1024, 1023) for x in args.point]
        decoder = v.video_frames(args.video, count=args.last_frame + 1)
        with torch.inference_mode():
            for index, timestamp, image, tensor, digest in decoder:
                previous = prior_rows[index]
                if (digest != previous['imageSHA256'] or timestamp.numerator != previous['ptsNumerator']
                        or timestamp.denominator != previous['ptsDenominator']):
                    raise ValueError(f'Prior decoded pixels/PTS differ at frame {index}.')
                report['activeFrameIndex'] = index
                print(f'Mask-choice replay: {index + 1}/{args.last_frame + 1}', flush=True)
                checkpoint('original reference')
                captured.clear()
                original = v.reference_step(model, torch.from_numpy(tensor), point, index, history,
                                            num_frames=total)
                if (len(captured) != 2 or not np.array_equal(captured[0][0], captured[1][0])
                        or captured[0][1] != 3 or captured[1][1] != 3):
                    raise ValueError('Expected two identical three-candidate reference head calls/tokens.')
                original_choice = choice(captured[0][0], np.argmax(captured[0][0], axis=-1))
                features = gpu_call('ImageEncoder', {'image': tensor, 'pil_image': image})
                row = {'index': index, 'ptsSeconds': float(timestamp), 'imageSHA256': digest,
                       'originalChoice': original_choice, 'referencePointerTokenCount': captured[0][1]}
                if index == 0:
                    actual = gpu_call('Initializer', {**features,
                        'point_coords': np.asarray([[point]], dtype=np.float32),
                        'point_labels': np.ones((1, 1), dtype=np.int32)})
                    actual.update(gpu_call('InitialMemoryEncoder', {**features, **actual}))
                else:
                    inputs = {**features, **state.pack(timestamp)}
                    actual = gpu_call('Propagator', inputs)
                    if index in selected:
                        row['inputTensors'] = tensor_hashes(inputs, v.INPUTS['Propagator'])
                        sides = {}
                        for label in ('unchanged', 'tapped'):
                            checkpoint(label + ' side call')
                            sides[label] = side_models[label].predict({name: np.array(inputs[name], dtype=np.float32, copy=True)
                                                                      for name in v.INPUTS['Propagator']})
                        controls = {label: control(actual, result) for label, result in sides.items()}
                        tapped_choice = choice(sides['tapped'][mapping['scores']], sides['tapped'][mapping['index']])
                        best_matches = bool(np.array_equal(np.max(sides['tapped'][mapping['scores']], axis=-1),
                                                           sides['tapped']['best_iou']))
                        allowed = all(check['allOutputsExact'] for check in controls.values()) and tapped_choice['indexSelectsMaximum'] and best_matches
                        row['taps'] = {'controls': controls, 'choice': tapped_choice,
                                       'bestIoUMatchesScores': best_matches, 'interpretationAllowed': allowed,
                                       'choiceDiffersFromOriginal': tapped_choice['selectedIndex'] != original_choice['selectedIndex']}
                row['gpuAgainstOriginal'] = v.compare(original, actual, coreml=True)
                row['priorGPUComparisonExact'] = row['gpuAgainstOriginal'] == previous['comparisons']['coremlGPUAgainstOriginal']
                row['priorOriginalObservationExact'] = occlusion.observation(original) == previous['streams']['original']
                state.commit(actual, timestamp, state.token)
                row['state'] = state.summary()
                report['frames'].append(row)
                v.write_json(path, report)
        report['completed'] = len(report['frames']) == args.last_frame + 1
        report['activeOperation'] = None
    except Exception as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        if hook is not None:
            hook.remove()
        if decoder is not None:
            decoder.close()
        report['summary'] = summary(report)
        v.write_json(path, report)
        v.write_json(args.output / 'summary.json', {k: x for k, x in report.items() if k != 'frames'})
        print(f'Report: {path}', flush=True)


if __name__ == '__main__':
    main()
