#!/usr/bin/env python3
"""Measure the downstream effect of changing one GPU candidate-mask choice.

Replay identical installed GPU outputs into two separate states until the fork.
At the fork only, the intervention uses the original model's candidate index.
Require exact tapped and same-index override controls before committing it.
All subsequent predictions use the unmodified installed GPU propagator, with
independent histories and shared encoder features. Existing accuracy gates stay
unchanged. Default 49-frame prefix precedes occlusion: no recovery, identity,
speed, device readiness or deployable selection-policy claim.
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
from diagnose_mask_choice import choice, control
from diagnose_occlusion_inputs import tensor_hashes
from mask_choice_taps import prepare as prepare_taps
from mask_choice_override import INDEX_INPUT, prepare as prepare_override
import owned
import validate_export as v
from state import TemporalState


def summarize(report):
    rows = [row for row in report['frames'] if 'comparisons' in row]
    post = [row for row in rows if row['index'] >= report['forkFrame']]
    result = {'framesReplayed': len(rows), 'forkApplied': report['forkApplied'],
              'controlsPassed': report.get('controlsPassed', False),
              'normalPriorComparisonMismatchFrames': [row['index'] for row in rows
                  if not row['priorNormalComparisonExact']],
              'originalPriorObservationMismatchFrames': [row['index'] for row in rows
                  if not row['priorOriginalObservationExact']],
              'postForkComparisons': {}}
    for label in ('normalGPUAgainstOriginal', 'overrideGPUAgainstOriginal'):
        checks = [(row['index'], row['comparisons'][label]) for row in post]
        if not checks:
            continue
        result['postForkComparisons'][label] = {
            'failedFrames': [index for index, check in checks if not check['passed']],
            'minimumMaskIoU': min(check['maskIoUAgainstReference'] for _, check in checks),
            'minimumPointerCosine': min(check['outputs']['object_pointer']['cosineSimilarity'] for _, check in checks),
            'minimumMemoryCosine': min(check['outputs']['memory_features']['cosineSimilarity'] for _, check in checks),
            'presenceMismatchFrames': [index for index, check in checks if not check['sameObjectPresence']]}
    result['normalFailureResolvedByOverrideFrames'] = [row['index'] for row in post
        if not row['comparisons']['normalGPUAgainstOriginal']['passed']
        and row['comparisons']['overrideGPUAgainstOriginal']['passed']]
    result['normalPassFailedWithOverrideFrames'] = [row['index'] for row in post
        if row['comparisons']['normalGPUAgainstOriginal']['passed']
        and not row['comparisons']['overrideGPUAgainstOriginal']['passed']]
    result['interpretationAllowed'] = (report['completed'] and report['forkApplied']
        and result['controlsPassed'] and not result['normalPriorComparisonMismatchFrames']
        and not result['originalPriorObservationMismatchFrames'])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('upstream', 'video', 'phone-report', 'fixture', 'comparison', 'choice-report', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--fork-frame', type=int, default=26)
    parser.add_argument('--frames', type=int, default=49)
    args = parser.parse_args()
    if platform.system() != 'Darwin':
        parser.error('Run locally on the Mac.')
    prior = json.loads(args.comparison.read_text())
    taps_prior = json.loads(args.choice_report.read_text())
    if (prior.get('contract') != 'bytewave.occlusion-comparison.diagnostic.v1'
            or prior.get('completed') is not True or prior.get('previewOnly') is not False
            or prior.get('inferencePerformed') is not True
            or taps_prior.get('contract') != 'bytewave.mask-choice.diagnostic.v1'
            or taps_prior.get('completed') is not True
            or taps_prior.get('sourceComparisonSHA256') != occlusion.sha(args.comparison)):
        parser.error('Expected completed, linked occlusion and mask-choice reports.')
    total = prior['effectiveFrames']
    previous_rows = prior['frames']
    if (len(previous_rows) != total
            or [row['index'] for row in previous_rows] != list(range(total))
            or not 1 <= args.fork_frame < args.frames <= min(total, 600)):
        parser.error('Invalid fork frame or replay length.')
    targets = [row for row in taps_prior['frames'] if row['index'] == args.fork_frame]
    if (len(targets) != 1 or not targets[0].get('taps', {}).get('interpretationAllowed')
            or not targets[0]['taps']['choiceDiffersFromOriginal']
            or not targets[0]['priorGPUComparisonExact']
            or not targets[0]['priorOriginalObservationExact']):
        parser.error('Fork requires a previously controlled, reproducible choice disagreement.')
    target = targets[0]
    normal_index = target['taps']['choice']['selectedIndex']
    forced_index = target['originalChoice']['selectedIndex']
    if normal_index not in range(3) or forced_index not in range(3) or normal_index == forced_index:
        parser.error('Expected distinct valid candidate indices.')
    args.new_video, args.point = True, prior['pointNormalizedTopLeft']
    _, fixture, _ = occlusion.verify_inputs(args)
    identity = {'sourceVideoSHA256': occlusion.sha(args.video),
                'sourceFileBytes': args.video.stat().st_size,
                'phoneReportSHA256': occlusion.sha(args.phone_report),
                'fixtureSHA256': occlusion.sha(args.fixture / 'fixture.json'),
                'modelFiles': fixture['modelFiles'], 'upstream': owned.UPSTREAM_REVISION,
                'checkpointSHA256': owned.CHECKPOINT_SHA256}
    for name, value in {**identity, 'pointNormalizedTopLeft': args.point}.items():
        if prior.get(name) != value or taps_prior.get(name) != value:
            raise ValueError(f'Prior report identity mismatch: {name}')
    if taps_prior.get('referenceNumFrames') != total:
        raise ValueError('Reference video length differs.')
    import coremltools as ct
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'previews').mkdir()
    path = args.output / 'report.json'
    report = {'scope': __doc__, 'contract': 'bytewave.mask-choice-intervention.diagnostic.v1',
              'completed': False, 'readyForDeviceValidation': False, **identity,
              'sourceComparisonSHA256': occlusion.sha(args.comparison),
              'sourceChoiceReportSHA256': occlusion.sha(args.choice_report),
              'pointNormalizedTopLeft': args.point, 'referenceNumFrames': total,
              'requestedFrames': args.frames, 'forkFrame': args.fork_frame,
              'normalCandidateIndex': normal_index, 'forcedCandidateIndex': forced_index,
              'computeUnits': 'CPU_AND_GPU', 'forkApplied': False,
              'policy': {'maskIoUMinimum': .95, 'cosineMinimum': .99,
                         'float32Atol': .001, 'float32Rtol': .001,
                         'interventionRequiresExactSameIndexControl': True},
              'torchThreads': {'intraOp': 1, 'interOp': 1},
              'versions': {name: importlib.metadata.version(name)
                           for name in ('torch', 'coremltools', 'numpy', 'av', 'Pillow')},
              'memoryPolicy': 'Two independent normal bounded states, initial condition retained. Identical GPU outputs before fork; own outputs from fork onward, including absent frames. No reference tensors enter GPU state.',
              'interpretation': 'One original-reference candidate index is a diagnostic oracle only. Candidate indices do not identify subjects. best_iou stays the maximum natural score. This intervention measures the combined downstream mask/pointer/memory effect, not any one contribution separately.',
              'frames': [], 'variants': {}, 'activeFrameIndex': None, 'activeOperation': None}
    decoder, hook = None, None
    captured = []

    def checkpoint(label):
        report['activeOperation'] = label
        v.write_json(path, report)

    def capture(_module, _inputs, output):
        captured.append((output[1].detach().cpu().numpy().copy(), int(output[2].shape[1])))

    def call(component, inputs, label):
        checkpoint(label)
        result = backend.call(component, {k: x.copy() if isinstance(x, np.ndarray) else x
                                          for k, x in inputs.items()})
        tensor_hashes(result, v.NAMES[component])
        return result

    def side_call(label, inputs, selected=None):
        checkpoint(label)
        feed = {name: np.array(inputs[name], dtype=np.float32, copy=True)
                for name in v.INPUTS['Propagator']}
        if selected is not None:
            feed[INDEX_INPUT] = np.asarray([selected], dtype=np.int32)
        result = sides[label].predict(feed)
        tensor_hashes(result, v.NAMES['Propagator'])
        return result

    try:
        package = args.fixture / 'models/BWTemporalPropagator.mlpackage'
        sides = {}
        for label in ('tapped', 'override'):
            checkpoint('prepare ' + label)
            destination = args.output / (label + '.mlpackage')
            report['variants'][label] = (prepare_taps(package, destination, True)
                                         if label == 'tapped' else prepare_override(package, destination))
            checkpoint('load ' + label)
            sides[label] = ct.models.MLModel(str(destination), compute_units=ct.ComputeUnit.CPU_AND_GPU)
        mapping = report['variants']['tapped']['mapping']
        override_mapping = report['variants']['override']['mapping']
        if any(mapping[key] != override_mapping[key] for key in ('scores', 'index')):
            raise ValueError('Tap and override mappings disagree.')
        checkpoint('load original reference and installed GPU models')
        model = owned.load_reference(args.upstream)
        backend = occlusion.FixtureBackend(args.fixture)
        states = {label: TemporalState() for label in ('normalGPU', 'overrideGPU')}
        history = {'cond_frame_outputs': {}, 'non_cond_frame_outputs': {}}
        point = [min(float(x) * 1024, 1023) for x in args.point]
        decoder = v.video_frames(args.video, count=args.frames)
        previous_empty = None
        with torch.inference_mode():
            for index, timestamp, image, tensor, digest in decoder:
                previous = previous_rows[index]
                if (digest != previous['imageSHA256'] or timestamp.numerator != previous['ptsNumerator']
                        or timestamp.denominator != previous['ptsDenominator']):
                    raise ValueError(f'Prior decoded pixels/PTS differ at frame {index}.')
                report['activeFrameIndex'] = index
                print(f'Choice intervention: {index + 1}/{args.frames}', flush=True)
                checkpoint('original reference')
                if index == args.fork_frame:
                    captured.clear()
                    hook = model.sam_mask_decoder.register_forward_hook(capture)
                original = v.reference_step(model, torch.from_numpy(tensor), point, index, history,
                                            num_frames=total)
                if hook is not None:
                    hook.remove()
                    hook = None
                tensor_hashes(original, v.NAMES['Propagator'])
                features = call('ImageEncoder', {'image': tensor, 'pil_image': image}, 'shared GPU encoder')
                row = {'index': index, 'ptsSeconds': float(timestamp),
                       'ptsNumerator': timestamp.numerator, 'ptsDenominator': timestamp.denominator,
                       'imageSHA256': digest, 'featureTensors': tensor_hashes(features, owned.IMAGE_OUTPUTS)}
                # Append early so failed fork controls are retained in report.json.
                report['frames'].append(row)
                if index == 0:
                    normal = call('Initializer', {**features,
                        'point_coords': np.asarray([[point]], dtype=np.float32),
                        'point_labels': np.ones((1, 1), dtype=np.int32)}, 'GPU initializer')
                    normal.update(call('InitialMemoryEncoder', {**features, **normal}, 'GPU initial memory'))
                    outputs = {'normalGPU': normal, 'overrideGPU': normal}
                else:
                    inputs = {label: {**features, **state.pack(timestamp)} for label, state in states.items()}
                    hashes = {label: tensor_hashes(feed, v.INPUTS['Propagator']) for label, feed in inputs.items()}
                    row['inputTensors'] = hashes
                    if index <= args.fork_frame:
                        if hashes['normalGPU'] != hashes['overrideGPU']:
                            raise ValueError('Histories diverged before intervention.')
                        normal = call('Propagator', inputs['normalGPU'], 'normal GPU')
                        outputs = {'normalGPU': normal, 'overrideGPU': normal}
                    else:
                        order = ('normalGPU', 'overrideGPU') if index % 2 else ('overrideGPU', 'normalGPU')
                        row['callOrder'] = list(order)
                        outputs = {label: call('Propagator', inputs[label], label) for label in order}
                    if index == args.fork_frame:
                        if (hashes['normalGPU'] != target['inputTensors']
                                or len(captured) != 2 or captured[0][1] != 3 or captured[1][1] != 3
                                or not np.array_equal(captured[0][0], captured[1][0])):
                            raise ValueError('Fork inputs or reference head calls differ from expected replay.')
                        original_choice = choice(captured[0][0], np.argmax(captured[0][0], axis=-1))
                        tapped = side_call('tapped', inputs['normalGPU'])
                        same = side_call('override', inputs['normalGPU'], normal_index)
                        tapped_choice = choice(tapped[mapping['scores']], tapped[mapping['index']])
                        same_choice = choice(same[mapping['scores']], same[mapping['index']])
                        controls = {'tapped': control(normal, tapped), 'sameIndexOverride': control(normal, same)}
                        row['fork'] = {'controls': controls, 'originalChoice': original_choice,
                                       'naturalGPUChoice': tapped_choice, 'overrideNaturalChoice': same_choice}
                        report['controlsPassed'] = (all(check['allOutputsExact'] for check in controls.values())
                            and original_choice == target['originalChoice']
                            and tapped_choice == target['taps']['choice'] and same_choice == tapped_choice
                            and occlusion.observation(original) == previous['streams']['original']
                            and v.compare(original, normal, coreml=True) == previous['comparisons']['coremlGPUAgainstOriginal'])
                        checkpoint('fork controls recorded')
                        if not report['controlsPassed']:
                            raise ValueError('Fork control failed; alternate selection was not committed.')
                        alternate = side_call('override', inputs['normalGPU'], forced_index)
                        # Changing the input index must not perturb the upstream scores,
                        # natural argmax, max-score output, object score or position encoding.
                        unaffected = (mapping['scores'], mapping['index'], 'best_iou',
                                      'object_score', 'memory_positions')
                        stable = {name: bool(np.array_equal(same[name], alternate[name])) for name in unaffected}
                        row['fork']['unaffectedOutputsExact'] = stable
                        row['fork']['forcedCandidateIndex'] = forced_index
                        row['fork']['sameInputAlternateAgainstNormal'] = v.compare(normal, alternate, coreml=True)
                        if not all(stable.values()):
                            report['controlsPassed'] = False
                            raise ValueError('Alternate index changed an unrelated output; not committed.')
                        outputs['overrideGPU'] = alternate
                row['comparisons'] = {
                    'normalGPUAgainstOriginal': v.compare(original, outputs['normalGPU'], coreml=True),
                    'overrideGPUAgainstOriginal': v.compare(original, outputs['overrideGPU'], coreml=True),
                    'overrideGPUAgainstNormal': v.compare(outputs['normalGPU'], outputs['overrideGPU'], coreml=True)}
                row['priorNormalComparisonExact'] = row['comparisons']['normalGPUAgainstOriginal'] == previous['comparisons']['coremlGPUAgainstOriginal']
                row['priorOriginalObservationExact'] = occlusion.observation(original) == previous['streams']['original']
                row['streams'] = {label: occlusion.observation(value)
                                  for label, value in {'original': original, **outputs}.items()}
                for label, state in states.items():
                    state.commit(outputs[label], timestamp, state.token)
                if index == args.fork_frame:
                    report['forkApplied'] = True
                row['statesAfter'] = {label: state.summary() for label, state in states.items()}
                empty = tuple(value['lowMaskEmpty'] for value in row['streams'].values())
                if index >= args.fork_frame - 1 or index == 0 or empty != previous_empty:
                    relative = f'previews/frame-{index:04d}.png'
                    occlusion.save_preview(args.output / relative, image,
                        {'original': original, 'normal GPU': outputs['normalGPU'],
                         'one choice changed': outputs['overrideGPU']}, index, float(timestamp))
                    row['preview'] = relative
                previous_empty = empty
                v.write_json(path, report)
        report['completed'] = len(report['frames']) == args.frames and report['forkApplied']
        report['activeOperation'] = None
    except Exception as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        if hook is not None:
            hook.remove()
        if decoder is not None:
            decoder.close()
        report['summary'] = summarize(report)
        v.write_json(path, report)
        v.write_json(args.output / 'summary.json', {k: value for k, value in report.items() if k != 'frames'})
        print(f'Report: {path}', flush=True)


if __name__ == '__main__':
    main()
