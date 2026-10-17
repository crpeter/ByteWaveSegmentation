#!/usr/bin/env python3
"""Compare two independent GPU histories that differ only in their initial seed.

Normal GPU tracking uses GPU initialization. The second stream receives original
PyTorch frame-0 memory/positions/pointer once, then commits only its own GPU
predictions. Both consume the same GPU encoder features on every later frame.
The conditioning anchor stays pinned under the existing policy. This measures
the seed's total downstream effect, not anchor versus recent-history effects.
Diagnostic only: unchanged models/gates, no export, runtime change, speed or
device-readiness claim. Default 49-frame prefix does not test occlusion recovery.
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


def summarize(report):
    # Frame 0 in the seeded branch is an intervention, not a GPU prediction.
    rows = [row for row in report['frames'] if row['index'] > 0]
    result = {'framesReplayed': len(report['frames']),
              'propagatedFramesCompared': len(rows), 'comparisons': {}}
    if not rows:
        return result
    for label in rows[0]['comparisons']:
        checks = [(row['index'], row['comparisons'][label]) for row in rows]
        failed = [index for index, check in checks if not check['passed']]
        result['comparisons'][label] = {
            'failedFrames': failed, 'firstFailedFrame': failed[0] if failed else None,
            'minimumMaskIoU': min(check['maskIoUAgainstReference'] for _, check in checks),
            'minimumPointerCosine': min(check['outputs']['object_pointer']['cosineSimilarity'] for _, check in checks),
            'minimumMemoryCosine': min(check['outputs']['memory_features']['cosineSimilarity'] for _, check in checks),
            'presenceMismatchFrames': [index for index, check in checks if not check['sameObjectPresence']]}
    result['priorRunObservationDifferences'] = {
        label: [row['index'] for row in report['frames']
                if not row['priorObservationMatches'][label]]
        for label in ('original', 'normalGPU')}
    result['priorNormalFailedFrames'] = [row['index'] for row in rows
                                        if not row['priorNormalComparison']['passed']]
    result['normalFailureResolvedBySeedFrames'] = [row['index'] for row in rows
        if not row['comparisons']['normalGPUAgainstOriginal']['passed']
        and row['comparisons']['originalSeedGPUAgainstOriginal']['passed']]
    result['normalPassFailedWithSeedFrames'] = [row['index'] for row in rows
        if row['comparisons']['normalGPUAgainstOriginal']['passed']
        and not row['comparisons']['originalSeedGPUAgainstOriginal']['passed']]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream', type=Path, required=True)
    parser.add_argument('--video', type=Path, required=True)
    parser.add_argument('--phone-report', type=Path, required=True)
    parser.add_argument('--fixture', type=Path, required=True)
    parser.add_argument('--comparison', type=Path, required=True,
                        help='Completed original occlusion comparison report for the exact video/prompt.')
    parser.add_argument('--frames', type=int, default=49)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if platform.system() != 'Darwin':
        parser.error('Run this diagnostic locally on the Mac.')
    prior = json.loads(args.comparison.read_text())
    if (prior.get('contract') != 'bytewave.occlusion-comparison.diagnostic.v1'
            or prior.get('completed') is not True or prior.get('previewOnly') is not False
            or prior.get('inferencePerformed') is not True):
        parser.error('Expected a completed occlusion inference report.')
    total = prior['effectiveFrames']
    previous_rows = prior['frames']
    if (len(previous_rows) != total
            or [row['index'] for row in previous_rows] != list(range(total))
            or not 2 <= args.frames <= min(total, 600)):
        parser.error('Choose 2..600 frames within the completed comparison prefix.')
    args.new_video = True
    args.point = prior['pointNormalizedTopLeft']
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
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'previews').mkdir()
    path = args.output / 'report.json'
    report = {
        'scope': __doc__, 'contract': 'bytewave.occlusion-seed.diagnostic.v1',
        'completed': False, 'readyForDeviceValidation': False, **identity,
        'sourceComparisonSHA256': occlusion.sha(args.comparison),
        'versions': {name: importlib.metadata.version(name)
                     for name in ('torch', 'numpy', 'av', 'Pillow', 'coremltools')},
        'torchThreads': {'intraOp': 1, 'interOp': 1},
        'computeUnits': 'CPU_AND_GPU', 'referenceNumFrames': total,
        'requestedFrames': args.frames, 'pointNormalizedTopLeft': args.point,
        'memoryPolicy': 'Normal bounded 7 spatial/16 pointer slots; initial condition pinned. All finite valid outputs including absent frames committed. No later reference injection.',
        'intervention': 'At frame 0 only, originalSeedGPU state receives original PyTorch memory_features, memory_positions and object_pointer instead of GPU initialization outputs. Its frame-0 displayed result is a reference seed, not a GPU prediction.',
        'interpretation': 'Both streams share encoder features and one immutable model set, but own separate state. Call order alternates by frame. This is a causal seed intervention on this clip, not a deployable hybrid, nor a separate test of anchor vs downstream history effects. Prior observation equality is descriptive and cannot prove full tensor reproducibility.',
        'policy': {'maskIoUMinimum': .95, 'cosineMinimum': .99,
                   'float32Atol': .001, 'float32Rtol': .001},
        'activeFrameIndex': None, 'activeOperation': 'load original PyTorch', 'frames': []}
    decoder = None

    def call(component, inputs, label):
        report['activeOperation'] = label
        v.write_json(path, report)
        copied = {key: value.copy() if isinstance(value, np.ndarray) else value
                  for key, value in inputs.items()}
        result = backend.call(component, copied)
        tensor_hashes(result, v.NAMES[component])
        return result

    try:
        v.write_json(path, report)
        model = owned.load_reference(args.upstream)
        report['activeOperation'] = 'load GPU packages'
        v.write_json(path, report)
        backend = occlusion.FixtureBackend(args.fixture)
        states = {label: TemporalState() for label in ('normalGPU', 'originalSeedGPU')}
        history = {'cond_frame_outputs': {}, 'non_cond_frame_outputs': {}}
        point = [min(float(x) * 1024, 1023) for x in args.point]
        decoder = v.video_frames(args.video, count=args.frames)
        previous_empty = None
        with torch.inference_mode():
            for index, timestamp, image, tensor, image_hash in decoder:
                previous = previous_rows[index]
                if (image_hash != previous['imageSHA256']
                        or timestamp.numerator != previous['ptsNumerator']
                        or timestamp.denominator != previous['ptsDenominator']):
                    raise ValueError(f'Decoded pixels/PTS differ at frame {index}.')
                report.update(activeFrameIndex=index, activeOperation='original reference')
                v.write_json(path, report)
                print(f'Seed comparison: {index + 1}/{args.frames}', flush=True)
                original = v.reference_step(model, torch.from_numpy(tensor), point, index,
                                            history, num_frames=total)
                tensor_hashes(original, v.NAMES['Propagator'])
                features = call('ImageEncoder', {'image': tensor, 'pil_image': image}, 'shared GPU encoder')
                row = {'index': index, 'ptsNumerator': timestamp.numerator,
                       'ptsDenominator': timestamp.denominator, 'ptsSeconds': float(timestamp),
                       'imageSHA256': image_hash, 'gpuFeatureTensors': tensor_hashes(features, owned.IMAGE_OUTPUTS),
                       'statesBefore': {label: state.summary() for label, state in states.items()}}
                if index == 0:
                    normal = call('Initializer', {**features,
                        'point_coords': np.asarray([[point]], dtype=np.float32),
                        'point_labels': np.ones((1, 1), dtype=np.int32)}, 'normal GPU initializer')
                    normal.update(call('InitialMemoryEncoder', {**features, **normal}, 'normal GPU initial memory'))
                    outputs = {'normalGPU': normal, 'originalSeedGPU': original}
                    row['seedTensors'] = {label: tensor_hashes(output,
                        ('memory_features', 'memory_positions', 'object_pointer'))
                        for label, output in outputs.items()}
                    row['comparisons'] = {'normalGPUInitializationAgainstOriginal': v.compare(original, normal, coreml=True)}
                    report['initializationComparison'] = row['comparisons']['normalGPUInitializationAgainstOriginal']
                    row['seededFrameIsPrediction'] = False
                else:
                    inputs = {label: {**features, **state.pack(timestamp)} for label, state in states.items()}
                    row['propagatorInputTensors'] = {label: tensor_hashes(value, v.INPUTS['Propagator'])
                                                    for label, value in inputs.items()}
                    # Both packs precede either call or commit. Alternate order
                    # to avoid consistently favoring a particular call position.
                    order = ('normalGPU', 'originalSeedGPU') if index % 2 else ('originalSeedGPU', 'normalGPU')
                    row['callOrder'] = list(order)
                    outputs = {label: call('Propagator', inputs[label], label) for label in order}
                    row['comparisons'] = {
                        'normalGPUAgainstOriginal': v.compare(original, outputs['normalGPU'], coreml=True),
                        'originalSeedGPUAgainstOriginal': v.compare(original, outputs['originalSeedGPU'], coreml=True),
                        'originalSeedGPUAgainstNormalGPU': v.compare(outputs['normalGPU'], outputs['originalSeedGPU'], coreml=True)}
                    row['seededFrameIsPrediction'] = True
                # checked() inside commit copies the tensors: no shared history.
                for label, state in states.items():
                    state.commit(outputs[label], timestamp, state.token)
                row['statesAfter'] = {label: state.summary() for label, state in states.items()}
                row['streams'] = {label: occlusion.observation(output)
                                  for label, output in {'original': original, **outputs}.items()}
                row['priorOriginalObservation'] = previous['streams']['original']
                row['priorNormalObservation'] = previous['streams']['coremlGPU']
                row['priorObservationMatches'] = {
                    'original': row['streams']['original'] == row['priorOriginalObservation'],
                    'normalGPU': row['streams']['normalGPU'] == row['priorNormalObservation']}
                row['priorNormalComparison'] = previous['comparisons']['coremlGPUAgainstOriginal']
                empty = tuple(row['streams'][label]['lowMaskEmpty']
                              for label in ('original', 'normalGPU', 'originalSeedGPU'))
                if index == 0 or index >= 26 or index % 5 == 0 or empty != previous_empty:
                    relative = f'previews/frame-{index:04d}.png'
                    seed_label = 'original seed (not GPU)' if index == 0 else 'GPU with original seed'
                    occlusion.save_preview(args.output / relative, image,
                        {'original': original, 'normal GPU': outputs['normalGPU'],
                         seed_label: outputs['originalSeedGPU']}, index, float(timestamp))
                    row['preview'] = relative
                previous_empty = empty
                report['frames'].append(row)
                v.write_json(path, report)
        report['completed'] = len(report['frames']) == args.frames
        report['activeOperation'] = None
        report['finalStates'] = {label: state.summary() for label, state in states.items()}
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
