#!/usr/bin/env python3
"""Localize occlusion differences with shared original-PyTorch memory inputs.

Replays a short prefix of a completed occlusion comparison. Original PyTorch
alone advances history; all experimental outputs are discarded after comparison.
Checks encoder features, identical-input CPU/GPU propagators and a GPU-encoder
feature substitution. Existing numerical gates are unchanged. No export, model
promotion, speed measurement, recovery-policy change or iPhone parity claim.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform

import numpy as np
import torch

import diagnose_occlusion as occlusion
import owned
import validate_export as v
from state import TemporalState


class ComponentBackend(v.CoreMLBackend):
    def __init__(self, root, units):
        import coremltools as ct
        # Exact package hashes are checked by occlusion.verify_inputs first.
        self.models = {name: ct.models.MLModel(
            str(root / f'models/BWTemporal{name}.mlpackage'),
            compute_units=getattr(ct.ComputeUnit, units))
            for name in ('ImageEncoder', 'Propagator')}


def tensor_hashes(values, names):
    result = {}
    for name in names:
        value = np.ascontiguousarray(values[name])
        if value.shape != v.SHAPES[name] or not np.isfinite(value).all():
            raise ValueError(f'Invalid input/output tensor: {name}')
        result[name] = {'shape': list(value.shape), 'dtype': str(value.dtype),
                        'sha256': hashlib.sha256(value.tobytes()).hexdigest()}
    return result


def encoder_metrics(reference, actual):
    # Descriptive feature metrics, not a new pass/fail gate.
    tensor_hashes(reference, owned.IMAGE_OUTPUTS)
    tensor_hashes(actual, owned.IMAGE_OUTPUTS)
    return {name: {
        'maximumAbsoluteError': float(np.max(np.abs(reference[name] - actual[name]))),
        'cosineSimilarity': v.cosine(reference[name], actual[name]),
        'float32Close': bool(np.allclose(reference[name], actual[name], atol=.001, rtol=.001))}
        for name in owned.IMAGE_OUTPUTS}


def summarize(report):
    result = {'framesReplayed': report['framesReplayed'],
              'framesInspected': len(report['frames']), 'comparisons': {}}
    labels = sorted({label for row in report['frames'] for label in row['comparisons']})
    for label in labels:
        rows = [(row['index'], row['comparisons'][label]) for row in report['frames']]
        result['comparisons'][label] = {
            'failedFrames': [index for index, check in rows if not check['passed']],
            'minimumMaskIoU': min(check['maskIoUAgainstReference'] for _, check in rows),
            'minimumPointerCosine': min(check['outputs']['object_pointer']['cosineSimilarity'] for _, check in rows),
            'minimumMemoryCosine': min(check['outputs']['memory_features']['cosineSimilarity'] for _, check in rows),
            'presenceMismatchFrames': [index for index, check in rows if not check['sameObjectPresence']]}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream', type=Path, required=True)
    parser.add_argument('--video', type=Path, required=True)
    parser.add_argument('--phone-report', type=Path, required=True)
    parser.add_argument('--fixture', type=Path, required=True)
    parser.add_argument('--comparison', type=Path, required=True,
                        help='Completed diagnose_occlusion report.json for this exact video and prompt.')
    parser.add_argument('--first-frame', type=int, default=26)
    parser.add_argument('--last-frame', type=int, default=48)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if platform.system() != 'Darwin':
        parser.error('Run locally on the Mac.')
    prior = json.loads(args.comparison.read_text())
    if (prior.get('contract') != 'bytewave.occlusion-comparison.diagnostic.v1'
            or prior.get('completed') is not True or prior.get('previewOnly') is not False
            or prior.get('inferencePerformed') is not True):
        parser.error('Expected a completed occlusion inference report.')
    total = prior['effectiveFrames']
    prior_rows = prior['frames']
    if (len(prior_rows) != total
            or [row['index'] for row in prior_rows] != list(range(total))
            or not 1 <= args.first_frame <= args.last_frame < total
            or args.last_frame > 599):
        parser.error('Invalid frame range or incomplete source report.')
    # Inherit the explicit prompt, but verify the exact video bytes against the
    # prior Mac experiment, not the older, untrimmed phone recording.
    args.new_video = True
    args.point = prior['pointNormalizedTopLeft']
    _, fixture, _ = occlusion.verify_inputs(args)
    expected = {'sourceVideoSHA256': occlusion.sha(args.video),
                'sourceFileBytes': args.video.stat().st_size,
                'phoneReportSHA256': occlusion.sha(args.phone_report),
                'fixtureSHA256': occlusion.sha(args.fixture / 'fixture.json'),
                'modelFiles': fixture['modelFiles'],
                'upstream': owned.UPSTREAM_REVISION,
                'checkpointSHA256': owned.CHECKPOINT_SHA256}
    for key, value in expected.items():
        if prior.get(key) != value:
            raise ValueError(f'Source comparison mismatch: {key}')
    selected = sorted(set(range(args.first_frame, args.last_frame + 1)) |
                      {index for index in (1, 5) if index <= args.last_frame})
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'previews').mkdir()
    path = args.output / 'report.json'
    report = {
        'scope': __doc__, 'contract': 'bytewave.occlusion-shared-inputs.diagnostic.v1',
        'completed': False, 'readyForDeviceValidation': False, **expected,
        'sourceComparisonSHA256': occlusion.sha(args.comparison),
        'versions': {name: importlib.metadata.version(name)
                     for name in ('torch', 'numpy', 'av', 'Pillow', 'coremltools')},
        'torchThreads': {'intraOp': 1, 'interOp': 1},
        'pointNormalizedTopLeft': args.point, 'selectedFrames': selected,
        'referenceNumFrames': total, 'prefixFramesRequested': args.last_frame + 1,
        'framesReplayed': 0, 'frames': [],
        'memoryOwner': 'Original PyTorch only; bounded banks populated from its outputs, including absent frames.',
        'interpretation': 'Same-input propagator checks isolate immediate backend/package differences. Feature substitution measures encoder effects on this frame. Prior independent-history results are context, not directly matched inputs. CPU_ONLY vs CPU_AND_GPU specifies allowed devices, not measured placement.',
        'initializationLimit': 'Frame 0 uses original initialization only. This does not isolate candidate initializer error or prove why independent histories diverged.',
        'policy': {'maskIoUMinimumCoreML': .95, 'cosineMinimumCoreML': .99,
                   'maskIoUMinimumFP32': .999, 'float32Atol': .001, 'float32Rtol': .001},
        'activeFrameIndex': None, 'activeOperation': None}
    decoder = None
    def call(backend, component, inputs, label):
        report['activeOperation'] = label
        v.write_json(path, report)
        # Each backend gets private arrays so an in-place operation cannot
        # contaminate another backend's supposedly identical inputs.
        copied = {key: value.copy() if isinstance(value, np.ndarray) else value
                  for key, value in inputs.items()}
        result = backend.call(component, copied)
        tensor_hashes(result, v.NAMES[component])
        return result
    try:
        v.write_json(path, report)
        report['activeOperation'] = 'load original PyTorch'
        v.write_json(path, report)
        model = owned.load_reference(args.upstream)
        fp32 = v.TorchBackend(owned.components(model))
        report['activeOperation'] = 'load CPU_ONLY packages'
        v.write_json(path, report)
        cpu = ComponentBackend(args.fixture, 'CPU_ONLY')
        report['activeOperation'] = 'load CPU_AND_GPU packages'
        v.write_json(path, report)
        gpu = ComponentBackend(args.fixture, 'CPU_AND_GPU')
        history = {'cond_frame_outputs': {}, 'non_cond_frame_outputs': {}}
        state = TemporalState()
        point = [min(float(x) * 1024, 1023) for x in args.point]
        decoder = v.video_frames(args.video, count=args.last_frame + 1)
        with torch.inference_mode():
            for index, timestamp, image, tensor, image_hash in decoder:
                previous = prior_rows[index]
                if (image_hash != previous['imageSHA256']
                        or timestamp.numerator != previous['ptsNumerator']
                        or timestamp.denominator != previous['ptsDenominator']):
                    raise ValueError(f'Decoded pixels/PTS differ from prior comparison at frame {index}.')
                report.update(activeFrameIndex=index, activeOperation='original reference')
                v.write_json(path, report)
                print(f'Shared-input replay: {index + 1}/{args.last_frame + 1}', flush=True)
                original = v.reference_step(model, torch.from_numpy(tensor), point, index, history,
                                            num_frames=total)
                tensor_hashes(original, v.NAMES['Propagator'])
                if index in selected:
                    # Pack BEFORE committing this frame; condition and recent
                    # memory are entirely from earlier original-PyTorch frames.
                    banks = state.pack(timestamp)
                    image_inputs = {'image': tensor, 'pil_image': image}
                    features = call(fp32, 'ImageEncoder', image_inputs, 'FP32 encoder')
                    cpu_features = call(cpu, 'ImageEncoder', image_inputs, 'CPU encoder')
                    gpu_features = call(gpu, 'ImageEncoder', image_inputs, 'GPU encoder')
                    shared = {**features, **banks}
                    substituted = {**gpu_features, **banks}
                    reference = call(fp32, 'Propagator', shared, 'FP32 shared-input propagator')
                    cpu_result = call(cpu, 'Propagator', shared, 'CPU shared-input propagator')
                    gpu_result = call(gpu, 'Propagator', shared, 'GPU shared-input propagator')
                    feature_only = call(fp32, 'Propagator', substituted, 'FP32 propagator with GPU features')
                    combined = call(gpu, 'Propagator', substituted, 'GPU propagator with GPU features')
                    comparisons = {
                        'packingControlFP32AgainstOriginal': v.compare(original, reference, coreml=False),
                        'cpuSameInputAgainstFP32': v.compare(reference, cpu_result, coreml=True),
                        'gpuSameInputAgainstFP32': v.compare(reference, gpu_result, coreml=True),
                        'gpuSameInputAgainstCPU': v.compare(cpu_result, gpu_result, coreml=True),
                        'gpuFeaturesThroughFP32AgainstFP32': v.compare(reference, feature_only, coreml=True),
                        'gpuCombinedAgainstFP32': v.compare(reference, combined, coreml=True),
                        'gpuCombinedAgainstOriginal': v.compare(original, combined, coreml=True)}
                    row = {'index': index, 'ptsSeconds': float(timestamp),
                           'imageSHA256': image_hash, 'stateBefore': state.summary(),
                           'sharedInputTensors': tensor_hashes(shared, v.INPUTS['Propagator']),
                           'substitutedInputTensors': tensor_hashes(substituted, v.INPUTS['Propagator']),
                           'encoder': {'cpuAgainstFP32': encoder_metrics(features, cpu_features),
                                       'gpuAgainstFP32': encoder_metrics(features, gpu_features),
                                       'gpuAgainstCPU': encoder_metrics(cpu_features, gpu_features)},
                           'comparisons': comparisons,
                           'priorIndependentHistoryCoreML': previous['comparisons']['coremlGPUAgainstOriginal'],
                           'priorOriginalObservation': previous['streams']['original'],
                           'currentOriginalObservation': occlusion.observation(original)}
                    relative = f'previews/frame-{index:04d}.png'
                    occlusion.save_preview(args.output / relative, image,
                        {'original': original, 'shared FP32': reference,
                         'shared GPU': gpu_result, 'GPU features + GPU': combined}, index, float(timestamp))
                    row['preview'] = relative
                    report['frames'].append(row)
                state.commit(original, timestamp, state.token)
                report['framesReplayed'] = index + 1
                v.write_json(path, report)
        report['completed'] = (report['framesReplayed'] == args.last_frame + 1
                               and [row['index'] for row in report['frames']] == selected)
        report['activeOperation'] = None
        report['finalState'] = state.summary()
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
