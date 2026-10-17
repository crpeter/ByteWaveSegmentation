#!/usr/bin/env python3
"""Measure appearance signals for occlusion candidates without accepting them.

Replay normal GPU tracking unchanged and reproduce its prior reference checks.
At explicitly selected frames, reproduce the conditioning-only memory probe.
Compare both outputs with fixed, earlier normal predictions using object-pointer
cosine, mask-pooled raw encoder cosine and masked RGB histogram intersection.
These signals are NOT calibrated identity scores. Reference masks need visual
review. No thresholds, recovery, state filtering, export or app changes.
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


def describe(features, output, image):
    """Descriptors only; never alter model outputs or mask arrays."""
    mask = output['high_res_mask'][0, 0] > 0
    pixels = int(mask.sum())
    if pixels == 0 or not (output['low_res_mask'] > 0).any() or output['object_score'].item() <= 0:
        return None
    # Exact area coverage of each 16x16 image tile weights the 64x64 features.
    # The receptive fields can still contain background; this is not a re-ID model.
    coverage = mask.reshape(64, 16, 64, 16).mean(axis=(1, 3), dtype=np.float64)
    raw = features['raw_vision_features'][0].astype(np.float64)
    pooled = (raw * coverage[None]).sum(axis=(1, 2)) / coverage.sum()
    rgb = np.asarray(image, dtype=np.uint8)[mask].astype(np.int32)
    bins = rgb // 32
    joint = bins[:, 0] * 64 + bins[:, 1] * 8 + bins[:, 2]
    histogram = np.bincount(joint, minlength=512).astype(np.float64) / pixels
    ys, xs = np.nonzero(mask)
    return {
        'pointer': output['object_pointer'].astype(np.float64).ravel().copy(),
        'pooledEncoder': pooled,
        'rgbHistogram': histogram,
        'highMaskPixels': pixels,
        'featureCoverageSum': float(coverage.sum()),
        'boxNormalizedXYXY': [float(xs.min() / 1024), float(ys.min() / 1024),
                              float((xs.max() + 1) / 1024), float((ys.max() + 1) / 1024)]}


def cosine(left, right):
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.clip(np.dot(left, right) / denominator, -1, 1)) if denominator > 0 else None


def metrics(value, anchor):
    if value is None:
        return None
    return {
        'pointerCosine': cosine(value['pointer'], anchor['pointer']),
        'pooledEncoderCosine': cosine(value['pooledEncoder'], anchor['pooledEncoder']),
        'rgbHistogramIntersection': float(np.minimum(value['rgbHistogram'], anchor['rgbHistogram']).sum()),
        'highMaskAreaRatio': value['highMaskPixels'] / anchor['highMaskPixels']}


def public_descriptor(value):
    if value is None:
        return None
    return {key: item.tolist() if isinstance(item, np.ndarray) else item
            for key, item in value.items()}


def summarize(report):
    rows = report['frames']
    return {
        'framesReplayed': sum(bool(row.get('rowCompleted')) for row in rows),
        'baselineMismatchFrames': [row['index'] for row in rows if row.get('baselineReproduced') is False],
        'probeMismatchFrames': [row['index'] for row in rows if row.get('probeReproduced') is False],
        'interpretationAllowed': bool(report['completed'] and report.get('conditioningUnchangedAtEnd')
            and all(row.get('baselineReproduced') is True for row in rows)
            and all(row.get('probeReproduced') is True for row in rows if 'probeReproduced' in row)),
        'measurements': [{key: row[key] for key in (
            'index', 'ptsSeconds', 'streams', 'appearance', 'searchVsNormal', 'preview')}
            for row in rows if row.get('rowCompleted') and 'appearance' in row],
        'meaning': 'Control reproduction permits inspection only. No identity pass/fail, recommended threshold, accuracy improvement or device-readiness claim.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('upstream', 'video', 'phone-report', 'fixture', 'comparison', 'memory-probe', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--anchor-frames', type=int, nargs='+', required=True,
                        help='Earlier normal predictions used only as fixed appearance references; inspect their previews.')
    parser.add_argument('--inspect-frames', type=int, nargs='+', required=True,
                        help='Post-snapshot frames to measure, not inputs to an online recovery rule.')
    args = parser.parse_args()
    if platform.system() != 'Darwin':
        parser.error('Run locally on the Mac.')
    prior = json.loads(args.comparison.read_text())
    probe = json.loads(args.memory_probe.read_text())
    if (prior.get('contract') != 'bytewave.occlusion-comparison.diagnostic.v1'
            or prior.get('completed') is not True or prior.get('previewOnly') is not False
            or prior.get('inferencePerformed') is not True
            or probe.get('contract') != 'bytewave.occlusion-memory-probe.diagnostic.v1'
            or probe.get('completed') is not True
            or probe.get('summary', {}).get('interpretationAllowed') is not True
            or probe.get('sourceComparisonSHA256') != occlusion.sha(args.comparison)):
        parser.error('Expected completed, linked comparison and controlled memory-probe reports.')
    total = prior['effectiveFrames']
    previous_rows, probe_rows = prior['frames'], probe['frames']
    fork = probe['snapshotInputFrame']
    anchors_requested = sorted(set(args.anchor_frames))
    inspections = sorted(set(args.inspect_frames))
    count = max(inspections) + 1
    if (len(previous_rows) != total
            or [row['index'] for row in previous_rows] != list(range(total))
            or [row['index'] for row in probe_rows] != list(range(len(probe_rows)))
            or not all(row.get('rowCompleted') for row in probe_rows)
            or not 1 <= len(anchors_requested) <= 8
            or not 1 <= len(inspections) <= 64
            or not 0 <= anchors_requested[0] <= anchors_requested[-1] < fork
            or not 1 <= fork <= inspections[0]
            or not 2 <= count <= min(total, len(probe_rows), 600)
            or probe.get('referenceNumFrames') != total):
        parser.error('Use 1-8 earlier anchors and 1-64 post-snapshot inspection frames within both reports.')
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
    for index in inspections:
        if 'conditioningOnly' not in probe_rows[index].get('streams', {}):
            raise ValueError(f'No archived conditioning-only control at frame {index}')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'previews').mkdir()
    path = args.output / 'report.json'
    report = {
        'scope': __doc__, 'contract': 'bytewave.occlusion-identity-signals.diagnostic.v1',
        'completed': False, 'readyForDeviceValidation': False, **identity,
        'diagnosticScriptSHA256': occlusion.sha(Path(__file__)),
        'sourceComparisonSHA256': occlusion.sha(args.comparison),
        'sourceMemoryProbeSHA256': occlusion.sha(args.memory_probe),
        'pointNormalizedTopLeft': args.point, 'referenceNumFrames': total,
        'framesToReplay': count, 'anchorFrames': anchors_requested, 'inspectFrames': inspections,
        'computeUnits': 'CPU_AND_GPU',
        'descriptorDefinition': {
            'pointer': 'Cosine of the 256-element output object pointer; same model supplies candidate and pointer, so not independent identity evidence.',
            'encoder': 'Raw 64x64 encoder map pooled using exact high-mask coverage of 16x16 image tiles. Context/receptive fields may include background.',
            'colour': 'Joint RGB 8x8x8 histogram within binary high mask, normalized by pixel count; histogram intersection. Sensitive to illumination and background leakage.',
            'geometry': 'High-mask pixel count, normalized XYXY box, candidate/anchor area ratio; no area gate.',
            'missing': 'Nonpositive object score or empty low/high mask yields null descriptor. Zero-norm cosine yields null, not a match.',
            'smallMasks': 'Tiny masks and partial visibility can produce unstable descriptors. Coverage and pixel counts are reported; no minimum-area threshold is imposed.'},
        'referenceLimit': 'Manually chosen earlier normal predictions, not ground truth. Inspect anchor previews. All anchors precede every inspected frame. No anchor updates after absence and no selection by measured similarity.',
        'policy': {'maskIoUMinimum': .95, 'cosineMinimum': .99, 'float32Atol': .001, 'float32Rtol': .001},
        'memoryPolicy': 'Only unchanged normal predictions enter bounded GPU state. Original PyTorch owns independent history. Conditioning-only candidate outputs are discarded. Appearance vectors never enter model state.',
        'controlMeaning': 'Every baseline observation/comparison must reproduce prior report exactly. Every inspected search observation/comparison must reproduce linked memory probe exactly. Prior frame annotations only select offline measurements.',
        'versions': {name: importlib.metadata.version(name) for name in ('torch', 'coremltools', 'numpy', 'av', 'Pillow')},
        'torchThreads': {'intraOp': 1, 'interOp': 1},
        'anchors': {}, 'frames': [], 'activeFrameIndex': None, 'activeOperation': None}
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
        checkpoint('load reference and exact GPU packages')
        model = owned.load_reference(args.upstream)
        backend = occlusion.FixtureBackend(args.fixture)
        state = TemporalState()
        anchors = {}
        initial_hashes = None
        history = {'cond_frame_outputs': {}, 'non_cond_frame_outputs': {}}
        point = [min(float(x) * 1024, 1023) for x in args.point]
        decoder = v.video_frames(args.video, count=count)
        with torch.inference_mode():
            for index, timestamp, image, tensor, digest in decoder:
                previous = previous_rows[index]
                if (digest != previous['imageSHA256'] or timestamp.numerator != previous['ptsNumerator']
                        or timestamp.denominator != previous['ptsDenominator']):
                    raise ValueError(f'Decoded pixels/PTS differ at frame {index}')
                report['activeFrameIndex'] = index
                print(f'Identity signals: {index + 1}/{count}', flush=True)
                checkpoint('original reference')
                original = v.reference_step(model, torch.from_numpy(tensor), point, index, history, num_frames=total)
                tensor_hashes(original, v.NAMES['Propagator'])
                features = call('ImageEncoder', {'image': tensor, 'pil_image': image}, 'GPU encoder')
                state.validate_time(timestamp)
                if state.next_index != index:
                    raise ValueError('Normal frame clock changed')
                token = state.token
                if index == 0:
                    normal = call('Initializer', {**features,
                        'point_coords': np.asarray([[point]], dtype=np.float32),
                        'point_labels': np.ones((1, 1), dtype=np.int32)}, 'initial selection')
                    normal.update(call('InitialMemoryEncoder', {**features, **normal}, 'initial memory'))
                else:
                    packed = state.pack(timestamp)
                    normal = call('Propagator', {**features, **packed}, 'normal GPU')
                    bank = conditioning_only(packed)
                    hashes = tensor_hashes(bank, BANK_NAMES)
                    if initial_hashes is None:
                        initial_hashes = hashes
                        report['conditioningOnlyBankHashes'] = hashes
                    elif hashes != initial_hashes:
                        raise ValueError('Original conditioning changed')
                baseline = v.compare(original, normal, coreml=True)
                row = {'index': index, 'ptsSeconds': float(timestamp),
                       'ptsNumerator': timestamp.numerator, 'ptsDenominator': timestamp.denominator,
                       'imageSHA256': digest, 'streams': {'normalGPU': occlusion.observation(normal)},
                       'normalGPUAgainstOriginal': baseline,
                       'baselineReproduced': baseline == previous['comparisons']['coremlGPUAgainstOriginal']
                           and occlusion.observation(normal) == previous['streams']['coremlGPU']
                           and occlusion.observation(original) == previous['streams']['original']}
                report['frames'].append(row)
                if not row['baselineReproduced']:
                    raise ValueError(f'Baseline mismatch at frame {index}; stop interpretation')
                if index in anchors_requested:
                    value = describe(features, normal, image)
                    if value is None:
                        raise ValueError(f'Anchor frame {index} has no present mask; choose a reviewed reference')
                    anchors[index] = value
                    preview = f'previews/anchor-{index:04d}.png'
                    occlusion.save_preview(args.output / preview, image, {'appearance reference; visually verify': normal}, index, float(timestamp))
                    report['anchors'][str(index)] = {'descriptor': public_descriptor(value),
                        'observation': occlusion.observation(normal), 'preview': preview,
                        'imageSHA256': digest, 'ptsSeconds': float(timestamp)}
                if index in inspections:
                    if len(anchors) != len(anchors_requested):
                        raise ValueError('Missing earlier appearance references')
                    candidate = call('Propagator', {**features, **bank}, 'conditioning-only diagnostic')
                    archived = probe_rows[index]
                    comparison = v.compare(original, candidate, coreml=True)
                    row['streams']['conditioningOnly'] = occlusion.observation(candidate)
                    row['conditioningOnlyAgainstOriginal'] = comparison
                    row['probeReproduced'] = (comparison == archived['comparisons']['conditioningOnlyAgainstOriginal']
                        and occlusion.observation(candidate) == archived['streams']['conditioningOnly'])
                    if not row['probeReproduced']:
                        raise ValueError(f'Search probe mismatch at frame {index}; stop interpretation')
                    values = {label: describe(features, output, image)
                              for label, output in [('normalGPU', normal), ('conditioningOnly', candidate)]}
                    row['appearance'] = {label: {
                        'geometry': None if value is None else {key: value[key] for key in ('highMaskPixels', 'featureCoverageSum', 'boxNormalizedXYXY')},
                        'againstAnchors': {str(frame): metrics(value, anchor) for frame, anchor in anchors.items()}}
                        for label, value in values.items()}
                    row['searchVsNormal'] = metrics(values['conditioningOnly'], values['normalGPU']) if values['normalGPU'] is not None else None
                    if tensor_hashes(bank, BANK_NAMES) != initial_hashes:
                        raise ValueError('Diagnostic mutated conditioning bank')
                    row['preview'] = f'previews/frame-{index:04d}.png'
                    occlusion.save_preview(args.output / row['preview'], image,
                        {'unchanged normal GPU': normal, 'search only; NEVER committed': candidate}, index, float(timestamp))
                state.commit(normal, timestamp, token)
                row['normalStateAfter'] = state.summary()
                row['rowCompleted'] = True
                v.write_json(path, report)
        report['conditioningUnchangedAtEnd'] = initial_hashes is not None
        report['completed'] = len(report['frames']) == count
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
