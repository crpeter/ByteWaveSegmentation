#!/usr/bin/env python3
"""Compare original, owned FP32 and installed Core ML tracking on an occlusion clip.

Run locally on Mac. Independent histories, identical decoded pixels and prompt.
Numerical mismatches are recorded and tracing continues to observe recovery;
shape/nonfinite errors stop immediately. No model export or state-policy change.
This is diagnostic evidence, not ground-truth identity or device validation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform

import av
import numpy as np
from PIL import Image, ImageDraw
import torch

import owned
import validate_export as v
from state import TemporalState


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verify_inputs(args):
    phone = json.loads(args.phone_report.read_text())
    fixture_path = args.fixture / 'fixture.json'
    fixture = json.loads(fixture_path.read_text())
    if (phone.get('schema') != 'bytewave.requested-video-tracking.v1'
            or phone.get('modelVariant') != 'memoryfp16projection'
            or phone.get('segmentsStarted') != 1 or phone.get('seeks') != 0
            or phone.get('cacheSelectionSegment') != 1
            or phone.get('fixtureSHA256') != sha(fixture_path)
            or phone.get('modelFiles') != fixture.get('modelFiles')
            or fixture.get('contract') != owned.CONTRACT
            or fixture.get('upstream') != owned.UPSTREAM_REVISION
            or fixture.get('checkpointSHA256') != owned.CHECKPOINT_SHA256
            or fixture.get('precisionPolicy') != 'encoder-projection-memoryfp16.diagnostic.v1'):
        raise ValueError('Expected one unseeked selection and the exact phone fixture/model bytes.')
    point = phone.get('lastPointNormalizedTopLeft', [])
    if len(point) != 2 or not all(np.isfinite(x) and 0 <= x <= 1 for x in point):
        raise ValueError('Phone report has no valid selection point.')
    if args.video.stat().st_size != phone.get('sourceFileBytes'):
        raise ValueError('Video byte count differs from phone source. Use the original, not a screen recording or re-encode.')
    with av.open(str(args.video)) as container:
        stream = container.streams.video[0]
        if [stream.codec_context.width, stream.codec_context.height] != phone.get('lastDecodedSize'):
            raise ValueError('Source dimensions differ from phone report.')
        duration = float(stream.duration * stream.time_base) if stream.duration is not None else None
        if duration is None or abs(duration - phone['videoDurationSeconds']) > 0.1:
            raise ValueError('Source duration differs from phone report or is unavailable.')
    expected = fixture['modelFiles']
    actual = {}
    for component in v.INPUTS:
        package = args.fixture / f'models/BWTemporal{component}.mlpackage'
        if not package.is_dir():
            raise ValueError(f'Missing package: {package}')
        for path in sorted(package.rglob('*')):
            if path.is_file():
                if not path.resolve().is_relative_to(args.fixture.resolve()):
                    raise ValueError('Model file escapes fixture directory.')
                actual[str(path.relative_to(args.fixture))] = sha(path)
    if not actual or actual != expected:
        raise ValueError('Model package files differ from phone report/fixture.')
    return phone, fixture


class FixtureBackend(v.CoreMLBackend):
    def __init__(self, root):
        import coremltools as ct
        # verify_inputs checks every package byte, including diagnostic metadata.
        self.models = {name: ct.models.MLModel(
            str(root / f'models/BWTemporal{name}.mlpackage'),
            compute_units=ct.ComputeUnit.CPU_AND_GPU) for name in v.INPUTS}


def step(backend, state, image, tensor, point, timestamp):
    state.validate_time(timestamp)
    token = state.token
    features = backend.call('ImageEncoder', {'image': tensor, 'pil_image': image})
    if state.next_index == 0:
        result = backend.call('Initializer', {
            **features, 'point_coords': np.asarray([[point]], dtype=np.float32),
            'point_labels': np.ones((1, 1), dtype=np.int32)})
        result.update(backend.call('InitialMemoryEncoder', {**features, **result}))
    else:
        result = backend.call('Propagator', {**features, **state.pack(timestamp)})
    for name in owned.MASK_OUTPUTS + owned.MEMORY_OUTPUTS:
        value = result[name]
        if value.shape != v.SHAPES[name] or not np.isfinite(value).all():
            raise ValueError(f'Invalid output: {name}')
    state.commit(result, timestamp, token)
    return result


def observation(output):
    low = output['low_res_mask'][0, 0] > 0
    high = output['high_res_mask'][0, 0] > 0
    return {'lowMaskEmpty': not bool(low.any()), 'highMaskEmpty': not bool(high.any()),
            'foregroundFraction': float(low.mean()),
            'objectScore': float(output['object_score'].item()),
            'predictedIoU': float(output['best_iou'].item())}


def save_preview(path, image, outputs, index, seconds):
    # Full 1024-square panels preserve visibility of the small distant subject.
    canvas = Image.new('RGB', (1024 * len(outputs), 1064), 'black')
    draw = ImageDraw.Draw(canvas)
    pixels = np.asarray(image).astype(np.float32)
    for column, (label, output) in enumerate(outputs.items()):
        overlay = pixels.copy()
        mask = output['high_res_mask'][0, 0] > 0
        overlay[mask] = 0.55 * overlay[mask] + 0.45 * np.asarray([90, 160, 255])
        canvas.paste(Image.fromarray(overlay.astype(np.uint8)), (column * 1024, 40))
        draw.text((column * 1024 + 8, 10), f'{label} | frame {index} | source {seconds:.3f}s', fill='white')
    canvas.save(path)


def summarize(report):
    rows = report['frames']
    result = {'framesCompared': len(rows), 'streams': {}, 'comparisons': {}}
    for label in ('original', 'ownedFP32', 'coremlGPU'):
        intervals = []
        active = None
        for row in rows:
            if row['streams'][label]['lowMaskEmpty']:
                if active is None:
                    active = {'firstFrame': row['index'], 'firstPTSSeconds': row['ptsSeconds'], 'frames': 0}
                active.update(lastFrame=row['index'], lastPTSSeconds=row['ptsSeconds'])
                active['frames'] += 1
            elif active is not None:
                active['nextNonemptyFrame'] = row['index']
                intervals.append(active)
                active = None
        if active is not None:
            active['nextNonemptyFrame'] = None
            intervals.append(active)
        result['streams'][label] = {
            'emptyFrames': sum(row['streams'][label]['lowMaskEmpty'] for row in rows),
            'emptyIntervals': intervals}
    for label in ('ownedFP32AgainstOriginal', 'coremlGPUAgainstOriginal'):
        failed = [row['index'] for row in rows if not row['comparisons'][label]['passed']]
        result['comparisons'][label] = {'failedFrames': len(failed),
                                      'firstFailedFrame': failed[0] if failed else None,
                                      'allFramesWithinExistingGates': bool(rows) and not failed}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream', type=Path, required=True)
    parser.add_argument('--video', type=Path, required=True)
    parser.add_argument('--phone-report', type=Path, required=True)
    parser.add_argument('--fixture', type=Path, required=True)
    parser.add_argument('--frames', type=int, default=300)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if platform.system() != 'Darwin':
        parser.error('Run this diagnostic locally on the Mac.')
    if not 20 <= args.frames <= 600:
        parser.error('Choose 20 to 600 frames; default 300 covers the worker disappearance/return.')
    phone, fixture = verify_inputs(args)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    args.output.mkdir(parents=True, exist_ok=False)
    previews = args.output / 'previews'
    previews.mkdir()
    point = [min(float(x) * 1024, 1023) for x in phone['lastPointNormalizedTopLeft']]
    report = {
        'scope': __doc__, 'completed': False, 'readyForDeviceValidation': False,
        'contract': 'bytewave.occlusion-comparison.diagnostic.v1',
        'sourceVideoSHA256': sha(args.video), 'sourceFileBytes': args.video.stat().st_size,
        'phoneReportSHA256': sha(args.phone_report), 'fixtureSHA256': sha(args.fixture / 'fixture.json'),
        'modelFiles': fixture['modelFiles'], 'upstream': owned.UPSTREAM_REVISION,
        'checkpointSHA256': owned.CHECKPOINT_SHA256, 'requestedFrames': args.frames,
        'pointNormalizedTopLeft': phone['lastPointNormalizedTopLeft'],
        'phoneComparisonLimit': 'Phone source has no content hash or saved input pixels. Byte count, dimensions, duration and initial PTS are guards, not proof of identical content. PyAV/Pillow preparation differs from iPhone Core Image. All three Mac streams receive identical pixels; no exact Mac/iPhone parity claim.',
        'memoryPolicy': 'Initial conditioning frame retained; all validated outputs, including absent frames, enter recent memory. No confidence filtering or recovery modification.',
        'qualityMeaning': 'Empty/nonempty transitions do not establish identity. Inspect preview panels. Numerical comparisons use unchanged validate_export gates; failures are retained while this diagnostic continues.',
        'frames': []}
    path = args.output / 'report.json'
    decoder = None
    try:
        v.write_json(path, report)
        model = owned.load_reference(args.upstream)
        torch_backend = v.TorchBackend(owned.components(model))
        gpu_backend = FixtureBackend(args.fixture)
        states = {'ownedFP32': TemporalState(), 'coremlGPU': TemporalState()}
        history = {'cond_frame_outputs': {}, 'non_cond_frame_outputs': {}}
        previous_empty = None
        decoder = v.video_frames(args.video, count=args.frames)
        with torch.inference_mode():
            for index, timestamp, image, tensor, image_hash in decoder:
                if index == 0 and abs(float(timestamp) - phone['cachedStartSeconds']) > 0.000001:
                    raise ValueError('Phone selection was not on the first decoded frame of this source.')
                if index == 0:
                    selection = image.copy()
                    draw = ImageDraw.Draw(selection)
                    x, y = point
                    draw.ellipse((x - 8, y - 8, x + 8, y + 8), outline='red', width=3)
                    selection.save(args.output / 'initial-selection.png')
                print(f'Occlusion comparison: {index + 1}/{args.frames}', flush=True)
                report['activeFrameIndex'] = index
                # Leave a checkpoint before each potentially long/native operation.
                report['activeStream'] = 'original'
                v.write_json(path, report)
                original = v.reference_step(model, torch.from_numpy(tensor), point, index, history,
                                            num_frames=args.frames)
                report['activeStream'] = 'ownedFP32'
                v.write_json(path, report)
                fp32 = step(torch_backend, states['ownedFP32'], image, tensor, point, timestamp)
                report['activeStream'] = 'coremlGPU'
                v.write_json(path, report)
                gpu = step(gpu_backend, states['coremlGPU'], image, tensor, point, timestamp)
                comparisons = {'ownedFP32AgainstOriginal': v.compare(original, fp32, coreml=False),
                               'coremlGPUAgainstOriginal': v.compare(original, gpu, coreml=True)}
                outputs = {'original': original, 'ownedFP32': fp32, 'coremlGPU': gpu}
                streams = {label: observation(value) for label, value in outputs.items()}
                row = {'index': index, 'ptsNumerator': timestamp.numerator,
                       'ptsDenominator': timestamp.denominator, 'ptsSeconds': float(timestamp),
                       'imageSHA256': image_hash, 'streams': streams, 'comparisons': comparisons,
                       'states': {label: state.summary() for label, state in states.items()}}
                # Original dynamic history remains independent and retains all entries.
                row['originalHistoryEntries'] = {k: len(value) for k, value in history.items()}
                empty = tuple(value['lowMaskEmpty'] for value in streams.values())
                if index % 15 == 0 or index + 1 == args.frames or empty != previous_empty:
                    relative = f'previews/frame-{index:04d}.png'
                    save_preview(args.output / relative, image, outputs, index, float(timestamp))
                    row['preview'] = relative
                previous_empty = empty
                report['frames'].append(row)
                v.write_json(path, report)
        report['completed'] = len(report['frames']) == args.frames
        report['activeStream'] = None
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
