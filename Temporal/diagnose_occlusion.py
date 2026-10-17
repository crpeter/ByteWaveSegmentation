#!/usr/bin/env python3
"""Compare original, owned FP32 and installed Core ML tracking on an occlusion clip.

Run locally on Mac. Independent histories, identical decoded pixels and prompt.
Numerical mismatches are recorded and tracing continues to observe recovery;
shape/nonfinite errors stop immediately. No model export or state-policy change.
Edited clips require --new-video and an explicit --point for inference.
--preview-only saves a tentative selection image without model inference.
--initialization-only compares first-frame masks only; no temporal recovery claim.
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
    point = args.point if args.point is not None else phone.get('lastPointNormalizedTopLeft', [])
    if len(point) != 2 or not all(np.isfinite(x) and 0 <= x <= 1 for x in point):
        raise ValueError('Selection point must contain two finite normalized coordinates in [0,1].')
    if not args.video.is_file():
        raise ValueError(f'Video not found: {args.video.resolve()}')
    if not args.new_video and args.video.stat().st_size != phone.get('sourceFileBytes'):
        raise ValueError('Video byte count differs from phone source. Use the original, or --new-video with an explicit --point for an edited clip.')
    with av.open(str(args.video)) as container:
        stream = container.streams.video[0]
        if not args.new_video and [stream.codec_context.width, stream.codec_context.height] != phone.get('lastDecodedSize'):
            raise ValueError('Source dimensions differ from phone report.')
        duration = float(stream.duration * stream.time_base) if stream.duration is not None else None
        if not args.new_video and (duration is None or abs(duration - phone['videoDurationSeconds']) > 0.1):
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
    return phone, fixture, point


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
    parser.add_argument('--new-video', action='store_true',
                        help='Treat trimmed/re-encoded media as a new experiment, not the phone source.')
    parser.add_argument('--point', type=float, nargs=2, metavar=('X', 'Y'),
                        help='Normalized top-left point on the first displayed frame; required for new-video inference.')
    parser.add_argument('--preview-only', action='store_true',
                        help='Save first-frame selection preview without loading/running models. Without --point the phone point is tentative.')
    parser.add_argument('--initialization-only', action='store_true',
                        help='Run only first-frame selection in all three streams; no temporal propagation.')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if platform.system() != 'Darwin':
        parser.error('Run this diagnostic locally on the Mac.')
    if args.initialization_only and args.preview_only:
        parser.error('--initialization-only and --preview-only are mutually exclusive.')
    if args.initialization_only and (not args.new_video or args.point is None):
        parser.error('--initialization-only requires --new-video and an explicit --point.')
    if not 20 <= args.frames <= 600:
        parser.error('Choose 20 to 600 frames; default 300 covers the worker disappearance/return.')
    if args.new_video and args.point is None and not args.preview_only:
        parser.error('--new-video requires --point X Y on its first frame. Use --preview-only to inspect it.')
    if not args.new_video and args.point is not None:
        parser.error('--point requires --new-video; matched-source mode uses the phone selection.')
    phone, fixture, normalized_point = verify_inputs(args)
    frame_count = 1 if args.initialization_only else args.frames
    if args.new_video and not args.preview_only and not args.initialization_only:
        # Count only the requested prefix before any inference. A shortened clip
        # can end before the requested cap; all three streams use the same count.
        frame_count = 0
        with av.open(str(args.video)) as container:
            for _ in container.decode(container.streams.video[0]):
                frame_count += 1
                if frame_count == args.frames:
                    break
        if frame_count < 20:
            parser.error('The edited clip must contain at least 20 decoded frames.')
        print(f'New-video experiment: comparing {frame_count} frames (cap {args.frames}).', flush=True)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    args.output.mkdir(parents=True, exist_ok=False)
    previews = args.output / 'previews'
    previews.mkdir()
    point = [min(float(x) * 1024, 1023) for x in normalized_point]
    report = {
        'scope': __doc__, 'completed': False, 'readyForDeviceValidation': False,
        'contract': 'bytewave.occlusion-comparison.diagnostic.v1',
        'sourceVideoSHA256': sha(args.video), 'sourceFileBytes': args.video.stat().st_size,
        'phoneReportSHA256': sha(args.phone_report), 'fixtureSHA256': sha(args.fixture / 'fixture.json'),
        'modelFiles': fixture['modelFiles'], 'upstream': owned.UPSTREAM_REVISION,
        'checkpointSHA256': owned.CHECKPOINT_SHA256, 'requestedFrames': args.frames,
        'pointNormalizedTopLeft': normalized_point,
        'sourceMode': 'new-video' if args.new_video else 'phone-source-guards',
        'phoneSourceGuardsApplied': not args.new_video,
        'pointSource': 'explicit' if args.point is not None else 'phone-report',
        'effectiveFrames': frame_count,
        'previewOnly': args.preview_only,
        'initializationOnly': args.initialization_only,
        'inferencePerformed': False,
        'phoneComparisonLimit': 'Phone source has no content hash or saved input pixels. Byte count, dimensions, duration and initial PTS are guards, not proof of identical content. PyAV/Pillow preparation differs from iPhone Core Image. All three Mac streams receive identical pixels; no exact Mac/iPhone parity claim.',
        'memoryPolicy': 'Initial conditioning frame retained; all validated outputs, including absent frames, enter recent memory. No confidence filtering or recovery modification.',
        'qualityMeaning': 'Empty/nonempty transitions do not establish identity. Inspect preview panels. Numerical comparisons use unchanged validate_export gates; failures are retained while this diagnostic continues.',
        'frames': []}
    if args.new_video:
        report['phoneComparisonLimit'] = (
            'Edited/new source: phone video identity, timing and selection equivalence are NOT asserted. '
            'Phone report is used only to identify exact model packages. All three Mac streams '
            'receive the same newly decoded pixels and selection point; previews may use a tentative phone point. No phone parity claim.')
    path = args.output / 'report.json'
    decoder = None
    try:
        v.write_json(path, report)
        if args.preview_only:
            decoder = v.video_frames(args.video, count=1)
            _, timestamp, image, _, image_hash = next(decoder)
            if not args.new_video and abs(float(timestamp) - phone['cachedStartSeconds']) > 0.000001:
                raise ValueError('Phone selection was not on the first decoded frame of this source.')
            selection = image.copy()
            draw = ImageDraw.Draw(selection)
            x, y = point
            draw.ellipse((x - 8, y - 8, x + 8, y + 8), outline='red', width=3)
            selection.save(args.output / 'initial-selection.png')
            report.update(previewSaved=True, firstFrameImageSHA256=image_hash,
                          firstPTSSeconds=float(timestamp),
                          selectionNeedsVisualConfirmation=True)
            print('Preview only: inspect initial-selection.png; no model inference performed.', flush=True)
            return
        model = owned.load_reference(args.upstream)
        torch_backend = v.TorchBackend(owned.components(model))
        gpu_backend = FixtureBackend(args.fixture)
        states = {'ownedFP32': TemporalState(), 'coremlGPU': TemporalState()}
        history = {'cond_frame_outputs': {}, 'non_cond_frame_outputs': {}}
        previous_empty = None
        decoder = v.video_frames(args.video, count=frame_count)
        with torch.inference_mode():
            for index, timestamp, image, tensor, image_hash in decoder:
                if index == 0 and not args.new_video and abs(float(timestamp) - phone['cachedStartSeconds']) > 0.000001:
                    raise ValueError('Phone selection was not on the first decoded frame of this source.')
                if index == 0:
                    selection = image.copy()
                    draw = ImageDraw.Draw(selection)
                    x, y = point
                    draw.ellipse((x - 8, y - 8, x + 8, y + 8), outline='red', width=3)
                    selection.save(args.output / 'initial-selection.png')
                print(f'Occlusion comparison: {index + 1}/{frame_count}', flush=True)
                report['activeFrameIndex'] = index
                # Leave a checkpoint before each potentially long/native operation.
                report['activeStream'] = 'original'
                report['inferencePerformed'] = True
                v.write_json(path, report)
                original = v.reference_step(model, torch.from_numpy(tensor), point, index, history,
                                            num_frames=frame_count)
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
                if index % 15 == 0 or index + 1 == frame_count or empty != previous_empty:
                    relative = f'previews/frame-{index:04d}.png'
                    save_preview(args.output / relative, image, outputs, index, float(timestamp))
                    row['preview'] = relative
                previous_empty = empty
                report['frames'].append(row)
                v.write_json(path, report)
        report['completed'] = len(report['frames']) == frame_count
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
