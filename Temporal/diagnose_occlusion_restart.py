#!/usr/bin/env python3
"""Check fresh segmentation and tracking at a visible reappearance landmark.

Start original PyTorch and installed GPU with empty histories and a NEW explicit
point at the chosen source frame. Compare their fresh outputs with unchanged
numerical gates, and record prior ongoing-track observations as context only.
This tests prompted segmentation ability, not automatic identity recovery.
No model conversion, export, memory-policy change or app modification.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import platform

import numpy as np
from PIL import ImageDraw
import torch

import diagnose_occlusion as occlusion
from diagnose_occlusion_inputs import tensor_hashes
import owned
import validate_export as v
from state import TemporalState


def summarize(report):
    rows = report['frames']
    return {'framesCompared': len(rows),
            'freshGPUParityAllPassed': bool(rows) and report['completed']
                and all(row['freshGPUAgainstFreshOriginal']['passed'] for row in rows),
            'failedSourceFrames': [row['sourceFrame'] for row in rows
                if not row['freshGPUAgainstFreshOriginal']['passed']],
            'nonemptySourceFrames': {label: [row['sourceFrame'] for row in rows
                if not row['observations'][label]['lowMaskEmpty']]
                for label in ('freshOriginal', 'freshGPU', 'priorOriginal', 'priorGPU')},
            'firstFrame': rows[0] if rows else None,
            'qualityMeaning': 'A nonempty mask must be visually checked against the intended worker. Fresh prompting is not automatic recovery; prior observations have different prompts and histories.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('upstream', 'video', 'phone-report', 'fixture', 'comparison', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--start-frame', type=int, required=True)
    parser.add_argument('--point', type=float, nargs=2, required=True,
                        help='New normalized top-left point in the decoded source frame.')
    parser.add_argument('--frames', type=int, default=31)
    args = parser.parse_args()
    if platform.system() != 'Darwin':
        parser.error('Run locally on the Mac.')
    prior = json.loads(args.comparison.read_text())
    if (prior.get('contract') != 'bytewave.occlusion-comparison.diagnostic.v1'
            or prior.get('completed') is not True or prior.get('previewOnly') is not False
            or prior.get('inferencePerformed') is not True):
        parser.error('Expected completed original occlusion comparison report.')
    total = prior['effectiveFrames']
    previous_rows = prior['frames']
    if (len(previous_rows) != total
            or [row['index'] for row in previous_rows] != list(range(total))
            or not 0 < args.start_frame < total or not 1 <= args.frames <= 600
            or args.start_frame + args.frames > total):
        parser.error('Restart range must fit inside the completed comparison.')
    args.new_video = True
    _, fixture, _ = occlusion.verify_inputs(args)
    identity = {'sourceVideoSHA256': occlusion.sha(args.video),
                'sourceFileBytes': args.video.stat().st_size,
                'phoneReportSHA256': occlusion.sha(args.phone_report),
                'fixtureSHA256': occlusion.sha(args.fixture / 'fixture.json'),
                'modelFiles': fixture['modelFiles'], 'upstream': owned.UPSTREAM_REVISION,
                'checkpointSHA256': owned.CHECKPOINT_SHA256}
    for key, value in identity.items():
        if prior.get(key) != value:
            raise ValueError(f'Prior comparison identity mismatch: {key}')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'previews').mkdir()
    path = args.output / 'report.json'
    report = {'scope': __doc__, 'contract': 'bytewave.occlusion-restart.diagnostic.v1',
              'completed': False, 'readyForDeviceValidation': False, **identity,
              'sourceComparisonSHA256': occlusion.sha(args.comparison),
              'startSourceFrame': args.start_frame, 'requestedFrames': args.frames,
              'freshReferenceNumFrames': total - args.start_frame,
              'pointNormalizedTopLeft': args.point,
              'priorPointNormalizedTopLeft': prior['pointNormalizedTopLeft'],
              'computeUnits': 'CPU_AND_GPU',
              'policy': {'maskIoUMinimum': .95, 'cosineMinimum': .99,
                         'float32Atol': .001, 'float32Rtol': .001},
              'torchThreads': {'intraOp': 1, 'interOp': 1},
              'versions': {name: importlib.metadata.version(name)
                           for name in ('torch', 'coremltools', 'numpy', 'av', 'Pillow')},
              'memoryPolicy': 'Fresh independent histories, local indices start at zero. Original reference has dynamic history; GPU uses unchanged bounded state. Original source PTS retained. No prior tracking memory copied.',
              'priorComparisonLimit': 'Stored ongoing-track observations only; ongoing models are not rerun here and have a different original point/history. Numerical parity compares only the two fresh streams.',
              'frames': [], 'activeSourceFrame': None, 'activeOperation': None}
    decoder = None

    def checkpoint(label):
        report['activeOperation'] = label
        v.write_json(path, report)

    try:
        checkpoint('load original reference')
        model = owned.load_reference(args.upstream)
        checkpoint('load exact installed GPU packages')
        backend = occlusion.FixtureBackend(args.fixture)
        state = TemporalState()
        history = {'cond_frame_outputs': {}, 'non_cond_frame_outputs': {}}
        point = [min(float(x) * 1024, 1023) for x in args.point]
        decoder = v.video_frames(args.video, count=args.start_frame + args.frames)
        with torch.inference_mode():
            for index, timestamp, image, tensor, digest in decoder:
                previous = previous_rows[index]
                if (digest != previous['imageSHA256'] or timestamp.numerator != previous['ptsNumerator']
                        or timestamp.denominator != previous['ptsDenominator']):
                    raise ValueError(f'Decoded pixels/PTS differ at source frame {index}.')
                if index < args.start_frame:
                    continue
                local = index - args.start_frame
                report['activeSourceFrame'] = index
                print(f'Fresh restart: {local + 1}/{args.frames} (source frame {index})', flush=True)
                if local == 0:
                    marked = image.copy()
                    draw = ImageDraw.Draw(marked)
                    x, y = point
                    draw.ellipse((x - 7, y - 7, x + 7, y + 7), outline='red', width=2)
                    marked.save(args.output / 'initial-selection.png')
                checkpoint('fresh original reference')
                original = v.reference_step(model, torch.from_numpy(tensor), point, local,
                    history, num_frames=total - args.start_frame)
                tensor_hashes(original, v.NAMES['Propagator'])
                checkpoint('fresh installed GPU')
                actual = occlusion.step(backend, state, image, tensor, point, timestamp)
                tensor_hashes(actual, v.NAMES['Propagator'])
                row = {'sourceFrame': index, 'localFrame': local,
                       'ptsSeconds': float(timestamp), 'ptsNumerator': timestamp.numerator,
                       'ptsDenominator': timestamp.denominator, 'imageSHA256': digest,
                       'freshGPUAgainstFreshOriginal': v.compare(original, actual, coreml=True),
                       'observations': {'freshOriginal': occlusion.observation(original),
                                        'freshGPU': occlusion.observation(actual),
                                        'priorOriginal': previous['streams']['original'],
                                        'priorGPU': previous['streams']['coremlGPU']},
                       'gpuState': state.summary()}
                relative = f'previews/frame-{index:04d}.png'
                occlusion.save_preview(args.output / relative, image,
                    {'fresh original': original, 'fresh GPU': actual}, index, float(timestamp))
                row['preview'] = relative
                report['frames'].append(row)
                v.write_json(path, report)
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
