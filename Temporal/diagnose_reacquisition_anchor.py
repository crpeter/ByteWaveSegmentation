#!/usr/bin/env python3
"""Measure an extra reviewed conditioning frame on shared native histories.

Original PyTorch CPU only. Reproduce native and conditioning-only controls from
the completed history probe, then add the native record from reviewed frame60.
All probe outputs are discarded. This measures immediate conditioning effects,
not an independently advancing tracker, automatic anchor choice or device speed.
"""
import argparse
import json
from pathlib import Path
import platform
import subprocess
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np
import torch
import owned
import validate_export as v
from diagnose_occlusion import sha, save_preview
from diagnose_initial_candidate_replay import (
    empty_history, sam_step, initialize, arrays, control, observation)
from diagnose_reacquisition_history import all_true, snapshot, compare_saved

WINDOWS = [(197, 236), (343, 386)]
ANCHOR = 60
NAMES = ['native', 'nativePlusAnchor', 'conditioningOnly', 'conditioningPlusAnchor']
HISTORY_SHA = '012d0cd5b86526e2e212430e2113a63dafcda7489f4ca017a73b3e0ab19f487e'


def extra_anchor(history, conditioning_only):
    # Reuse the actual native memory AND corresponding pointer. No mask/pointer
    # reconstruction, re-encoding, future frame or probe-generated record.
    result = snapshot(history, conditioning_only=conditioning_only)
    record = history['non_cond_frame_outputs'][ANCHOR]
    result['cond_frame_outputs'][ANCHOR] = {
        k: list(value) if isinstance(value, list) else value
        for k, value in record.items()}
    result['non_cond_frame_outputs'].pop(ANCHOR, None)
    return result


def track(model, index, features, positions, sizes, history, total):
    from sam2.modeling.sam2_utils import select_closest_cond_frames
    selected, unused = select_closest_cond_frames(
        index, history['cond_frame_outputs'], model.max_cond_frames_in_attn)
    if set(selected) != set(history['cond_frame_outputs']) or unused:
        raise ValueError('Expected all one or two conditioning records to be selected.')
    if any(t >= index for t in selected):
        raise ValueError('Conditioning frame must precede the measured frame.')
    if set(history['cond_frame_outputs']) & set(history['non_cond_frame_outputs']):
        raise ValueError('Duplicate conditioning/nonconditioning frame.')
    before = {k: tuple(x) for k, x in history.items()}
    calls = []

    def capture(module, args, kwargs):
        calls.append({
            'spatialMemoryRecords': int(kwargs['num_spatial_mem']),
            'objectPointerTokens': int(kwargs['num_obj_ptr_tokens']),
            'memoryShape': list(kwargs['memory'].shape)})

    handle = model.memory_attention.register_forward_pre_hook(capture, with_kwargs=True)
    try:
        output = model.track_step(index, False, features, positions, sizes,
                                  None, None, history, total)
    finally:
        handle.remove()
    if before != {k: tuple(x) for k, x in history.items()} or len(calls) != 1:
        raise ValueError('Unexpected history mutation or attention call count.')
    return output, {'selectedConditioningFrames': sorted(selected),
                    'historyEntries': {k: len(x) for k, x in history.items()},
                    'attentionInput': calls[0]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', type=Path, required=True)
    parser.add_argument('--upstream', type=Path, default=Path('.upstream-edgetam'))
    parser.add_argument('--replay', type=Path, default=Path('.temporal-runs/occlusion-new-whole-person-01/report.json'))
    parser.add_argument('--history-report', type=Path, default=Path('.temporal-runs/occlusion-new-reacquisition-history-01/report.json'))
    parser.add_argument('--output', type=Path, default=Path('.temporal-runs/occlusion-new-reacquisition-anchor-01'))
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    if platform.system() != 'Darwin' or Path.cwd().resolve() != root:
        parser.error(f'Run on your Mac from {root}')
    output = args.output.resolve()
    bundle = root / '.temporal-runs/share' / (output.name + '.zip')
    if not output.is_relative_to((root / '.temporal-runs').resolve()) or output.exists() or bundle.exists():
        parser.error('Choose a fresh output within .temporal-runs; existing results are preserved.')
    prior = json.loads(args.replay.read_text())
    previous = json.loads(args.history_report.read_text())
    if (sha(args.history_report) != HISTORY_SHA
            or previous.get('contract') != 'bytewave.reacquisition-history.diagnostic.v1'
            or previous.get('completed') is not True
            or not all_true(previous.get('controls', {}))
            or previous.get('sourceReplaySHA256') != sha(args.replay)
            or previous.get('scriptSHA256') != sha(Path(__file__).with_name('diagnose_reacquisition_history.py'))
            or previous.get('replayHelperSHA256') != sha(Path(__file__).with_name('diagnose_initial_candidate_replay.py'))):
        raise ValueError('Expected the reviewed completed history report and unchanged helper scripts.')
    if (prior.get('contract') != 'bytewave.initial-candidate-replay.diagnostic.v1'
            or prior.get('completed') is not True or not all_true(prior.get('controls', {}))
            or prior.get('sourceVideoSHA256') != sha(args.video)
            or prior.get('sourceFileBytes') != args.video.stat().st_size
            or prior.get('upstream') != owned.UPSTREAM_REVISION
            or prior.get('checkpointSHA256') != owned.CHECKPOINT_SHA256
            or prior.get('scriptSHA256') != previous['replayHelperSHA256']
            or prior.get('initialCandidate') != 1
            or prior.get('pointNormalizedTopLeft') != [.36, .4]):
        raise ValueError('Expected matching controlled original replay, source and selection.')
    rows = prior['frames']; old_rows = previous['frames']
    total = prior['effectiveFrames']; count = WINDOWS[-1][1] + 1
    if (total != 590 or [r['index'] for r in rows] != list(range(total))
            or [r['index'] for r in old_rows] != list(range(count))):
        raise ValueError('Incomplete source frame sequence.')
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    output.mkdir(parents=True); (output / 'previews').mkdir()
    for name in NAMES: (output / 'masks' / name).mkdir(parents=True)
    report = {
        'scope': __doc__, 'contract': 'bytewave.reacquisition-anchor.diagnostic.v1',
        'completed': False, 'readyForDeviceValidation': False,
        'sourceReplaySHA256': sha(args.replay), 'sourceHistoryReportSHA256': sha(args.history_report),
        'sourceVideoSHA256': sha(args.video), 'sourceFileBytes': args.video.stat().st_size,
        'scriptSHA256': sha(Path(__file__)), 'historyHelperSHA256': previous['scriptSHA256'],
        'replayHelperSHA256': prior['scriptSHA256'], 'upstream': owned.UPSTREAM_REVISION,
        'checkpointSHA256': owned.CHECKPOINT_SHA256, 'torchVersion': torch.__version__,
        'initialCandidate': 1, 'pointNormalizedTopLeft': [.36, .4],
        'referenceNumFrames': total, 'framesToReplay': count, 'windows': WINDOWS,
        'anchorFrame': ANCHOR, 'controls': {}, 'nativeFramesChecked': 0,
        'conditioningControlsChecked': 0, 'frames': [], 'activeFrameIndex': None,
        'policy': 'Only unchanged native outputs advance history. At each inspected frame, fresh probes use native history plus frame60 as conditioning, frame0 only, or frames0+60 only. All probe outputs are discarded.',
        'anchorMeaning': 'Frame60 was visually reviewed before this run. Its native predicted memory/pointer are promoted together to conditioning; no manual pixel labels or new click. Native source history retains frame60 in its original nonconditioning bucket.',
        'interpretation': 'A single extra reference changes memory content, temporal conditioning encoding and token count together. NativePlusAnchor still receives original recent history, including wrong masks. This is an immediate-input experiment, not an independently advancing candidate or automatic reference selection.',
        'deploymentLimit': 'Two conditioning frames can exceed the installed fixed Core ML memory/pointer budget. No export, model promotion, speed or device claim.',
        'review': 'Retrospectively selected development windows. Compare identity, partial visibility and missed targets together; nonempty does not mean correct.'}
    decoder = None
    try:
        model = owned.load_reference(args.upstream)
        report['maxConditioningFramesInAttention'] = model.max_cond_frames_in_attn
        history = empty_history()
        decoder = v.video_frames(args.video, count=count)
        with torch.inference_mode():
            for index, timestamp, image, tensor, image_hash in decoder:
                print(f'Reacquisition anchor: {index+1}/{count}', flush=True)
                report['activeFrameIndex'] = index
                if (image_hash != rows[index]['imageSHA256']
                        or image_hash != old_rows[index]['imageSHA256']
                        or float(timestamp) != rows[index]['ptsSeconds']
                        or float(timestamp) != old_rows[index]['ptsSeconds']):
                    raise ValueError(f'Decoded source differs at {index}.')
                backbone = model.forward_image(owned.normalize(torch.from_numpy(tensor)))
                _, features, positions, sizes = model._prepare_backbone_features(backbone)
                details = {}
                if index == 0:
                    coords = [min(float(x)*1024, 1023) for x in prior['pointNormalizedTopLeft']]
                    prompts = {'point_coords': torch.tensor([[coords]], dtype=torch.float32),
                               'point_labels': torch.ones((1,1), dtype=torch.int32)}
                    auto = sam_step(model, features, positions, sizes, prompts, total)
                    best = int(auto[2][0].argmax().item())
                    forced = sam_step(model, features, positions, sizes, prompts, total, best)
                    chosen = sam_step(model, features, positions, sizes, prompts, total, 0)
                    auto_result = initialize(model, auto, features, sizes, prompts)
                    forced_result = initialize(model, forced, features, sizes, prompts)
                    native_auto = model.track_step(0, True, features, positions, sizes,
                                                  prompts, None, empty_history(), total)
                    report['controls']['initialization'] = {
                        'automaticChoice': best+1 == prior['automaticInitialCandidate'],
                        'scores': bool(np.allclose(auto[2][0].numpy(), prior['initialCandidateScores'], rtol=1e-5, atol=1e-6)),
                        'forcedAutomatic': control(auto_result, forced_result),
                        'nativeAssembly': control(auto_result, native_auto)}
                    if not all_true(report['controls']['initialization']):
                        raise ValueError('Initialization controls failed.')
                    result = initialize(model, chosen, features, sizes, prompts)
                else:
                    result, details['native'] = track(model, index, features, positions, sizes, history, total)
                outputs = {'native': arrays(result)}
                mask_path = args.replay.parent / 'masks/chosen' / f'frame-{index:04d}.png'
                if sha(mask_path) != old_rows[index]['priorMaskSHA256']:
                    raise ValueError('Prior native mask hash changed.')
                checks = compare_saved(outputs['native'], rows[index]['streams']['chosen'], mask_path)
                report['nativeFramesChecked'] += 1
                inspected = any(a <= index <= b for a, b in WINDOWS)
                conditioning_check = None
                if inspected:
                    if index in [a for a, b in WINDOWS]:
                        copied, _ = track(model, index, features, positions, sizes, snapshot(history), total)
                        report['controls'][f'window-{index}-snapshot'] = control(result, copied)
                        if not all_true(report['controls'][f'window-{index}-snapshot']):
                            raise ValueError('No-op snapshot control failed.')
                    probes = {
                        'nativePlusAnchor': extra_anchor(history, False),
                        'conditioningOnly': snapshot(history, conditioning_only=True),
                        'conditioningPlusAnchor': extra_anchor(history, True)}
                    for name, probe in probes.items():
                        candidate, details[name] = track(model, index, features, positions, sizes, probe, total)
                        outputs[name] = arrays(candidate)
                    conditioning_path = args.history_report.parent / 'masks/conditioningOnly' / f'frame-{index:04d}.png'
                    conditioning_check = compare_saved(outputs['conditioningOnly'],
                        old_rows[index]['streams']['conditioningOnly'], conditioning_path)
                    report['conditioningControlsChecked'] += 1
                    # Pinned model uses 6 recent spatial records and 15 recent pointers.
                    # Each pointer splits into four64-channel tokens. The extra frame
                    # is old enough not to occupy a recent slot in either window.
                    expected = {'native': (7,64), 'nativePlusAnchor': (8,68),
                                'conditioningOnly': (1,4), 'conditioningPlusAnchor': (2,8)}
                    for name, (spatial, pointers) in expected.items():
                        actual = details[name]['attentionInput']
                        if (actual['spatialMemoryRecords'], actual['objectPointerTokens']) != (spatial, pointers):
                            raise ValueError(f'Unexpected attention input budget: {name}: {actual}')
                # Commit ONLY original native outputs, after every probe has read
                # the same prior history. Frame60 remains nonconditioning here.
                bucket = 'cond_frame_outputs' if index == 0 else 'non_cond_frame_outputs'
                history[bucket][index] = {k: result[k] for k in ('maskmem_features','maskmem_pos_enc','obj_ptr')}
                row = {'index': index, 'ptsSeconds': float(timestamp), 'imageSHA256': image_hash,
                       'nativeControl': checks, 'priorMaskSHA256': sha(mask_path),
                       'streams': {name: observation(o) for name, o in outputs.items()}}
                if index == ANCHOR:
                    if observation(outputs['native'])['highMaskEmpty']:
                        raise ValueError('Reviewed reference unexpectedly empty.')
                    save_preview(output / 'anchor-0060.jpg', image, outputs, index, float(timestamp))
                    v.save_mask(output / 'anchor-0060.png', outputs['native']['high_res_mask'])
                    report['anchor'] = dict(row)
                if inspected:
                    row['conditioningOnlyControl'] = conditioning_check
                    row['conditioningControlMaskSHA256'] = sha(conditioning_path)
                    row['historyInputs'] = details
                    row['preview'] = f'previews/frame-{index:04d}.jpg'
                    save_preview(output / row['preview'], image, outputs, index, float(timestamp))
                    for name, value in outputs.items():
                        v.save_mask(output / 'masks' / name / f'frame-{index:04d}.png', value['high_res_mask'])
                report['frames'].append(row)
                v.write_json(output / 'report.json', report)
        report['completed'] = (len(report['frames']) == count and report['nativeFramesChecked'] == count
                               and report['conditioningControlsChecked'] == 84 and all_true(report['controls']))
        report['activeFrameIndex'] = None
    except Exception as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        if decoder is not None: decoder.close()
        v.write_json(output / 'report.json', report)
        bundle.parent.mkdir(parents=True, exist_ok=True)
        with ZipFile(bundle, 'x', ZIP_DEFLATED) as archive:
            for path in sorted(output.rglob('*')):
                if path.is_file(): archive.write(path, path.relative_to(output))
        print(f'Upload this ZIP (completed={report["completed"]}): {bundle}', flush=True)
        subprocess.run(['open', '-R', str(bundle)], check=False)


if __name__ == '__main__': main()
