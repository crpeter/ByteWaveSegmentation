#!/usr/bin/env python3
"""Track backward from a reviewed later mask and compare saved forward masks.

Original PyTorch CPU only. Frame 375's saved prediction is a manually reviewed
experimental mask prompt, not ground truth or an automatically selected anchor.
Rebuild its pointer/memory using upstream's mask-input path, then advance a fresh
reverse history through frame 165. A relabeled same-input forward call checks
reverse indexing at every step; its outputs never enter history. No mask fusion,
identity gate, app/export change, device timing or independent accuracy claim.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
from PIL import Image
import torch
import owned
import validate_export as v
from diagnose_occlusion import sha, save_preview
from diagnose_initial_candidate_replay import empty_history, arrays, control, observation
from diagnose_reacquisition_history import all_true, snapshot

SEED = 375
LAST = 165
TOTAL = 590
REPLAY_SHA = '5f9378e84e1b78ca19e00574245434e07180065dbcd608bdb34051338156cc56'
ANCHOR_REPORT_SHA = 'ce4496788c83d981cf4c5391b9ae5cd6ecad4b8f1715db4887633715d478c6ce'
SEED_MASK_SHA = '8f5c205d46986543443bcefbb951dc6a8b90c4076dabbfa6501c4573703fed9d'
VIDEO_SHA = '5170ab7d885e131cbcd79be302d422f58e014f0c7ffbde51dcdcf8ad27bfa1e9'


def read_mask(path):
    with Image.open(path) as image:
        pixels = np.array(image)
        if image.mode != 'L' or pixels.shape != (1024, 1024) or not np.isin(pixels, [0, 255]).all():
            raise ValueError(f'Invalid saved binary mask: {path}')
    return pixels > 0


def record(result):
    return {k: result[k] for k in ('maskmem_features', 'maskmem_pos_enc', 'obj_ptr')}


def relabel(history):
    # Separate containers, shared read-only tensors. The seed becomes 0; each
    # decreasing source index becomes an increasing index in the reversed clip.
    return {bucket: {SEED - index: value for index, value in records.items()}
            for bucket, records in snapshot(history).items()}


def tracked(model, index, features, positions, sizes, history, reverse):
    anchor = SEED if reverse else 0
    step = SEED - index if reverse else index
    expected_recent = set(range(index + 1, SEED)) if reverse else set(range(1, index))
    if (set(history['cond_frame_outputs']) != {anchor}
            or set(history['non_cond_frame_outputs']) != expected_recent or step < 1):
        raise ValueError('History does not match an uninterrupted single-direction rollout.')
    before = {k: tuple(values) for k, values in history.items()}
    calls = []

    def capture(module, args, kwargs):
        calls.append({'spatialMemoryRecords': int(kwargs['num_spatial_mem']),
                      'objectPointerTokens': int(kwargs['num_obj_ptr_tokens']),
                      'memoryShape': list(kwargs['memory'].shape)})

    handle = model.memory_attention.register_forward_pre_hook(capture, with_kwargs=True)
    try:
        result = model.track_step(index, False, features, positions, sizes,
                                  None, None, history, TOTAL, track_in_reverse=reverse)
    finally:
        handle.remove()
    if before != {k: tuple(values) for k, values in history.items()} or len(calls) != 1:
        raise ValueError('Unexpected history mutation or memory-attention call count.')
    spatial = 1 + min(6, step - 1)
    pointers = 4 * (1 + min(15, step - 1))
    expected = {'spatialMemoryRecords': spatial, 'objectPointerTokens': pointers,
                'memoryShape': [spatial * 512 + pointers, 1, 64]}
    if calls[0] != expected:
        raise ValueError(f'Unexpected directional memory budget: {calls[0]} != {expected}')
    return result, calls[0]


def compare_masks(forward, backward):
    overlap = int(np.logical_and(forward, backward).sum())
    union = int(np.logical_or(forward, backward).sum())
    return {'intersectionPixels': overlap, 'unionPixels': union,
            'iou': overlap / union if union else None,
            'forwardOnlyPixels': int(np.logical_and(forward, ~backward).sum()),
            'backwardOnlyPixels': int(np.logical_and(backward, ~forward).sum()),
            'bothEmpty': union == 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', type=Path, required=True)
    parser.add_argument('--upstream', type=Path, default=Path('.upstream-edgetam'))
    parser.add_argument('--replay', type=Path, default=Path('.temporal-runs/occlusion-new-whole-person-01/report.json'))
    parser.add_argument('--anchor-report', type=Path, default=Path('.temporal-runs/occlusion-new-reacquisition-anchor-01/report.json'))
    parser.add_argument('--output', type=Path, default=Path('.temporal-runs/occlusion-new-backward-01'))
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    if platform.system() != 'Darwin' or Path.cwd().resolve() != root:
        parser.error(f'Run on your Mac from {root}')
    output = args.output.resolve()
    runs = (root / '.temporal-runs').resolve()
    bundle = runs / 'share' / (output.name + '.zip')
    if not output.is_relative_to(runs) or output.exists() or bundle.exists():
        parser.error('Choose a fresh output within .temporal-runs; existing results are preserved.')
    if sha(args.replay) != REPLAY_SHA or sha(args.anchor_report) != ANCHOR_REPORT_SHA:
        raise ValueError('Expected the exact reviewed whole-person and extra-reference reports.')
    prior = json.loads(args.replay.read_text())
    previous = json.loads(args.anchor_report.read_text())
    if (prior.get('completed') is not True or previous.get('completed') is not True
            or not all_true(prior.get('controls', {})) or not all_true(previous.get('controls', {}))
            or previous['sourceReplaySHA256'] != REPLAY_SHA
            or prior['sourceVideoSHA256'] != VIDEO_SHA or sha(args.video) != VIDEO_SHA
            or args.video.stat().st_size != prior['sourceFileBytes']
            or prior['effectiveFrames'] != TOTAL or prior['initialCandidate'] != 1
            or prior['pointNormalizedTopLeft'] != [.36, .4]
            or prior['upstream'] != owned.UPSTREAM_REVISION
            or prior['checkpointSHA256'] != owned.CHECKPOINT_SHA256
            or prior['scriptSHA256'] != sha(Path(__file__).with_name('diagnose_initial_candidate_replay.py'))
            or previous['historyHelperSHA256'] != sha(Path(__file__).with_name('diagnose_reacquisition_history.py'))):
        raise ValueError('Source, initialization, helper or model provenance mismatch.')
    rows, reviewed = prior['frames'], previous['frames']
    if ([r['index'] for r in rows] != list(range(TOTAL))
            or [r['index'] for r in reviewed] != list(range(387))):
        raise ValueError('Incomplete prior frame sequence.')
    mask_dir = args.replay.parent / 'masks/chosen'
    # Check every saved forward mask before model loading. The latest reviewed
    # report binds its exact PNG bytes to the reproduced native forward stream.
    for index in range(LAST, SEED + 1):
        path = mask_dir / f'frame-{index:04d}.png'
        mask = read_mask(path)
        old = rows[index]['streams']['chosen']
        if (sha(path) != reviewed[index]['priorMaskSHA256']
                or not all_true(reviewed[index]['nativeControl'])
                or reviewed[index]['imageSHA256'] != rows[index]['imageSHA256']
                or reviewed[index]['ptsSeconds'] != rows[index]['ptsSeconds']
                or bool(mask.any()) == old['highMaskEmpty']
                or float(mask.mean()) != old['highForegroundFraction']):
            raise ValueError(f'Saved forward mask/metadata mismatch at {index}.')
    seed_path = mask_dir / f'frame-{SEED:04d}.png'
    seed = read_mask(seed_path)
    if sha(seed_path) != SEED_MASK_SHA or not seed.any():
        raise ValueError('Reviewed later-frame seed mask changed or is empty.')
    # Cache RGB bytes, never float tensors/features or the entire clip in RAM.
    # About 633 MiB for 211 frames, plus room for outputs; removed after the run.
    cache_bytes = (SEED - LAST + 1) * 1024 * 1024 * 3
    if shutil.disk_usage(runs).free < cache_bytes + 512 * 1024 * 1024:
        raise ValueError('Need about 1.2 GiB free under .temporal-runs for this diagnostic.')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    output.mkdir(parents=True)
    (output / 'previews').mkdir()
    for name in ('savedForward', 'backward'):
        (output / 'masks' / name).mkdir(parents=True)
    helpers = ['owned.py', 'validate_export.py', 'diagnose_occlusion.py',
               'diagnose_initial_candidate_replay.py', 'diagnose_reacquisition_history.py']
    report = {
        'scope': __doc__, 'contract': 'bytewave.backward-reacquisition.diagnostic.v1',
        'completed': False, 'readyForDeviceValidation': False,
        'sourceReplaySHA256': REPLAY_SHA, 'sourceAnchorReportSHA256': ANCHOR_REPORT_SHA,
        'sourceVideoSHA256': VIDEO_SHA, 'sourceFileBytes': args.video.stat().st_size,
        'scriptSHA256': sha(Path(__file__)),
        'helperSHA256': {name: sha(Path(__file__).with_name(name)) for name in helpers},
        'upstream': owned.UPSTREAM_REVISION, 'checkpointSHA256': owned.CHECKPOINT_SHA256,
        'torchVersion': torch.__version__, 'referenceNumFrames': TOTAL,
        'seedFrame': SEED, 'lastFrame': LAST, 'expectedFrames': SEED - LAST + 1,
        'seedMaskSHA256': SEED_MASK_SHA, 'controls': {}, 'decodedFramesChecked': 0,
        'cachedFrames': 0, 'reverseIndexControlsChecked': 0,
        'activeFrameIndex': None, 'activeOperation': None, 'frames': [],
        'seedPolicy': 'Use reviewed saved forward binary mask 375 as an explicit mask prompt. Native mask-input path generates a fresh corresponding object pointer and memory from frame 375 pixels; no forward-history tensor is reused. The input mask is preserved as output logits (-10/+10), and its nonempty object logit (+10) is forced, not predicted identity confidence.',
        'historyPolicy': 'One conditioning record at 375. Every backward output, including empty/wrong outputs, advances only reverse history. At each step a fresh relabeling maps source t to 375-t for a same-input track_in_reverse=False control; control outputs are discarded.',
        'comparisonPolicy': 'Compare against hash-verified SAVED original forward predictions seeded with candidate 1 at frame 0. Forward inference is not rerun. Seeds and available temporal evidence differ; this is a practical later-reference experiment, not an isolated causal estimate of direction alone.',
        'controlMeaning': 'Relabeled same-input outputs check reverse temporal indexing under pinned stride 1/settings. They do not validate identity or independently reproduce upstream algorithms.',
        'identityLimit': 'Seed is a visually reviewed model prediction, not pixel ground truth. Later-frame selection is manual development supervision. Forward and backward evidence is correlated; agreement, presence, and overlap do not establish identity. No automatic seed choice, acceptance threshold, mask intersection/union, or production replacement.',
        'review': 'Every frame 165..375 has two full-resolution overlays and both binary masks. Inspect genuine partials, wrong foreground people/hands, complete absence, and missed visible targets. Empty/empty overlap is null, not success.',
        'runtimeScope': 'Original PyTorch CPU on Mac only. No Core ML, export, installed fixture, app/recovery policy, speed measurement or iPhone parity claim. Temporary lossless RGB cache is deleted and excluded from ZIP.'}
    try:
        v.write_json(output / 'report.json', report)
        with TemporaryDirectory(prefix='decoded-', dir=output) as cache_name:
            cache = Path(cache_name)
            decoder = v.video_frames(args.video, count=SEED + 1)
            report['activeOperation'] = 'decode-and-verify'
            try:
                for index, timestamp, image, _, image_hash in decoder:
                    report['activeFrameIndex'] = index
                    if image_hash != rows[index]['imageSHA256'] or float(timestamp) != rows[index]['ptsSeconds']:
                        raise ValueError(f'Decoded source differs at frame {index}.')
                    report['decodedFramesChecked'] += 1
                    if index >= LAST:
                        pixels = np.array(image)
                        np.save(cache / f'{index:04d}.npy', pixels, allow_pickle=False)
                        report['cachedFrames'] += 1
                    if index % 30 == 0 or index == SEED:
                        print(f'Verifying video: {index + 1}/{SEED + 1}', flush=True)
                        v.write_json(output / 'report.json', report)
            finally:
                decoder.close()
            if report['decodedFramesChecked'] != SEED + 1 or report['cachedFrames'] != SEED - LAST + 1:
                raise ValueError('Incomplete decoded cache.')
            report['activeOperation'] = 'load-reference'
            v.write_json(output / 'report.json', report)
            model = owned.load_reference(args.upstream)
            if not model.use_mask_input_as_output_without_sam or not model.only_obj_ptrs_in_the_past_for_eval:
                raise ValueError('Expected native direct-mask initialization and directional pointer selection.')
            history = empty_history()
            with torch.inference_mode():
                for index in range(SEED, LAST - 1, -1):
                    step = SEED - index
                    print(f'Backward tracking: {step + 1}/{SEED - LAST + 1} (source frame {index})', flush=True)
                    report.update(activeFrameIndex=index, activeOperation='backward-and-index-control')
                    v.write_json(output / 'report.json', report)
                    pixels = np.load(cache / f'{index:04d}.npy', allow_pickle=False)
                    if (pixels.shape != (1024, 1024, 3) or pixels.dtype != np.uint8
                            or hashlib.sha256(pixels.tobytes()).hexdigest() != rows[index]['imageSHA256']):
                        raise ValueError('Lossless decoded cache failed source hash check.')
                    image = Image.fromarray(pixels)
                    tensor = pixels.astype(np.float32).transpose(2, 0, 1)[None] / 255.0
                    backbone = model.forward_image(owned.normalize(torch.from_numpy(tensor)))
                    _, features, positions, sizes = model._prepare_backbone_features(backbone)
                    if index == SEED:
                        mask_input = torch.from_numpy(seed.astype(np.float32)[None, None])
                        result = model.track_step(SEED, True, features, positions, sizes,
                            None, mask_input, history, TOTAL, track_in_reverse=True)
                        check_result = model.track_step(0, True, features, positions, sizes,
                            None, mask_input, empty_history(), TOTAL, track_in_reverse=False)
                        checks = control(result, check_result)
                        checks['inputHighMaskExact'] = bool(np.array_equal(arrays(result)['high_res_mask'][0, 0] > 0, seed))
                        checks['inputMaskUnmodified'] = bool(np.array_equal(mask_input.numpy()[0, 0], seed.astype(np.float32)))
                        checks['seedHistoryInitiallyEmpty'] = history == empty_history()
                        report['controls']['maskPromptInitialization'] = checks
                        budget = None
                    else:
                        result, budget = tracked(model, index, features, positions, sizes, history, True)
                        check_result, relabeled_budget = tracked(model, step, features, positions, sizes,
                                                                 relabel(history), False)
                        checks = control(result, check_result)
                        checks['attentionBudgetEqual'] = budget == relabeled_budget
                    if not all_true(checks):
                        report['failedControl'] = {'index': index, 'checks': checks}
                        raise ValueError(f'Reverse indexing/initialization control failed at frame {index}.')
                    report['reverseIndexControlsChecked'] += 1
                    value = arrays(result)
                    forward_path = mask_dir / f'frame-{index:04d}.png'
                    if sha(forward_path) != reviewed[index]['priorMaskSHA256']:
                        raise ValueError('Saved forward mask changed during the run.')
                    forward = read_mask(forward_path)
                    backward = value['high_res_mask'][0, 0] > 0
                    # History receives ONLY this reverse result, with its matching
                    # mask-derived pointer and memory. No forward/control outputs.
                    bucket = 'cond_frame_outputs' if index == SEED else 'non_cond_frame_outputs'
                    history[bucket][index] = record(result)
                    preview = f'previews/frame-{index:04d}.jpg'
                    save_preview(output / preview, image, {
                        'savedForward': {'high_res_mask': forward[None, None]},
                        'backwardFrom375': value}, index, rows[index]['ptsSeconds'])
                    mask_files = {}
                    for name, path in [('savedForward', forward_path), ('backward', None)]:
                        dest = output / 'masks' / name / f'frame-{index:04d}.png'
                        if path is None:
                            v.save_mask(dest, value['high_res_mask'])
                        else:
                            shutil.copyfile(path, dest)
                        mask_files[name] = {'path': str(dest.relative_to(output)), 'sha256': sha(dest)}
                    report['frames'].append({
                        'index': index, 'processingOrdinal': step, 'ptsSeconds': rows[index]['ptsSeconds'],
                        'imageSHA256': rows[index]['imageSHA256'], 'preview': preview,
                        'sourceForwardMaskSHA256': reviewed[index]['priorMaskSHA256'],
                        'streams': {'savedForward': rows[index]['streams']['chosen'], 'backward': observation(value)},
                        'directionIndexControl': checks, 'attentionInput': budget,
                        'highMaskAgreement': compare_masks(forward, backward), 'maskFiles': mask_files})
                    v.write_json(output / 'report.json', report)
        report['completed'] = (len(report['frames']) == SEED - LAST + 1
            and report['reverseIndexControlsChecked'] == SEED - LAST + 1 and all_true(report['controls']))
        report.update(activeFrameIndex=None, activeOperation=None)
        report['summary'] = {
            'comparedFrames': len(report['frames']),
            'backwardNonemptyFrames': sum(not r['streams']['backward']['highMaskEmpty'] for r in report['frames']),
            'savedForwardNonemptyFrames': sum(not r['streams']['savedForward']['highMaskEmpty'] for r in report['frames']),
            'meaning': 'Presence counts are descriptive only. Review masks for identity; no accuracy or recovery success score.'}
    except Exception as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        v.write_json(output / 'report.json', report)
        (output / 'README.txt').write_text(
            'Upload this one ZIP. report.json contains completion status and controls.\n'
            'Previews: left=saved forward, right=backward from reviewed frame 375.\n'
            'Files sort chronologically; backward inference ran 375 down to 165.\n'
            'The seed is a predicted mask, not ground truth. No masks are merged.\n')
        bundle.parent.mkdir(parents=True, exist_ok=True)
        with ZipFile(bundle, 'x', ZIP_DEFLATED) as archive:
            # Explicit roots keep temporary RGB cache out even on partial failure.
            for path in [output / 'report.json', output / 'README.txt']:
                archive.write(path, path.relative_to(output))
            for directory in ('previews', 'masks'):
                for path in sorted((output / directory).rglob('*')):
                    if path.is_file():
                        archive.write(path, path.relative_to(output))
        print(f'Upload this ZIP (completed={report["completed"]}): {bundle}', flush=True)
        subprocess.run(['open', '-R', str(bundle)], check=False)


if __name__ == '__main__':
    main()
