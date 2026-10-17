#!/usr/bin/env python3
"""Replay automatic versus visually chosen first-frame masks in original EdgeTAM.

Local Mac diagnostic, original PyTorch CPU only. Two independent histories;
identical decoded pixels and point. Only the initial candidate selection differs.
No Core ML export, app change, recovery rule, timing or ground-truth identity claim.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import platform
import subprocess
from zipfile import ZIP_DEFLATED, ZipFile

import av
import numpy as np
from PIL import Image
import torch

import owned
import validate_export as v
from diagnose_occlusion import sha, save_preview


def empty_history():
    return {'cond_frame_outputs': {}, 'non_cond_frame_outputs': {}}


def sam_step(model, features, positions, sizes, prompts, count, candidate=None):
    """Use upstream mask/token selection; a temporary rank override selects a token.

    The hook returns new ranking scores only, never modifies real decoder scores,
    masks, tokens, weights or object logits. Restore genuine quality scores before
    returning. Upstream computes the selected pointer and its absence handling.
    """
    captured = []
    def select(_module, _inputs, result):
        low, scores, tokens, object_score = result
        if (tuple(scores.shape) != (1, 3) or tuple(tokens.shape) != (1, 3, 256)
                or not all(torch.isfinite(x).all().item() for x in result)):
            raise ValueError('Expected three finite masks with corresponding pointer tokens.')
        captured.append(scores.detach().clone())
        ranks = torch.full_like(scores, -1000000.0)
        ranks[:, candidate] = 1000000.0
        return low, ranks, tokens, object_score
    hook = model.sam_mask_decoder.register_forward_hook(select) if candidate is not None else None
    try:
        _, sam, _, _ = model._track_step(
            0, True, features, positions, sizes, prompts, None,
            empty_history(), count, False, None)
    finally:
        if hook is not None:
            hook.remove()
    if candidate is not None:
        if len(captured) != 1:
            raise ValueError('Expected exactly one decoder call for initial selection.')
        sam = (*sam[:2], captured[0], *sam[3:])
    if (tuple(sam[0].shape) != (1, 3, 256, 256)
            or tuple(sam[1].shape) != (1, 3, 1024, 1024)
            or tuple(sam[2].shape) != (1, 3)
            or not all(torch.isfinite(x).all().item() for x in sam)):
        raise ValueError('Unexpected initial SAM output.')
    index = int(sam[2][0].argmax().item()) if candidate is None else candidate
    if (not torch.equal(sam[0][:, index:index + 1], sam[3])
            or not torch.equal(sam[1][:, index:index + 1], sam[4])):
        raise ValueError('Selected mask does not match requested candidate.')
    return sam


def initialize(model, sam, features, sizes, prompts):
    result = {'pred_masks': sam[3], 'pred_masks_high_res': sam[4],
              'obj_ptr': sam[5], 'object_score_logits': sam[6]}
    model._encode_memory_in_output(features, sizes, prompts, True, sam[4], sam[6], result)
    return result


def arrays(result):
    values = {'low_res_mask': result['pred_masks'],
              'high_res_mask': result['pred_masks_high_res'],
              'object_pointer': result['obj_ptr'], 'object_score': result['object_score_logits'],
              'memory_features': result['maskmem_features'],
              'memory_positions': result['maskmem_pos_enc'][0]}
    output = {}
    for name, value in values.items():
        value = value.detach().cpu().numpy()
        if value.shape != v.SHAPES[name] or not np.isfinite(value).all():
            raise ValueError(f'Invalid output {name}: {value.shape}')
        output[name] = value
    return output


def control(left, right):
    a, b = arrays(left), arrays(right)
    checks = {name: bool(np.allclose(a[name], b[name], rtol=0.001, atol=0.001))
              for name in a}
    for name in ('low_res_mask', 'high_res_mask'):
        checks[name + 'BinaryExact'] = bool(np.array_equal(a[name] > 0, b[name] > 0))
    return checks


def observation(output):
    low, high = output['low_res_mask'] > 0, output['high_res_mask'] > 0
    return {'lowMaskEmpty': not bool(low.any()), 'highMaskEmpty': not bool(high.any()),
            'lowForegroundFraction': float(low.mean()), 'highForegroundFraction': float(high.mean()),
            'objectScore': float(output['object_score'].item())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', type=Path, required=True)
    parser.add_argument('--upstream', type=Path, default=Path('.upstream-edgetam'))
    parser.add_argument('--alternatives', type=Path,
                        default=Path('.temporal-runs/occlusion-new-mask-alternatives-01/report.json'))
    parser.add_argument('--point-number', type=int, default=2, help='One-based reviewed point.')
    parser.add_argument('--candidate', type=int, choices=(1, 2, 3), default=1)
    parser.add_argument('--frames', type=int, default=600)
    parser.add_argument('--output', type=Path,
                        default=Path('.temporal-runs/occlusion-new-whole-person-01'))
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    if platform.system() != 'Darwin' or Path.cwd().resolve() != root:
        parser.error(f'Run on your Mac from {root}')
    if not 20 <= args.frames <= 600:
        parser.error('Choose 20 to 600 frames.')
    output = args.output.resolve()
    bundle = root / '.temporal-runs/share' / (output.name + '.zip')
    if not output.is_relative_to((root / '.temporal-runs').resolve()):
        parser.error('Output must be within .temporal-runs.')
    if output.exists() or bundle.exists():
        parser.error('Output or ZIP already exists; choose a fresh --output.')
    prior = json.loads(args.alternatives.read_text())
    source_hash = sha(args.video)
    if (prior.get('contract') != 'bytewave.initial-mask-alternatives.diagnostic.v1'
            or prior.get('completed') is not True or prior.get('framesDecoded') != 1
            or prior.get('sourceVideoSHA256') != source_hash
            or prior.get('sourceFileBytes') != args.video.stat().st_size
            or prior.get('upstream') != owned.UPSTREAM_REVISION
            or prior.get('checkpointSHA256') != owned.CHECKPOINT_SHA256):
        raise ValueError('Expected completed alternatives report for this exact video and model.')
    if not 1 <= args.point_number <= len(prior['points']):
        parser.error('Point number is outside alternatives report.')
    selected = prior['points'][args.point_number - 1]
    if not all(selected['priorSummaryControl'].values()):
        raise ValueError('Prior initial-selection controls did not reproduce.')
    point = selected['pointNormalizedTopLeft']
    if len(point) != 2 or not all(np.isfinite(x) and 0 <= x <= 1 for x in point):
        raise ValueError('Invalid selection point.')
    mask_path = (args.alternatives.parent / selected['candidates'][args.candidate - 1]['mask']).resolve()
    if not mask_path.is_relative_to(args.alternatives.parent.resolve()):
        raise ValueError('Mask path escapes alternatives directory.')
    with Image.open(mask_path) as mask_image:
        saved_mask = np.asarray(mask_image.convert('L')) > 0
    if saved_mask.shape != (1024, 1024):
        raise ValueError('Expected 1024-square reviewed mask.')
    with av.open(str(args.video)) as container:
        count = 0
        for _ in container.decode(container.streams.video[0]):
            count += 1
            if count == args.frames:
                break
    if count < 20:
        raise ValueError('Need at least 20 decoded frames.')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    output.mkdir(parents=True)
    (output / 'previews').mkdir()
    for name in ('automatic', 'chosen'):
        (output / 'masks' / name).mkdir(parents=True)
    report = {
        'scope': __doc__, 'contract': 'bytewave.initial-candidate-replay.diagnostic.v1',
        'completed': False, 'readyForDeviceValidation': False,
        'sourceVideoSHA256': source_hash, 'sourceFileBytes': args.video.stat().st_size,
        'alternativesReportSHA256': sha(args.alternatives), 'reviewedMaskSHA256': sha(mask_path),
        'scriptSHA256': sha(Path(__file__)), 'upstream': owned.UPSTREAM_REVISION,
        'checkpointSHA256': owned.CHECKPOINT_SHA256, 'torchVersion': torch.__version__,
        'pointNormalizedTopLeft': point, 'pointNumber': args.point_number,
        'initialCandidate': args.candidate, 'effectiveFrames': count,
        'historyPolicy': 'Independent original histories. Initial conditioning retained; all subsequent outputs, including absence, enter history. No backup/recovery logic.',
        'selectionPolicy': 'Frame 0 only: native candidate/token selection with temporary rank override. Genuine quality scores restored. Later frames use unchanged native track_step.',
        'qualityMeaning': 'Empty/nonempty is not identity. Different initial targets; no parity assertion between streams.',
        'previewPolicy': 'Every 15th frame, final frame and low-mask empty/nonempty transitions. Binary high masks saved for every frame.',
        'controls': {}, 'frames': []}
    decoder = None
    try:
        v.write_json(output / 'report.json', report)
        model = owned.load_reference(args.upstream)
        histories = {name: empty_history() for name in ('automatic', 'chosen')}
        previous_empty = None
        decoder = v.video_frames(args.video, count=count)
        with torch.inference_mode():
            for index, timestamp, image, tensor, image_hash in decoder:
                print(f'Whole-person replay: {index + 1}/{count}', flush=True)
                report['activeFrameIndex'] = index
                v.write_json(output / 'report.json', report)
                backbone = model.forward_image(owned.normalize(torch.from_numpy(tensor)))
                _, features, positions, sizes = model._prepare_backbone_features(backbone)
                if index == 0:
                    if image_hash != prior['imageSHA256'] or float(timestamp) != prior['ptsSeconds']:
                        raise ValueError('First decoded frame differs from reviewed alternatives.')
                    coords = [min(float(x) * 1024, 1023) for x in point]
                    prompts = {'point_coords': torch.tensor([[coords]], dtype=torch.float32),
                               'point_labels': torch.ones((1, 1), dtype=torch.int32)}
                    auto = sam_step(model, features, positions, sizes, prompts, count)
                    best = int(auto[2][0].argmax().item())
                    report['automaticInitialCandidate'] = best + 1
                    report['initialCandidateScores'] = auto[2][0].tolist()
                    scores_match = np.allclose(auto[2][0].numpy(),
                                              [c['predictedIoU'] for c in selected['candidates']],
                                              rtol=1e-5, atol=1e-6)
                    if best + 1 != selected['selectedCandidate'] or not scores_match:
                        raise ValueError('Prior alternative scores/automatic choice did not reproduce.')
                    control_sam = sam_step(model, features, positions, sizes, prompts, count, best)
                    chosen = sam_step(model, features, positions, sizes, prompts, count, args.candidate - 1)
                    if not np.array_equal(chosen[4][0, 0].numpy() > 0, saved_mask):
                        raise ValueError('Chosen initial mask does not reproduce reviewed mask.')
                    results = {
                        'automatic': initialize(model, auto, features, sizes, prompts),
                        'chosen': initialize(model, chosen, features, sizes, prompts)}
                    forced_auto = initialize(model, control_sam, features, sizes, prompts)
                    native_auto = model.track_step(
                        0, True, features, positions, sizes, prompts, None, empty_history(), count)
                    checks = {
                        'reviewedMaskReproduced': True, 'priorScoresReproduced': True,
                        'forceAutomaticAgainstUnchanged': control(results['automatic'], forced_auto),
                        'initialAssemblyAgainstNative': control(results['automatic'], native_auto)}
                    report['controls'] = checks
                    if not all(checks['forceAutomaticAgainstUnchanged'].values()) or not all(checks['initialAssemblyAgainstNative'].values()):
                        raise ValueError('Initial mask/pointer/memory control failed.')
                else:
                    results = {
                        name: model.track_step(index, False, features, positions, sizes,
                                               None, None, history, count)
                        for name, history in histories.items()}
                outputs = {name: arrays(result) for name, result in results.items()}
                rows = {name: observation(value) for name, value in outputs.items()}
                for name, result in results.items():
                    bucket = 'cond_frame_outputs' if index == 0 else 'non_cond_frame_outputs'
                    histories[name][bucket][index] = {
                        key: result[key] for key in ('maskmem_features', 'maskmem_pos_enc', 'obj_ptr')}
                    v.save_mask(output / 'masks' / name / f'frame-{index:04d}.png', outputs[name]['high_res_mask'])
                empty = tuple(row['lowMaskEmpty'] for row in rows.values())
                preview = None
                if index % 15 == 0 or index == count - 1 or empty != previous_empty:
                    preview = f'previews/frame-{index:04d}.png'
                    save_preview(output / preview, image, outputs, index, float(timestamp))
                previous_empty = empty
                report['frames'].append({'index': index, 'ptsSeconds': float(timestamp),
                                         'imageSHA256': image_hash, 'streams': rows, 'preview': preview})
                v.write_json(output / 'report.json', report)
        report['summary'] = {name: {
            'emptyFrames': sum(r['streams'][name]['lowMaskEmpty'] for r in report['frames']),
            'framesProcessed': len(report['frames'])} for name in histories}
        report.update(completed=len(report['frames']) == count, activeFrameIndex=None)
    except Exception as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        if decoder is not None:
            decoder.close()
        v.write_json(output / 'report.json', report)
        bundle.parent.mkdir(parents=True, exist_ok=True)
        with ZipFile(bundle, 'x', ZIP_DEFLATED) as archive:
            for path in sorted(output.rglob('*')):
                if path.is_file():
                    archive.write(path, path.relative_to(output))
        print(f'Upload this ZIP (completed={report["completed"]}): {bundle}', flush=True)
        subprocess.run(['open', '-R', str(bundle)], check=False)


if __name__ == '__main__':
    main()
