#!/usr/bin/env python3
"""Compare native and non-updating history probes around reviewed reacquisitions.

Original PyTorch CPU diagnostic. Native chosen-mask history must reproduce the
saved replay. In two fixed windows, compare an immutable pre-window history and
initial-conditioning-only history; neither probe ever commits its own outputs.
No automatic identity decision, production recovery rule, export or speed claim.
"""
import argparse
import json
from pathlib import Path
import platform
import subprocess
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np
from PIL import Image
import torch
import owned
import validate_export as v
from diagnose_occlusion import sha, save_preview
from diagnose_initial_candidate_replay import (
    empty_history, sam_step, initialize, arrays, control, observation)

WINDOWS = [(197, 236), (343, 386)]


def all_true(value):
    return all(all_true(x) for x in value.values()) if isinstance(value, dict) and value else value is True


def snapshot(history, conditioning_only=False):
    # Pinned native track_step reads input records; it does not modify their tensors.
    # Separate dictionary/list containers; feature tensors shared read-only, as in
    # the controlled candidate replay. Never add probe outputs to these histories.
    return {bucket: {} if conditioning_only and bucket == 'non_cond_frame_outputs' else
            {i: {k: list(value) if isinstance(value, list) else value
                 for k, value in record.items()} for i, record in records.items()}
            for bucket, records in history.items()}


def compare_saved(output, old, mask_path):
    with Image.open(mask_path) as image:
        pixels = np.asarray(image)
        if image.mode != 'L' or pixels.shape != (1024, 1024) or not np.isin(pixels, [0,255]).all():
            raise ValueError('Invalid prior binary mask.')
    obs = observation(output)
    checks = {k: obs[k] == old[k] for k in obs if k != 'objectScore'}
    checks['objectScoreClose'] = bool(np.isclose(obs['objectScore'], old['objectScore'], rtol=1e-5, atol=1e-6))
    checks['savedHighMaskExact'] = bool(np.array_equal(output['high_res_mask'][0,0] > 0, pixels > 0))
    if not all(checks.values()):
        raise ValueError(f'Native replay control failed: {checks}')
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', type=Path, required=True)
    parser.add_argument('--upstream', type=Path, default=Path('.upstream-edgetam'))
    parser.add_argument('--replay', type=Path, default=Path('.temporal-runs/occlusion-new-whole-person-01/report.json'))
    parser.add_argument('--output', type=Path, default=Path('.temporal-runs/occlusion-new-reacquisition-history-01'))
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    if platform.system() != 'Darwin' or Path.cwd().resolve() != root:
        parser.error(f'Run on your Mac from {root}')
    output = args.output.resolve()
    bundle = root / '.temporal-runs/share' / (output.name + '.zip')
    if not output.is_relative_to((root / '.temporal-runs').resolve()) or output.exists() or bundle.exists():
        parser.error('Choose a fresh output within .temporal-runs; existing ZIPs are preserved.')
    prior = json.loads(args.replay.read_text())
    if (prior.get('contract') != 'bytewave.initial-candidate-replay.diagnostic.v1'
            or prior.get('completed') is not True or not all_true(prior.get('controls', {}))
            or prior.get('sourceVideoSHA256') != sha(args.video)
            or prior.get('sourceFileBytes') != args.video.stat().st_size
            or prior.get('upstream') != owned.UPSTREAM_REVISION
            or prior.get('checkpointSHA256') != owned.CHECKPOINT_SHA256
            or prior.get('scriptSHA256') != sha(Path(__file__).with_name('diagnose_initial_candidate_replay.py'))):
        raise ValueError('Expected controlled matching native chosen-mask replay.')
    # These windows are reviewed development cases, not automatic event discovery.
    if (prior.get('sourceVideoSHA256') != '5170ab7d885e131cbcd79be302d422f58e014f0c7ffbde51dcdcf8ad27bfa1e9'
            or prior.get('initialCandidate') != 1 or prior.get('pointNormalizedTopLeft') != [.36, .4]):
        raise ValueError('This fixed-window diagnostic requires the reviewed whole-person clip/selection.')
    rows = prior['frames']; total = prior['effectiveFrames']; count = WINDOWS[-1][1]+1
    if total < count or [x['index'] for x in rows] != list(range(total)):
        raise ValueError('Incomplete prior frame sequence.')
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    output.mkdir(parents=True); (output/'previews').mkdir()
    names = ['native', 'frozenPreWindow', 'conditioningOnly']
    for name in names: (output/'masks'/name).mkdir(parents=True)
    report = {'scope': __doc__, 'contract': 'bytewave.reacquisition-history.diagnostic.v1',
              'completed': False, 'readyForDeviceValidation': False,
              'sourceReplaySHA256': sha(args.replay), 'sourceVideoSHA256': sha(args.video),
              'sourceFileBytes': args.video.stat().st_size,
              'scriptSHA256': sha(Path(__file__)), 'replayHelperSHA256': prior['scriptSHA256'],
              'upstream': owned.UPSTREAM_REVISION, 'checkpointSHA256': owned.CHECKPOINT_SHA256,
              'torchVersion': torch.__version__, 'pointNormalizedTopLeft': prior['pointNormalizedTopLeft'],
              'initialCandidate': prior['initialCandidate'], 'referenceNumFrames': total,
              'framesToReplay': count, 'windows': WINDOWS, 'controls': {}, 'nativeFramesChecked': 0,
              'policy': 'Native owns unchanged history. Both probes read fixed snapshots; their new memory and pointers are discarded every frame. All variants share read-only original encoder features.',
              'interpretation': 'Frozen-pre-window and native are identical at window start. Later differences measure the combined effect of withheld updates and aging recent slots, not solely bad-mask contamination. Conditioning-only also removes earlier nonconditioning evidence. No automatic acceptance, no candidate history or identity guarantee.',
              'review': 'Windows chosen retrospectively: correct shoe return197 and wrong-hand return343. These are development cases, not held-out performance. Previews show every window frame.',
              'frames': [], 'activeFrameIndex': None}
    decoder = None
    try:
        model = owned.load_reference(args.upstream)
        history = empty_history(); probes = {}; window_end = None
        decoder = v.video_frames(args.video, count=count)
        with torch.inference_mode():
            for index, timestamp, image, tensor, image_hash in decoder:
                print(f'Reacquisition history: {index+1}/{count}', flush=True)
                report['activeFrameIndex'] = index
                if image_hash != rows[index]['imageSHA256'] or float(timestamp) != rows[index]['ptsSeconds']:
                    raise ValueError(f'Decoded source differs at {index}.')
                backbone = model.forward_image(owned.normalize(torch.from_numpy(tensor)))
                _, features, positions, sizes = model._prepare_backbone_features(backbone)
                if index == 0:
                    point = prior['pointNormalizedTopLeft']
                    coords = [min(float(x)*1024,1023) for x in point]
                    prompts = {'point_coords': torch.tensor([[coords]],dtype=torch.float32),
                               'point_labels': torch.ones((1,1),dtype=torch.int32)}
                    auto = sam_step(model, features, positions, sizes, prompts, total)
                    best = int(auto[2][0].argmax().item())
                    forced = sam_step(model, features, positions, sizes, prompts, total, best)
                    chosen = sam_step(model, features, positions, sizes, prompts, total, prior['initialCandidate']-1)
                    auto_result = initialize(model, auto, features, sizes, prompts)
                    forced_result = initialize(model, forced, features, sizes, prompts)
                    native_auto = model.track_step(0, True, features, positions, sizes, prompts, None, empty_history(), total)
                    report['controls']['initialization'] = {
                        'automaticChoice': best+1 == prior['automaticInitialCandidate'],
                        'scores': bool(np.allclose(auto[2][0].numpy(),prior['initialCandidateScores'],rtol=1e-5,atol=1e-6)),
                        'forcedAutomatic': control(auto_result, forced_result),
                        'nativeAssembly': control(auto_result, native_auto)}
                    if not all_true(report['controls']['initialization']):
                        raise ValueError('Initialization controls failed.')
                    result = initialize(model, chosen, features, sizes, prompts)
                else:
                    for start, end in WINDOWS:
                        if index == start:
                            probes = {'frozenPreWindow': snapshot(history),
                                      'conditioningOnly': snapshot(history, conditioning_only=True)}
                            window_end = end
                    result = model.track_step(index, False, features, positions, sizes, None, None, history, total)
                outputs = {'native': arrays(result)}
                mask_path = args.replay.parent/'masks/chosen'/f'frame-{index:04d}.png'
                checks = compare_saved(outputs['native'], rows[index]['streams']['chosen'], mask_path)
                report['nativeFramesChecked'] += 1
                if probes:
                    for name, probe in probes.items():
                        before = {k: tuple(x.keys()) for k,x in probe.items()}
                        probe_result = model.track_step(index, False, features, positions, sizes, None, None, probe, total)
                        if before != {k: tuple(x.keys()) for k,x in probe.items()}:
                            raise ValueError('Native call mutated a probe history container.')
                        outputs[name] = arrays(probe_result)
                        if name == 'frozenPreWindow' and index in [w[0] for w in WINDOWS]:
                            c = control(result,probe_result)
                            report['controls'][f'window-{index}-same-input'] = c
                            if not all(c.values()): raise ValueError('Window fork control failed.')
                bucket = 'cond_frame_outputs' if index == 0 else 'non_cond_frame_outputs'
                history[bucket][index] = {k: result[k] for k in ('maskmem_features','maskmem_pos_enc','obj_ptr')}
                row = {'index':index,'ptsSeconds':float(timestamp),'imageSHA256':image_hash,
                       'nativeControl':checks,'priorMaskSHA256':sha(mask_path),
                       'streams':{name:observation(o) for name,o in outputs.items()}}
                if probes:
                    preview = f'previews/frame-{index:04d}.jpg'
                    save_preview(output/preview,image,outputs,index,float(timestamp))
                    row['preview'] = preview
                    row['probeHistoryEntries'] = {name:{k:len(x) for k,x in h.items()} for name,h in probes.items()}
                    for name,o in outputs.items():
                        f = output/'masks'/name/f'frame-{index:04d}.png'
                        v.save_mask(f,o['high_res_mask'])
                    if index == window_end: probes = {}
                report['frames'].append(row)
                v.write_json(output/'report.json',report)
        report['completed'] = report['nativeFramesChecked'] == count and len(report['frames']) == count and all_true(report['controls'])
        report['activeFrameIndex'] = None
    except Exception as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        if decoder is not None: decoder.close()
        v.write_json(output/'report.json',report)
        bundle.parent.mkdir(parents=True,exist_ok=True)
        with ZipFile(bundle,'x',ZIP_DEFLATED) as archive:
            for path in sorted(output.rglob('*')):
                if path.is_file(): archive.write(path,path.relative_to(output))
        print(f'Upload this ZIP (completed={report["completed"]}): {bundle}',flush=True)
        subprocess.run(['open','-R',str(bundle)],check=False)


if __name__ == '__main__': main()
