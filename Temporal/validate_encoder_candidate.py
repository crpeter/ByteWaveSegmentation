#!/usr/bin/env python3
"""Replay 20 saved frames against original PyTorch with a diagnostic encoder.

Run candidate encoder on CPU and CPU_AND_NE separately; tracker stays CPU_ONLY.
Reuses the original comparison gates and bounded state. No export or promotion.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np
from PIL import Image
import torch

import owned
import validate_export as v
from diagnose_encoder_prefix import file_hashes
from inspect_encoder_placement import sha, write_json
from state import TemporalState


def verify(args):
    manifest_path = args.run / 'models/manifest.json'
    manifest = json.loads(manifest_path.read_text())
    status = json.loads((args.run / 'status.json').read_text())
    source = json.loads((args.run / 'coreml/report.json').read_text())
    diagnostic_path = args.candidate / 'report.json'
    diagnostic = json.loads(diagnostic_path.read_text())
    for item in (manifest, status):
        if (item.get('contract') != owned.CONTRACT or item.get('graphRevision') != owned.GRAPH_REVISION
                or item.get('precisionPolicy') != v.COREML_PRECISION_POLICY):
            raise ValueError('Source models must match current graph and precision policy')
    if not all(status.get(key) is True for key in ('referencePassed', 'coremlPassed', 'readyForDeviceValidation')):
        raise ValueError('Source run must have passed normal validation')
    if source.get('passed') is not True or len(source['frames']) != v.COUNT:
        raise ValueError('Expected complete 20-frame source comparison')
    for component, details in manifest['models'].items():
        if file_hashes(args.run / f'models/BWTemporal{component}.mlpackage') != details['files']:
            raise ValueError(f'Source {component} bytes changed')
    if diagnostic.get('completed') is not True or diagnostic.get('sourceManifestSHA256') != sha(manifest_path):
        raise ValueError('Candidate diagnostic must match source models')
    candidate = diagnostic['variants'][args.variant]
    if any(candidate.get(mode, {}).get('passed') is not True for mode in ('cpu', 'ne')):
        raise ValueError('Candidate must pass the one-frame CPU and NE encoder gates first')
    if file_hashes(args.candidate / f'{args.variant}.mlpackage') != candidate['files']:
        raise ValueError('Candidate encoder bytes changed')
    point = status['pointNormalizedTopLeft']
    if len(point) != 2 or not all(np.isfinite(x) and 0 <= x <= 1 for x in point):
        raise ValueError('Invalid original prompt coordinates')
    return status, source, diagnostic


class CandidateBackend(v.CoreMLBackend):
    def __init__(self, args, report, report_path):
        import coremltools as ct
        super().__init__(args.run / 'models', graph_revision=owned.GRAPH_REVISION)
        self.models['ImageEncoder'] = ct.models.MLModel(str(args.candidate / f'{args.variant}.mlpackage'),
                                                       compute_units=getattr(ct.ComputeUnit, args.encoder_units))
        model = self.models['ImageEncoder']
        if model.user_defined_metadata.get('bytewave.contract') != 'bytewave.encoder-residuals.diagnostic.v1':
            raise ValueError('Unexpected candidate encoder contract')
        features = model.get_spec().description.input
        if len(features) != 1 or features[0].name != 'image':
            raise ValueError('Unexpected candidate input names')
        self.input_kind = features[0].type.WhichOneof('Type')
        self.report, self.report_path = report, report_path

    def call(self, component, inputs):
        self.report['activeComponent'] = component
        write_json(self.report_path, self.report)
        if component != 'ImageEncoder':
            return super().call(component, inputs)
        image = inputs['pil_image']
        if self.input_kind == 'imageType':
            value = image
        elif self.input_kind == 'multiArrayType':
            # Serialized preprocessing already divides raw RGB bytes by 255.
            value = np.asarray(image, dtype=np.float32).transpose(2, 0, 1)[None]
        else:
            raise ValueError('Unsupported candidate encoder image interface')
        output = self.models[component].predict({'image': value})
        if set(output) != set(owned.IMAGE_OUTPUTS):
            raise ValueError('Unexpected candidate encoder output names')
        result = {name: np.array(output[name], dtype=np.float32, copy=True) for name in owned.IMAGE_OUTPUTS}
        for name, value in result.items():
            if value.shape != v.SHAPES[name] or not np.isfinite(value).all():
                raise ValueError(f'Invalid candidate ImageEncoder output: {name}')
        return result


def worker(args, status, source):
    args.output.mkdir(parents=True, exist_ok=False)
    path = args.output / 'report.json'
    report = {'passed': False, 'completed': False, 'readyForDeviceValidation': False,
              'contract': owned.CONTRACT + '.diagnostic', 'reference': 'Pinned original PyTorch; independent history',
              'sourceManifestSHA256': sha(args.run / 'models/manifest.json'),
              'candidateReportSHA256': sha(args.candidate / 'report.json'), 'candidateVariant': args.variant,
              'upstream': owned.UPSTREAM_REVISION, 'checkpointSHA256': owned.CHECKPOINT_SHA256,
              'encoderComputeUnits': args.encoder_units, 'trackerComputeUnits': 'CPU_ONLY',
              'pointNormalizedTopLeft': status['pointNormalizedTopLeft'],
              'torchThreads': {'intraOp': torch.get_num_threads(), 'interOp': torch.get_num_interop_threads()},
              'policy': {'maskIoUMinimum': .95, 'cosineMinimum': .99, 'float32Atol': .001, 'float32Rtol': .001},
              'frames': []}
    try:
        write_json(path, report)
        model = owned.load_reference(args.upstream)
        backend = CandidateBackend(args, report, path)
        state = TemporalState()
        history = {'cond_frame_outputs': {}, 'non_cond_frame_outputs': {}}
        point = [min(max(float(x) * 1024, 0), 1023) for x in status['pointNormalizedTopLeft']]
        with torch.inference_mode():
            for index, saved in enumerate(source['frames']):
                report['activeFrameIndex'] = index
                write_json(path, report)
                print(f'{args.encoder_units}: frame {index + 1}/{v.COUNT}', flush=True)
                with Image.open(args.run / f'coreml/frame-{index:02d}.png') as image:
                    image = image.convert('RGB')
                    pixels = np.asarray(image).copy()
                digest = hashlib.sha256(pixels.tobytes()).hexdigest()
                if saved['index'] != index or pixels.shape != (1024, 1024, 3) or digest != saved['imageSHA256']:
                    raise ValueError('Saved frame pixels or index changed')
                timestamp = Fraction(saved['ptsNumerator'], saved['ptsDenominator'])
                state.validate_time(timestamp)
                token = state.token
                features = backend.call('ImageEncoder', {'pil_image': image})
                if index == 0:
                    result = backend.call('Initializer', {**features,
                                          'point_coords': np.array([[point]], dtype=np.float32),
                                          'point_labels': np.ones((1, 1), dtype=np.int32)})
                    result.update(backend.call('InitialMemoryEncoder', {**features, **result}))
                else:
                    result = backend.call('Propagator', {**features, **state.pack(timestamp)})
                report['activeComponent'] = 'OriginalPyTorchReference'
                write_json(path, report)
                tensor = pixels.astype(np.float32).transpose(2, 0, 1)[None] / 255.0
                reference = v.reference_step(model, torch.from_numpy(tensor), point, index, history)
                comparison = v.compare(reference, result, coreml=True)
                if index == 0 and not (reference['low_res_mask'] > 0).any():
                    raise ValueError('Original reference mask is empty')
                state.commit(result, timestamp, token)
                report['frames'].append({'index': index, 'imageSHA256': digest,
                                         'ptsNumerator': timestamp.numerator, 'ptsDenominator': timestamp.denominator,
                                         'state': state.summary(), **comparison})
                v.save_mask(args.output / f'candidate-{index:02d}.png', result['low_res_mask'])
                v.save_mask(args.output / f'reference-{index:02d}.png', reference['low_res_mask'])
                write_json(path, report)
                if not comparison['passed']:
                    raise ValueError(f'Temporal parity failed at frame {index}')
        if len(report['frames']) != v.COUNT or state.summary()['spatialEntries'] != 7 or state.summary()['pointerEntries'] != 16:
            raise ValueError('Full temporal banks were not exercised')
        report.update(passed=True, completed=True)
        report.pop('activeComponent', None)
        report.pop('activeFrameIndex', None)
    except Exception as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        write_json(path, report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--variant', default='fanout', choices=('single', 'matching', 'fanout'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--encoder-units', choices=('CPU_ONLY', 'CPU_AND_NE'), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if platform.system() != 'Darwin':
        parser.error('Run on Mac')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    status, source, diagnostic = verify(args)
    if args.encoder_units:
        return worker(args, status, source)
    args.output.mkdir(parents=True, exist_ok=False)
    path = args.output / 'report.json'
    report = {'scope': __doc__, 'completed': False, 'passed': False, 'readyForDeviceValidation': False,
              'sourceManifestSHA256': diagnostic['sourceManifestSHA256'],
              'candidateReportSHA256': sha(args.candidate / 'report.json'), 'candidateVariant': args.variant,
              'sourceFrameReportSHA256': sha(args.run / 'coreml/report.json'),
              'pointNormalizedTopLeft': status['pointNormalizedTopLeft'], 'modes': {}}
    for label, units in (('cpu', 'CPU_ONLY'), ('ne', 'CPU_AND_NE')):
        report['activeMode'] = label
        write_json(path, report)
        log = args.output / f'{label}.log'
        command = [sys.executable, str(Path(__file__).resolve()), '--upstream', str(args.upstream.resolve()),
                   '--run', str(args.run.resolve()), '--candidate', str(args.candidate.resolve()),
                   '--variant', args.variant, '--output', str((args.output / label).resolve()), '--encoder-units', units]
        print(f'Running 20-frame comparison: encoder {units}, tracker CPU_ONLY...', flush=True)
        with log.open('w') as stream:
            try:
                process = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, timeout=900, check=False)
                row = {'returnCode': process.returncode, 'log': log.name}
            except subprocess.TimeoutExpired:
                row = {'error': 'Worker timed out', 'log': log.name}
        child = args.output / label / 'report.json'
        details = json.loads(child.read_text()) if child.exists() else {}
        frames = details.get('frames', [])
        row.update(passed=bool(row.get('returnCode') == 0 and details.get('passed') is True),
                   framesCompared=len(frames), report=f'{label}/report.json')
        if frames:
            row['minimumMaskIoU'] = min(f['maskIoUAgainstReference'] for f in frames)
            row['minimumPointerCosine'] = min(f['outputs']['object_pointer']['cosineSimilarity'] for f in frames)
            row['minimumMemoryCosine'] = min(f['outputs']['memory_features']['cosineSimilarity'] for f in frames)
            row['finalState'] = frames[-1]['state']
            if not row['passed']:
                row['lastComparedFrame'] = frames[-1]
        for key in ('error', 'activeFrameIndex', 'activeComponent'):
            if key in details:
                row[key] = details[key]
        report['modes'][label] = row
        write_json(path, report)
        print(f'{label}: passed={row["passed"]}, frames={len(frames)}; see {child}', flush=True)
    report['completed'] = True
    report['passed'] = all(row['passed'] for row in report['modes'].values())
    report.pop('activeMode', None)
    write_json(path, report)
    print(f'Report: {path}', flush=True)


if __name__ == '__main__':
    main()
