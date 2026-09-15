#!/usr/bin/env python3
"""Test two equivalent 1x1-convolution replacements on the saved 20-frame fixture.

Unchanged reconversion control on CPU, then candidate CPU/GPU/NE propagation.
Other components stay on CPU. Original PyTorch gates and bounded state apply.
Diagnostic only: no normal export, fixture installation or device readiness.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

import numpy as np
from PIL import Image
import torch

import owned
import validate_export as v
from inspect_encoder_placement import sha, write_json
from propagator_convs import CONTRACT, make_variant
from state import TemporalState


def file_hashes(package):
    return {str(p.relative_to(package)): sha(p) for p in package.rglob('*') if p.is_file()}


def verify(args):
    manifest_path = args.run / 'models/manifest.json'
    manifest = json.loads(manifest_path.read_text())
    status = json.loads((args.run / 'status.json').read_text())
    source = json.loads((args.run / 'coreml/report.json').read_text())
    inspection = (json.loads((args.inspection / 'report.json').read_text())
                  if args.inspection is not None else None)
    for item in (status, manifest):
        if (item.get('contract') != owned.CONTRACT or item.get('graphRevision') != owned.GRAPH_REVISION
                or item.get('precisionPolicy') != v.COREML_PRECISION_POLICY):
            raise ValueError('Expected current validated fanout model set')
    if not all(status.get(k) is True for k in ('referencePassed', 'coremlPassed', 'readyForDeviceValidation')):
        raise ValueError('Source normal validation did not pass')
    if (source.get('passed') is not True or len(source['frames']) != v.COUNT
            or any(f.get('passed') is not True for f in source['frames'])):
        raise ValueError('Expected the complete passing 20-frame source report')
    if set(manifest['models']) != set(v.INPUTS):
        raise ValueError('Source model set is incomplete')
    for component, entry in manifest['models'].items():
        if file_hashes(args.run / f'models/BWTemporal{component}.mlpackage') != entry['files']:
            raise ValueError(f'{component} package bytes changed')
    if inspection is not None and (
            inspection.get('completed') is not True or inspection.get('sourceManifestSHA256') != sha(manifest_path)
            or inspection.get('files') != manifest['models']['Propagator']['files']):
        raise ValueError('Inspection does not match the source propagator')
    point = status['pointNormalizedTopLeft']
    if len(point) != 2 or not all(np.isfinite(n) and 0 <= n <= 1 for n in point):
        raise ValueError('Invalid original point')
    return status, source


class Backend(v.CoreMLBackend):
    def __init__(self, args, report, report_path, contract=CONTRACT):
        import coremltools as ct
        super().__init__(args.run / 'models', graph_revision=owned.GRAPH_REVISION)
        self.original_propagator = self.models['Propagator'] if args.variant == 'unchanged' else None
        package = args.candidate / f'{args.variant}.mlpackage'
        variants = json.loads((args.candidate / 'variants.json').read_text())
        if variants['sourceManifestSHA256'] != sha(args.run / 'models/manifest.json'):
            raise ValueError('Diagnostic source manifest mismatch')
        if file_hashes(package) != variants['variants'][args.variant]['files']:
            raise ValueError('Diagnostic model bytes changed')
        model = ct.models.MLModel(str(package), compute_units=getattr(ct.ComputeUnit, args.units))
        metadata = model.user_defined_metadata
        if metadata.get('bytewave.contract') != contract or metadata.get('bytewave.diagnostic.variant') != args.variant:
            raise ValueError('Unexpected diagnostic contract/variant')
        self.models['Propagator'] = model
        self.report, self.path = report, report_path
        self.last_seconds = None
        self.last_control = None

    def call(self, component, inputs):
        self.report['activeComponent'] = component
        write_json(self.path, self.report)
        if component != 'Propagator':
            return super().call(component, inputs)
        values = {name: np.ascontiguousarray(inputs[name], dtype=np.float32) for name in v.INPUTS[component]}
        start = time.perf_counter()
        output = self.models[component].predict(values)
        self.last_seconds = time.perf_counter() - start
        if set(output) != set(v.NAMES[component]):
            raise ValueError('Diagnostic output names changed')
        result = {n: np.array(output[n], dtype=np.float32, copy=True) for n in v.NAMES[component]}
        for name, value in result.items():
            if value.shape != v.SHAPES[name] or not np.isfinite(value).all():
                raise ValueError(f'Invalid diagnostic output: {name}')
        if self.original_propagator is not None:
            self.report['activeComponent'] = 'OriginalCPUPropagatorControl'
            write_json(self.path, self.report)
            original = self.original_propagator.predict(values)
            self.last_control = v.compare(original, result, coreml=True)
            self.last_control['allOutputsFloat32Close'] = all(
                row['float32Close'] for row in self.last_control['outputs'].values())
            self.report['latestSameInputControl'] = self.last_control
            if not self.last_control['passed'] or not self.last_control['allOutputsFloat32Close']:
                raise ValueError('Unchanged reconversion differs from original CPU on identical inputs')
        return result


def worker(args, status, source, contract=CONTRACT, scope=__doc__):
    args.output.mkdir(parents=True, exist_ok=False)
    path = args.output / 'report.json'
    report = {'scope': scope, 'passed': False, 'completed': False, 'readyForDeviceValidation': False,
              'contract': contract, 'variant': args.variant, 'propagatorComputeUnits': args.units,
              'otherComponentsComputeUnits': 'CPU_ONLY', 'reference': 'Pinned original PyTorch; independent history',
              'sourceManifestSHA256': sha(args.run / 'models/manifest.json'),
              'variantsSHA256': sha(args.candidate / 'variants.json'),
              'upstream': owned.UPSTREAM_REVISION, 'checkpointSHA256': owned.CHECKPOINT_SHA256,
              'pointNormalizedTopLeft': status['pointNormalizedTopLeft'],
              'torchThreads': {'intraOp': torch.get_num_threads(), 'interOp': torch.get_num_interop_threads()},
              'policy': {'maskIoUMinimum': .95, 'cosineMinimum': .99, 'float32Atol': .001, 'float32Rtol': .001},
              'timingScope': 'Mac predict wall time only, including Python/Core ML bridge; first propagator call is cold. '
                             'No paired GPU/NE speed benchmark, iPhone timing or sustained playback claim.',
              'frames': []}
    try:
        report['activeComponent'] = 'LoadModels'
        write_json(path, report)
        model = owned.load_reference(args.upstream)
        backend = Backend(args, report, path, contract=contract)
        state = TemporalState()
        history = {'cond_frame_outputs': {}, 'non_cond_frame_outputs': {}}
        point = [min(max(float(n) * 1024, 0), 1023) for n in status['pointNormalizedTopLeft']]
        with torch.inference_mode():
            for index, saved in enumerate(source['frames']):
                report['activeFrameIndex'] = index
                write_json(path, report)
                print(f'{args.variant}/{args.units}: frame {index + 1}/{v.COUNT}', flush=True)
                with Image.open(args.run / f'coreml/frame-{index:02d}.png') as image:
                    image = image.convert('RGB')
                    pixels = np.asarray(image).copy()
                digest = hashlib.sha256(pixels.tobytes()).hexdigest()
                if saved['index'] != index or pixels.shape != (1024, 1024, 3) or digest != saved['imageSHA256']:
                    raise ValueError('Saved frame changed')
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
                row = {'index': index, 'imageSHA256': digest, 'ptsNumerator': timestamp.numerator,
                       'ptsDenominator': timestamp.denominator, **comparison}
                if index:
                    row['propagatorWallSecondsDiagnosticOnly'] = backend.last_seconds
                    if backend.last_control is not None:
                        row['sameInputControl'] = backend.last_control
                if comparison['passed']:
                    state.commit(result, timestamp, token)
                row['state'] = state.summary()
                report['frames'].append(row)
                v.save_mask(args.output / f'candidate-{index:02d}.png', result['low_res_mask'])
                v.save_mask(args.output / f'reference-{index:02d}.png', reference['low_res_mask'])
                write_json(path, report)
                if not comparison['passed']:
                    raise ValueError(f'Temporal parity failed at frame {index}')
        if state.summary() != {'acceptedFrames': 20, 'spatialEntries': 7, 'pointerEntries': 16}:
            raise ValueError('Full bounded temporal state was not exercised')
        report.update(passed=True, completed=True)
        report.pop('activeComponent', None)
        report.pop('activeFrameIndex', None)
    except Exception as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        write_json(path, report)


def run_worker(args, label, variant, units, script_path=None):
    log = args.output / f'{label}.log'
    command = [sys.executable, str(Path(script_path or __file__).resolve()), '--upstream', str(args.upstream.resolve()),
               '--run', str(args.run.resolve()),
               '--output', str((args.output / label).resolve()), '--candidate', str(args.output.resolve()),
               '--variant', variant, '--units', units]
    if args.inspection is not None:
        command.extend(['--inspection', str(args.inspection.resolve())])
    print(f'Comparing {label}: 20 frames against original PyTorch...', flush=True)
    with log.open('w') as stream:
        try:
            process = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, timeout=1200, check=False)
            row = {'returnCode': process.returncode, 'log': log.name}
        except subprocess.TimeoutExpired:
            row = {'error': 'Worker timed out', 'log': log.name}
    child = args.output / label / 'report.json'
    details = json.loads(child.read_text()) if child.exists() else {}
    frames = details.get('frames', [])
    row.update(passed=bool(row.get('returnCode') == 0 and details.get('passed') is True and len(frames) == v.COUNT),
               framesCompared=len(frames), report=f'{label}/report.json')
    if frames:
        row.update(minimumMaskIoU=min(f['maskIoUAgainstReference'] for f in frames),
                   minimumPointerCosine=min(f['outputs']['object_pointer']['cosineSimilarity'] for f in frames),
                   minimumMemoryCosine=min(f['outputs']['memory_features']['cosineSimilarity'] for f in frames),
                   finalState=frames[-1]['state'])
        # Frame 0 initializes; frame 1 is the first (cold) propagator call.
        warm = [f['propagatorWallSecondsDiagnosticOnly'] for f in frames if f['index'] >= 2]
        if warm:
            row['medianWarmPropagatorMillisecondsDiagnosticOnly'] = 1000 * statistics.median(warm)
    for key in ('error', 'activeFrameIndex', 'activeComponent', 'latestSameInputControl'):
        if key in details:
            row[key] = details[key]
    print(f'{label}: passed={row["passed"]}; report: {child}', flush=True)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--inspection', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--variant', choices=('unchanged', 'conv2'), help=argparse.SUPPRESS)
    parser.add_argument('--units', choices=('CPU_ONLY', 'CPU_AND_GPU', 'CPU_AND_NE'), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if platform.system() != 'Darwin':
        parser.error('Run on Mac')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    status, source = verify(args)
    if args.units:
        if args.candidate is None or args.variant is None:
            parser.error('Worker requires candidate and variant')
        return worker(args, status, source)
    args.output.mkdir(parents=True, exist_ok=False)
    path = args.output / 'report.json'
    report = {'scope': __doc__, 'completed': False, 'passed': False, 'readyForDeviceValidation': False,
              'sourceManifestSHA256': sha(args.run / 'models/manifest.json'),
              'sourceFrameReportSHA256': sha(args.run / 'coreml/report.json'),
              'sourceInspectionSHA256': sha(args.inspection / 'report.json'),
              'variants': {}, 'runs': {}}
    try:
        for variant in ('unchanged', 'conv2'):
            report['stage'] = f'Preparing {variant}'
            write_json(path, report)
            print(report['stage'], flush=True)
            package = args.output / f'{variant}.mlpackage'
            row = make_variant(args.run / 'models/BWTemporalPropagator.mlpackage', package, variant)
            row['files'] = file_hashes(package)
            report['variants'][variant] = row
        write_json(args.output / 'variants.json', {'sourceManifestSHA256': report['sourceManifestSHA256'],
                                                  'variants': report['variants']})
        for label, variant, units in (('unchanged-cpu', 'unchanged', 'CPU_ONLY'),
                                      ('conv2-cpu', 'conv2', 'CPU_ONLY'),
                                      ('conv2-gpu', 'conv2', 'CPU_AND_GPU'),
                                      ('conv2-ne', 'conv2', 'CPU_AND_NE')):
            report['stage'] = label
            write_json(path, report)
            row = report['runs'][label] = run_worker(args, label, variant, units)
            write_json(path, report)
            if not row['passed'] and units == 'CPU_ONLY':
                raise ValueError(f'{label} failed; accelerated runs skipped')
        report['completed'] = True
        report['passed'] = all(row['passed'] for row in report['runs'].values())
        report.pop('stage', None)
    except Exception as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        write_json(path, report)
        print(f'Report: {path}', flush=True)


if __name__ == '__main__':
    main()
