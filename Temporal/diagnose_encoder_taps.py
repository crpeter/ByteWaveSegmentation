#!/usr/bin/env python3
"""Sample encoder convolution boundaries without changing the production package.

Exposing intermediate outputs can change placement; report that explicitly and
check the original final outputs before interpreting sampled tensors.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np
from PIL import Image

from diagnose_initializer import summarize
from inspect_encoder_placement import inspect, write_json, sha

FINAL = ('raw_vision_features', 'initial_vision_features', 'high_res_feature_0', 'high_res_feature_1')


def worker():
    import coremltools as ct
    package, image_path, output, units = sys.argv[2:]
    print(f'Loading {units}: {package}', flush=True)
    model = ct.models.MLModel(package, compute_units=getattr(ct.ComputeUnit, units))
    features = model.get_spec().description.input
    if len(features) != 1 or features[0].name != 'image':
        raise ValueError('Unexpected diagnostic input interface')
    with Image.open(image_path) as image:
        image = image.convert('RGB')
        kind = features[0].type.WhichOneof('Type')
        if kind == 'imageType':
            value = image
        elif kind == 'multiArrayType':
            # extract_submodel can expose the raw NCHW MIL image input. Its
            # existing preprocessing multiply already includes 1/255 scaling.
            value = np.asarray(image, dtype=np.float32).transpose(2, 0, 1)[None]
        else:
            raise ValueError(f'Unexpected input type {kind}')
        print('Beginning encoder prediction', flush=True)
        result = model.predict({'image': value})
    np.savez(output, **result)
    print('Prediction returned', flush=True)


def predict(package, frame, root, label, units):
    log = root / f'{label}.log'
    output = root / f'{label}.npz'
    with log.open('w') as stream:
        try:
            result = subprocess.run([sys.executable, str(Path(__file__).resolve()), '__predict',
                                     str(package.resolve()), str(frame.resolve()), str(output.resolve()), units],
                                    stdout=stream, stderr=subprocess.STDOUT, timeout=300, check=False)
        except subprocess.TimeoutExpired as error:
            raise RuntimeError(f'{label}: worker timed out; see {log.name}') from error
    if result.returncode:
        raise RuntimeError(f'{label}: worker exited {result.returncode}; see {log.name}')
    with np.load(output, allow_pickle=False) as data:
        return {name: np.asarray(data[name], dtype=np.float32) for name in data.files}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--ne-indices', type=int, nargs='+', help='Zero-based indices among NE-preferred convolutions; default five evenly spaced samples')
    args = parser.parse_args()
    if platform.system() != 'Darwin':
        parser.error('Run on Mac')
    import coremltools as ct
    from coremltools.converters.mil.debugging_utils import extract_submodel
    plan_path = args.plan / 'report.json'
    plan = json.loads(plan_path.read_text())
    manifest_path = args.run / 'models/manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if plan.get('allPlansLoaded') is not True or plan.get('sourceManifestSHA256') != sha(manifest_path):
        raise ValueError('Source plan and model manifest do not match')
    package = args.run / 'models/BWTemporalImageEncoder.mlpackage'
    files = {str(p.relative_to(package)): sha(p) for p in package.rglob('*') if p.is_file()}
    if files != manifest['models']['ImageEncoder']['files'] or files != plan['models']['existing']['files']:
        raise ValueError('Source encoder package changed')
    operations = plan['models']['existing']['operations']
    ne = [op for op in operations if op['type'].split('.')[-1] == 'conv' and op['preferred'] == 'MLNeuralEngineComputeDevice']
    if not ne:
        raise ValueError('No NE-preferred convolutions to inspect')
    indices = sorted(set(args.ne_indices)) if args.ne_indices is not None else sorted({round(i * (len(ne)-1) / 4) for i in range(5)})
    if any(i < 0 or i >= len(ne) for i in indices):
        raise ValueError('NE convolution sample index out of range')
    frame = args.run / 'coreml/frame-00.png'
    with Image.open(frame) as image:
        pixels = np.asarray(image.convert('RGB'))
    source_report = json.loads((args.run / 'coreml/report.json').read_text())
    digest = hashlib.sha256(pixels.tobytes()).hexdigest()
    if pixels.shape != (1024, 1024, 3) or digest != source_report['frames'][0]['imageSHA256']:
        raise ValueError('Source image identity changed')
    args.output.mkdir(parents=True, exist_ok=False)
    report_path = args.output / 'report.json'
    report = {'scope': 'Sampled convolution inputs/outputs on CPU versus CPU_AND_NE. Exposed outputs may change partitioning. '
                       'First failing sample is not necessarily the first failing operation. Planned devices are not observed execution.',
              'readyForDeviceValidation': False, 'completed': False, 'sourceManifestSHA256': sha(manifest_path),
              'sourcePlanSHA256': sha(plan_path), 'imageSHA256': digest, 'coremltoolsVersion': ct.__version__,
              'neConvolutionCount': len(ne), 'sampleIndices': indices}
    try:
        report['stage'] = 'Extracting diagnostic graph with sampled outputs'
        write_json(report_path, report)
        model = ct.models.MLModel(str(package), compute_units=ct.ComputeUnit.CPU_ONLY)
        spec = model.get_spec()
        function = spec.mlProgram.functions['main']
        block = function.block_specializations[function.opset]
        by_output = {out.name: op for op in block.operations for out in op.outputs}
        samples = []
        for index in indices:
            output_names = ne[index]['outputs']
            if len(output_names) != 1 or output_names[0] not in by_output:
                raise ValueError('Plan output cannot be mapped to one serialized convolution')
            op = by_output[output_names[0]]
            if op.type != 'conv':
                raise ValueError('Plan operation does not map to conv')
            bindings = op.inputs['x'].arguments
            if len(bindings) != 1 or bindings[0].WhichOneof('binding') != 'name':
                raise ValueError('Unexpected convolution input')
            samples.append({'neIndex': index, 'input': bindings[0].name, 'output': output_names[0]})
        taps = list(dict.fromkeys(name for sample in samples for name in (sample['input'], sample['output'])))
        probe = extract_submodel(model, outputs=list(FINAL) + [n for n in taps if n not in FINAL])
        probe.user_defined_metadata['bytewave.contract'] = 'bytewave.encoder-taps.diagnostic.v1'
        probe_package = args.output / 'DiagnosticEncoderTaps.mlpackage'
        probe.save(str(probe_package))
        del probe, model
        report['samples'] = samples
        report['probeFiles'] = {str(p.relative_to(probe_package)): sha(p) for p in probe_package.rglob('*') if p.is_file()}
        report['stage'] = 'Inspecting diagnostic CPU/NE plan'
        write_json(report_path, report)
        probe_plan = inspect(probe_package)
        write_json(args.output / 'probe-plan.json', probe_plan)
        report['probePreferredCounts'] = probe_plan['preferredCounts']
        assignment = {name: op['preferred'] for op in probe_plan['operations'] for name in op['outputs']}
        for sample in samples:
            sample['probePreferredDevice'] = assignment.get(sample['output'], 'missing')
        values = {}
        for label, pkg, units in (('original-cpu', package, 'CPU_ONLY'), ('probe-cpu', probe_package, 'CPU_ONLY'),
                                  ('probe-ne', probe_package, 'CPU_AND_NE')):
            report['stage'] = label
            write_json(report_path, report)
            print(f'Predicting {label}...', flush=True)
            values[label] = predict(pkg, frame, args.output, label, units)
            if label == 'probe-cpu':
                report['cpuFinalAgreement'] = summarize({n: values[label][n] for n in FINAL}, values['original-cpu'])
                if not all(np.isfinite(values[label][n]).all() and np.isfinite(values['original-cpu'][n]).all()
                           and np.allclose(values[label][n], values['original-cpu'][n], atol=.001, rtol=.001) for n in FINAL):
                    raise ValueError('Diagnostic CPU final outputs differ from original; do not interpret NE taps')
        cpu, actual = values['probe-cpu'], values['probe-ne']
        if set(cpu) != set(actual) or any(not np.isfinite(x).all() for x in cpu.values()):
            raise ValueError('Probe CPU baseline invalid or NE output names changed')
        report['neFinalOutputs'] = summarize({n: actual[n] for n in FINAL}, {n: cpu[n] for n in FINAL})
        report['neTapOutputs'] = summarize({n: actual[n] for n in taps}, {n: cpu[n] for n in taps})
        report['finalNonfiniteFailureReproduced'] = any(row['nonfinite'] > 0 for row in report['neFinalOutputs'].values())
        report['sampledConvolutionPreferencesPreserved'] = all(s['probePreferredDevice'] == 'MLNeuralEngineComputeDevice' for s in samples)
        report['completed'] = True
        report.pop('stage', None)
    except Exception as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        write_json(report_path, report)
    print(f'Report: {report_path}', flush=True)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '__predict':
        worker()
    else:
        main()
