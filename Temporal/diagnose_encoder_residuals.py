#!/usr/bin/env python3
"""Full-encoder controls for residual and other shared FP32/FP16 activation paths.

One saved frame, CPU and CPU_AND_NE; no temporal/device readiness or speed claim.
Activation rounding changes numerical precision and may change placement and lifetimes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys

import numpy as np
from PIL import Image

from diagnose_encoder_prefix import file_hashes
from diagnose_encoder_taps import FINAL, predict
from diagnose_initializer import summarize
from inspect_encoder_placement import inspect, sha, write_json


def through_convolution(value, source):
    """Does a data path from source to value include a convolution?"""
    pending, visited = [(value, False)], set()
    while pending:
        value, crossed = pending.pop()
        key = (id(value), crossed)
        if key in visited:
            continue
        visited.add(key)
        if value is source:
            if crossed:
                return True
            continue
        op = getattr(value, 'op', None)
        if op is None or op.op_type == 'const':
            continue
        # Weights/bias are not activation ancestry.
        inputs = [op.x] if op.op_type == 'conv' else op.get_flattened_inputs()
        pending.extend((item, crossed or op.op_type == 'conv') for item in inputs)
    return False


def residual_edges(function):
    from coremltools.converters.mil.mil import types
    casts = [op for op in function.operations if op.op_type == 'cast'
             and op.x.dtype == types.fp32 and op.outputs[0].dtype == types.fp16]
    edges = []
    for op in function.operations:
        if op.op_type != 'add' or op.outputs[0].dtype != types.fp32:
            continue
        matches = []
        for operand, other in (('x', op.y), ('y', op.x)):
            skip = getattr(op, operand)
            if skip.dtype != types.fp32 or skip.shape != op.outputs[0].shape:
                continue
            for cast in casts:
                half = cast.outputs[0]
                if cast.x is skip and through_convolution(other, half):
                    matches.append((op, operand, skip, half))
        if len(matches) > 1:
            raise ValueError(f'Ambiguous residual/cast pattern: {op.name}')
        edges.extend(matches)
    return edges


def describe(edge):
    op, operand, skip, half = edge
    return {'output': op.outputs[0].name, 'operand': operand,
            'oldSource': skip.name, 'roundedThrough': half.name}


def fanout_edges(function):
    """Find later consumers of an FP32 activation also cast for convolution.

    Includes saved feature maps and non-add consumers. Preserve the original
    convolution casts and consumers before the chosen cast. Follow identities
    and graph order rather than guessing tensor names or module names.
    """
    from coremltools.converters.mil.mil import types
    operations = list(function.operations)
    sources = {}
    for index, op in enumerate(operations):
        if (op.op_type != 'cast' or op.x.dtype != types.fp32
                or op.outputs[0].dtype != types.fp16):
            continue
        half = op.outputs[0]
        if not any(child.op_type == 'conv' and child.x is half for child in operations[index + 1:]):
            continue
        # Repeated casts have identical rounding; use the earliest eligible one.
        sources.setdefault(id(op.x), (index, op.x, half))
    edges = []
    for index, source, half in sources.values():
        if any(source is output for output in function.outputs):
            raise ValueError('Fanout control does not rewrite a source that is also a model output')
        for consumer in operations[index + 1:]:
            if (consumer.op_type == 'cast' and consumer.x is source
                    and consumer.outputs[0].dtype == types.fp16):
                continue
            for operand, value in consumer.inputs.items():
                values = value if isinstance(value, (list, tuple)) else (value,)
                if any(item is source for item in values):
                    edges.append((consumer, operand, source, half))
    return edges


def replace_operand(op, operand, source, replacement):
    value = op.inputs[operand]
    if isinstance(value, (list, tuple)):
        value = type(value)(replacement if item is source else item for item in value)
    elif value is source:
        value = replacement
    else:
        raise ValueError('Selected operand no longer contains the expected source')
    op.set_inputs(**{operand: value})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--prefix-control', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--variants', nargs='+', choices=('unchanged', 'single', 'matching', 'fanout'),
                        default=['unchanged', 'single', 'matching'], help='Choose only new controls when earlier variants are already recorded')
    args = parser.parse_args()
    if platform.system() != 'Darwin':
        parser.error('Run on Mac')
    import coremltools as ct
    from coremltools.converters.mil.debugging_utils import extract_submodel
    from coremltools.converters.mil.mil import Builder as mb

    manifest_path = args.run / 'models/manifest.json'
    manifest = json.loads(manifest_path.read_text())
    package = args.run / 'models/BWTemporalImageEncoder.mlpackage'
    if file_hashes(package) != manifest['models']['ImageEncoder']['files']:
        raise ValueError('Original encoder bytes changed')
    control_path = args.prefix_control / 'report.json'
    control = json.loads(control_path.read_text())
    if control.get('completed') is not True or control.get('sourceManifestSHA256') != sha(manifest_path):
        raise ValueError('Prefix control and original encoder manifest must match')
    success = control['prefixes']['input_39']
    if success.get('neAllFinite') is not True or success.get('neCosineAtLeast099') is not True:
        raise ValueError('Expected the passing rounded residual prefix control')
    if success.get('modification', {}).get('newSourceExpression') != 'float32(input_33_to_fp16)':
        raise ValueError('Unexpected prefix intervention')
    if file_hashes(args.prefix_control / 'input_39.mlpackage') != success['files']:
        raise ValueError('Prefix control package bytes changed')
    frame = args.run / 'coreml/frame-00.png'
    with Image.open(frame) as image:
        pixels = np.asarray(image.convert('RGB'))
    digest = hashlib.sha256(pixels.tobytes()).hexdigest()
    if pixels.shape != (1024, 1024, 3) or digest != control['imageSHA256']:
        raise ValueError('Source frame changed')

    args.output.mkdir(parents=True, exist_ok=False)
    report_path = args.output / 'report.json'
    report = {'scope': __doc__, 'completed': False, 'readyForDeviceValidation': False,
              'sourceManifestSHA256': sha(manifest_path), 'sourcePrefixControlSHA256': sha(control_path),
              'imageSHA256': digest, 'coremltoolsVersion': ct.__version__,
              'policy': {'encoderCosineMinimum': .99, 'unchangedCPUAtol': .001, 'unchangedCPURtol': .001},
              'variants': {}}
    try:
        report['stage'] = 'Original CPU reference'
        write_json(report_path, report)
        print('Predicting original-cpu...', flush=True)
        reference = predict(package, frame, args.output, 'original-cpu', 'CPU_ONLY')
        if set(reference) != set(FINAL) or any(not np.isfinite(v).all() for v in reference.values()):
            raise ValueError('Invalid original CPU baseline')
        report['cpuReference'] = summarize(reference)
        report['cpuReferenceFileSHA256'] = sha(args.output / 'original-cpu.npz')
        for variant in dict.fromkeys(args.variants):
            row = report['variants'][variant] = {}
            report['stage'] = f'{variant}: extraction'
            write_json(report_path, report)
            print(f'Preparing {variant} full encoder...', flush=True)
            model = ct.models.MLModel(str(package), compute_units=ct.ComputeUnit.CPU_ONLY)
            previous = sys.getrecursionlimit()
            try:
                sys.setrecursionlimit(max(previous, 20_000))
                extracted = extract_submodel(model, outputs=list(FINAL))
                program = extracted._mil_program
                if program is None:
                    raise ValueError('Extractor did not retain MIL program')
                function = program.functions['main']
                edges = residual_edges(function)
                known = [edge for edge in edges if describe(edge) ==
                         {'output': 'input_39', 'operand': 'y', 'oldSource': 'input_33', 'roundedThrough': 'input_33_to_fp16'}]
                if len(known) != 1:
                    raise ValueError('Known successful residual pattern was not found exactly once')
                selected = [] if variant == 'unchanged' else known if variant == 'single' else edges
                row['matchedEdges'] = [describe(edge) for edge in edges]
                if variant == 'fanout':
                    selected = fanout_edges(function)
                    if len(selected) <= len(edges) or describe(known[0]) not in [describe(edge) for edge in selected]:
                        raise ValueError('Fanout control must include the known residual and additional consumers')
                    row['matchedFanoutEdges'] = [describe(edge) for edge in selected]
                row['modifiedEdges'] = []
                with function:
                    for index, edge in enumerate(selected):
                        op, operand, skip, half = edge
                        restored = mb.cast(x=half, dtype='fp32', name=f'diagnostic_rounded_skip_{index}', before_op=op)
                        replace_operand(op, operand, skip, restored)
                        row['modifiedEdges'].append({**describe(edge), 'newSource': restored.name})
                program.validate()
                program.skip_all_passes = True
                # Apply the same second conversion to the unchanged control.
                candidate = ct.convert(program, source='milinternal', convert_to='mlprogram',
                                       minimum_deployment_target=ct.target.iOS18,
                                       compute_precision=ct.precision.FLOAT32, skip_model_load=True)
            finally:
                sys.setrecursionlimit(previous)
            candidate.user_defined_metadata['bytewave.contract'] = 'bytewave.encoder-residuals.diagnostic.v1'
            destination = args.output / f'{variant}.mlpackage'
            candidate.save(str(destination))
            del candidate, extracted, model, program, function, edges, known, selected
            row['files'] = file_hashes(destination)
            report['stage'] = f'{variant}: plan'
            write_json(report_path, report)
            plan = inspect(destination)
            write_json(args.output / f'{variant}-plan.json', plan)
            row['preferredCounts'] = plan['preferredCounts']
            row['preferredCountsByOperationType'] = plan['preferredCountsByOperationType']
            values = {}
            for label, units in (('cpu', 'CPU_ONLY'), ('ne', 'CPU_AND_NE')):
                report['stage'] = f'{variant}: {label}'
                write_json(report_path, report)
                print(f'Predicting {variant}-{label}...', flush=True)
                try:
                    actual = predict(destination, frame, args.output, f'{variant}-{label}', units)
                except RuntimeError as error:
                    row[label] = {'passed': False, 'error': str(error)}
                    break
                if set(actual) != set(FINAL) or any(actual[n].shape != reference[n].shape for n in FINAL):
                    raise ValueError('Encoder output interface changed')
                values[label] = actual
                checks = summarize(actual, reference)
                passed = all(v['nonfinite'] == 0 and v.get('cosineSimilarity', -1) >= .99 for v in checks.values())
                if label == 'cpu' and variant == 'unchanged':
                    passed = passed and all(np.allclose(actual[n], reference[n], atol=.001, rtol=.001) for n in FINAL)
                row[label] = {'passed': bool(passed), 'againstOriginalCPU': checks}
                if label == 'cpu' and not passed:
                    if variant == 'unchanged':
                        raise ValueError('Unchanged conversion control failed CPU parity')
                    row['neSkipped'] = 'CPU accuracy gate failed'
                    break
                if label == 'ne':
                    local = summarize(actual, values['cpu'])
                    row[label]['againstVariantCPU'] = local
                    row[label]['passed'] = bool(passed and all(v['nonfinite'] == 0 and
                                                              v.get('cosineSimilarity', -1) >= .99 for v in local.values()))
                write_json(report_path, report)
            write_json(report_path, report)
        report['completed'] = True
        report.pop('stage', None)
    except Exception as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        write_json(report_path, report)
    print(f'Report: {report_path}', flush=True)


if __name__ == '__main__':
    main()
