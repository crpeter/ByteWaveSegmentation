"""Diagnostic FP32 composition of the two high-resolution encoder projections.

Pinned EdgeTAM FPN levels 0/1 are lateral-only 1x1 projections, followed directly
by conv_s0/conv_s1. Compose W2 @ W1 and W2 @ b1 + b2 in float64, then store FP32.
Real-arithmetic equivalence does not imply identical floating-point results.
No precision pass, normal export change, or device-readiness claim.
"""
from __future__ import annotations

import hashlib
import sys

import numpy as np

from propagator_convs import signatures

CONTRACT = 'bytewave.encoder-projection.diagnostic.v1'
TARGETS = {'high_res_feature_0': (48, 32, 256), 'high_res_feature_1': (96, 64, 128)}


def digest(value):
    return hashlib.sha256(np.asarray(value).tobytes()).hexdigest()


def check_conv(op, channels_in, channels_out, side, types):
    if (op.op_type != 'conv' or tuple(op.x.shape) != (1, channels_in, side, side)
            or tuple(op.outputs[0].shape) != (1, channels_out, side, side)
            or tuple(op.weight.shape) != (channels_out, channels_in, 1, 1)
            or op.bias is None or tuple(op.bias.shape) != (channels_out,)
            or any(v.dtype != types.fp32 for v in (op.x, op.weight, op.bias, op.outputs[0]))
            or op.weight.val is None or op.bias.val is None
            or not np.isfinite(op.weight.val).all() or not np.isfinite(op.bias.val).all()
            or int(op.groups.val) != 1
            or not np.array_equal(op.strides.val, [1, 1])
            or not np.array_equal(op.dilations.val, [1, 1])
            or str(op.pad_type.val) not in ('valid', 'custom')
            or np.any(op.pad.val != 0)):
        raise ValueError(f'Expected finite FP32 unit-stride ungrouped 1x1 convolution: {op.name}')


def make_variant(package, destination, variant):
    import coremltools as ct
    from coremltools.converters.mil.frontend.milproto.load import load
    from coremltools.converters.mil.mil import Builder as mb, types

    if ct.__version__ != '9.0' or variant not in ('unchanged', 'projection'):
        raise ValueError('Requires coremltools 9.0 and a known projection variant')
    original = ct.models.MLModel(str(package), skip_model_load=True)
    spec = original.get_spec()
    if (len(spec.description.input) != 1 or spec.description.input[0].name != 'image'
            or spec.description.input[0].type.WhichOneof('Type') != 'imageType'):
        raise ValueError('Expected original RGB image interface')
    previous = sys.getrecursionlimit()
    try:
        sys.setrecursionlimit(max(previous, 20_000))
        program = load(model_spec=spec, specification_version=spec.specificationVersion,
                       file_weights_dir=original.weights_dir)
        function = program.functions['main']
        before = signatures(function, {})
        operations = list(function.operations)
        outputs = {v.name: v for v in function.outputs}
        if sum(op.op_type == 'conv' for op in operations) != 132:
            raise ValueError('Expected 132 original encoder convolutions')
        selected = []
        for name, (channels_in, channels_out, side) in TARGETS.items():
            second = outputs[name].op
            check_conv(second, 256, channels_out, side, types)
            first = second.x.op
            check_conv(first, channels_in, 256, side, types)
            intermediate = first.outputs[0]
            if (any(intermediate is v for v in function.outputs)
                    or len(intermediate.child_ops) != 1 or intermediate.child_ops[0] is not second):
                raise ValueError('Intermediate projection has another consumer; cannot remove it')
            selected.append((name, first, second))
        changes = []
        removed = set()
        if variant == 'projection':
            with function:
                for index, (name, first, second) in enumerate(selected):
                    w1 = np.asarray(first.weight.val, dtype=np.float64)[:, :, 0, 0]
                    w2 = np.asarray(second.weight.val, dtype=np.float64)[:, :, 0, 0]
                    weight = (w2 @ w1).astype(np.float32)[:, :, None, None]
                    bias = (w2 @ np.asarray(first.bias.val, dtype=np.float64)
                            + np.asarray(second.bias.val, dtype=np.float64)).astype(np.float32)
                    if not np.isfinite(weight).all() or not np.isfinite(bias).all():
                        raise ValueError('Composed constants are nonfinite')
                    changes.append({'output': name, 'sourceInput': first.x.name,
                                    'removedIntermediate': first.outputs[0].name,
                                    'firstWeightSHA256': digest(first.weight.val),
                                    'secondWeightSHA256': digest(second.weight.val),
                                    'firstBiasSHA256': digest(first.bias.val),
                                    'secondBiasSHA256': digest(second.bias.val),
                                    'composedWeightSHA256': digest(weight), 'composedBiasSHA256': digest(bias),
                                    'weightShape': list(weight.shape), 'computationPrecision': 'float32'})
                    replacement = mb.conv(x=first.x, weight=weight, bias=bias, strides=[1, 1],
                                          pad_type='valid', dilations=[1, 1], groups=1,
                                          name=f'bw_projection_{index}', before_op=second)
                    old = second.outputs[0]
                    function.replace_uses_of_var_after_op(anchor_op=second, old_var=old, new_var=replacement)
                    removed.update((first.outputs[0].name, name))
                    function.remove_ops([second, first])
                    replacement.set_name(name)
        function.validate(force_validate=True)
        program.validate()
        program.skip_all_passes = True
        converted = ct.convert(program, source='milinternal', convert_to='mlprogram',
                               minimum_deployment_target=ct.target.iOS18,
                               compute_precision=ct.precision.FLOAT32, skip_model_load=True)
        result_function = converted._mil_program.functions['main']
        after = signatures(result_function, {})
        for name, signature in before.items():
            if signature['type'] != 'const' and name not in removed and after.get(name) != signature:
                raise ValueError(f'Unrelated operation/precision changed: {name}')
        expected = {n for n, s in before.items() if s['type'] != 'const'} - removed
        if variant == 'projection':
            expected.update(TARGETS)
        if {n for n, s in after.items() if s['type'] != 'const'} != expected:
            raise ValueError('Unexpected nonconstant operation added/removed')
        for change in changes:
            op = next(op for op in result_function.operations if op.outputs[0].name == change['output'])
            check_conv(op, *TARGETS[change['output']], types)
            if (op.x.name != change['sourceInput']
                    or digest(op.weight.val) != change['composedWeightSHA256']
                    or digest(op.bias.val) != change['composedBiasSHA256']):
                raise ValueError('Composed input/weights changed during conversion')
        actual = converted.get_spec()
        # The serialized loader represents the existing image preprocessing as
        # MIL operations. Restore its external image descriptor, never add scale.
        if spec.mlProgram.functions['main'].inputs[0] != actual.mlProgram.functions['main'].inputs[0]:
            raise ValueError('MIL input type changed')
        actual.description.input[0].CopyFrom(spec.description.input[0])
        for direction in ('input', 'output'):
            if ({f.name: f.type for f in getattr(actual.description, direction)}
                    != {f.name: f.type for f in getattr(spec.description, direction)}):
                raise ValueError(f'External {direction} interface changed')
        final = ct.models.MLModel(actual, weights_dir=converted.weights_dir, skip_model_load=True)
        final.user_defined_metadata.update(dict(spec.description.metadata.userDefined))
        final.user_defined_metadata['bytewave.contract'] = CONTRACT
        final.user_defined_metadata['bytewave.diagnostic.variant'] = variant
        final.save(str(destination))
        return {'variant': variant, 'modifications': changes, 'unrelatedOperationsPreserved': True,
                'externalInterfacesPreserved': True, 'noNewPrecisionPass': True}
    finally:
        sys.setrecursionlimit(previous)
