"""Diagnostic-only replacement of two inspected memory-attention linear layers.

Same FP16 weights/bias and channel reduction, expressed as 1x1 convolution.
Floating-point accumulation and device placement may differ; validation is required.
"""
from __future__ import annotations

import hashlib
import sys

import numpy as np

CONTRACT = 'bytewave.propagator-convs.diagnostic.v1'
TARGETS = {'linear_9_cast_fp16': 0, 'linear_19_cast_fp16': 1}


def value_signature(value, aliases):
    signature = {'name': aliases.get(value.name, value.name), 'dtype': str(value.dtype),
                 'shape': [str(n) for n in value.shape]}
    if value.val is not None:
        array = np.asarray(value.val)
        if array.dtype.hasobject:
            raise ValueError(f'Unsupported object-valued constant: {value.name}')
        signature['constantSHA256'] = hashlib.sha256(array.tobytes()).hexdigest()
    return signature


def signatures(function, aliases):
    result = {}
    for op in function.operations:
        if op.blocks or not op.outputs:
            raise ValueError('Expected the inspected flat graph with named outputs')
        result[op.outputs[0].name] = {
            'type': op.op_type,
            'outputs': [value_signature(v, aliases) for v in op.outputs],
            'inputs': {k: [value_signature(v, aliases) for v in
                           (values if isinstance(values, (tuple, list)) else [values])]
                       for k, values in op.inputs.items()
                       if values is not None and op.op_type != 'const'}}
    return result


def make_variant(package, destination, variant):
    import coremltools as ct
    from coremltools.converters.mil.frontend.milproto.load import load
    from coremltools.converters.mil.mil import Builder as mb, types

    if variant not in ('unchanged', 'conv2') or ct.__version__ != '9.0':
        raise ValueError('This controlled rewrite requires Core ML Tools 9.0 and a known variant')
    original = ct.models.MLModel(str(package), skip_model_load=True)
    spec = original.get_spec()
    previous = sys.getrecursionlimit()
    try:
        sys.setrecursionlimit(max(previous, 20_000))
        # The same serialized loader used by coremltools.extract_submodel,
        # without its intermediate conversion/compilation or graph extraction.
        program = load(model_spec=spec, specification_version=spec.specificationVersion,
                       file_weights_dir=original.weights_dir)
        function = program.functions['main']
        before = signatures(function, {})
        selected = [op for op in function.operations if op.outputs[0].name in TARGETS]
        if len(selected) != 2:
            raise ValueError('Expected exactly two inspected linear operations')
        for op in selected:
            layer = TARGETS[op.outputs[0].name]
            prefix = f'model_memory_attention_layers_{layer}_linear2_'
            if (op.op_type != 'linear' or tuple(op.x.shape) != (1, 4096, 2048)
                    or tuple(op.weight.shape) != (256, 2048) or op.bias is None
                    or tuple(op.bias.shape) != (256,) or tuple(op.outputs[0].shape) != (1, 4096, 256)
                    or op.weight.name != prefix + 'weight_to_fp16'
                    or op.bias.name != prefix + 'bias_to_fp16'
                    or any(v.dtype != types.fp16 for v in (op.x, op.weight, op.bias, op.outputs[0]))
                    or op.weight.val is None or op.bias.val is None):
                raise ValueError(f'Inspected shape, source or precision changed: {op.name}')
        changes, aliases = [], {}
        if variant == 'conv2':
            with function:
                for op in selected:
                    old = op.outputs[0]
                    bias_name = op.bias.name
                    prefix = f'bwconv_{TARGETS[old.name]}'
                    weight = np.array(op.weight.val, copy=True).reshape(256, 2048, 1, 1)
                    if weight.dtype != np.float16 or not np.isfinite(weight).all():
                        raise ValueError('Expected finite FP16 constant weights')
                    x = mb.transpose(x=op.x, perm=[0, 2, 1], name=prefix + '_channels', before_op=op)
                    x = mb.reshape(x=x, shape=[1, 2048, 64, 64], name=prefix + '_spatial', before_op=op)
                    x = mb.conv(x=x, weight=weight, bias=op.bias, strides=[1, 1],
                                pad_type='valid', dilations=[1, 1], groups=1,
                                name=prefix + '_conv', before_op=op)
                    x = mb.reshape(x=x, shape=[1, 256, 4096], name=prefix + '_tokens', before_op=op)
                    replacement = mb.transpose(x=x, perm=[0, 2, 1], name=prefix + '_restore', before_op=op)
                    if replacement.dtype != old.dtype or replacement.shape != old.shape:
                        raise ValueError('Replacement interface changed')
                    function.replace_uses_of_var_after_op(anchor_op=op, old_var=old, new_var=replacement)
                    function.remove_ops([op])
                    aliases[replacement.name] = old.name
                    changes.append({'oldOutput': old.name, 'newOutput': replacement.name,
                                    'convolutionOutput': prefix + '_conv',
                                    'weightSHA256': hashlib.sha256(weight.tobytes()).hexdigest(),
                                    'sourceWeightSHA256': before[old.name]['inputs']['weight'][0]['constantSHA256'],
                                    'sourceBiasSHA256': before[old.name]['inputs']['bias'][0]['constantSHA256'],
                                    'biasSource': bias_name, 'weightShape': list(weight.shape)})
        program.validate()
        program.skip_all_passes = True
        candidate = ct.convert(program, source='milinternal', convert_to='mlprogram',
                               minimum_deployment_target=ct.target.iOS18,
                               compute_precision=ct.precision.FLOAT32, skip_model_load=True)
        after_function = candidate._mil_program.functions['main']
        after = signatures(after_function, aliases)
        for name, signature in before.items():
            if signature['type'] == 'const':
                # Constants are checked at every consuming input. Unused
                # weights may disappear when their old linear is removed.
                continue
            if variant == 'conv2' and name in TARGETS:
                if name in after:
                    raise ValueError('Original linear unexpectedly retained')
                continue
            if after.get(name) != signature:
                raise ValueError(f'Unrelated operation or precision changed: {name}')
        added = {name for name in set(after) - set(before) if after[name]['type'] != 'const'}
        expected_added = ({f'bwconv_{layer}_{suffix}' for layer in (0, 1)
                           for suffix in ('channels', 'spatial', 'conv', 'tokens', 'restore')}
                          if variant == 'conv2' else set())
        if added != expected_added:
            raise ValueError('Unexpected added operation')
        for change in changes:
            convolution = next(op for op in after_function.operations
                               if op.outputs[0].name == change['convolutionOutput'])
            if (convolution.op_type != 'conv'
                    or any(v.dtype != types.fp16 for v in
                           (convolution.x, convolution.weight, convolution.bias, convolution.outputs[0]))
                    or hashlib.sha256(np.asarray(convolution.weight.val).tobytes()).hexdigest()
                    != change['sourceWeightSHA256']
                    or hashlib.sha256(np.asarray(convolution.bias.val).tobytes()).hexdigest()
                    != change['sourceBiasSHA256']):
                raise ValueError('Convolution precision, weights or bias changed during conversion')
        actual_spec = candidate.get_spec()
        for direction in ('input', 'output'):
            expected = {f.name: f.type for f in getattr(spec.description, direction)}
            actual = {f.name: f.type for f in getattr(actual_spec.description, direction)}
            if actual != expected:
                raise ValueError(f'External {direction} interface changed')
        candidate.user_defined_metadata.update(dict(spec.description.metadata.userDefined))
        candidate.user_defined_metadata['bytewave.contract'] = CONTRACT
        candidate.user_defined_metadata['bytewave.diagnostic.variant'] = variant
        candidate.save(str(destination))
        return {'variant': variant, 'modifications': changes,
                'unrelatedOperationsPreserved': True, 'externalInterfacesPreserved': True}
    finally:
        sys.setrecursionlimit(previous)
