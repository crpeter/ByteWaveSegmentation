"""Pack the two memory self-attention Q/K/V projections; diagnostic only.

MIL linear computes x @ weight.T + bias. Concatenate the existing FP16
weight/bias rows in Q/K/V order, then slice the result into the original outputs.
No arithmetic on constants, precision change, or attention/context reduction.
Larger GEMMs and added slices may help or hurt; device timing must decide.
https://apple.github.io/coremltools/_modules/coremltools/converters/mil/mil/ops/defs/iOS15/linear.html
"""
from __future__ import annotations

import hashlib

import numpy as np


def digest(value):
    return hashlib.sha256(np.asarray(value).tobytes()).hexdigest()


def pack_self_qkv(function):
    from coremltools.converters.mil.mil import Builder as mb, types

    operations = list(function.operations)
    groups = []
    for layer in (0, 1):
        group = []
        for role in ('q', 'k', 'v'):
            prefix = f'model_memory_attention_layers_{layer}_self_attn_{role}_proj_'
            matches = [op for op in operations if op.op_type == 'linear'
                       and op.weight.name in (prefix + 'weight', prefix + 'weight_to_fp16')]
            if len(matches) != 1:
                raise ValueError(f'Expected one self-attention {prefix}linear; found {len(matches)}')
            op = matches[0]
            if (set(op.inputs) != {'x', 'weight', 'bias'} or op.bias is None
                    or op.bias.name not in (prefix + 'bias', prefix + 'bias_to_fp16')
                    or tuple(op.x.shape) != (1, 4096, 256)
                    or tuple(op.weight.shape) != (256, 256)
                    or tuple(op.bias.shape) != (256,)
                    or len(op.outputs) != 1 or tuple(op.outputs[0].shape) != (1, 4096, 256)
                    or any(v.dtype != types.fp16 for v in (op.x, op.weight, op.bias, op.outputs[0]))
                    or any(v.val is None or np.asarray(v.val).dtype != np.float16
                           or not np.isfinite(v.val).all() for v in (op.weight, op.bias))
                    or any(op.outputs[0] is v for v in function.outputs)):
                raise ValueError(f'Unexpected self-attention projection interface: {op.name}')
            group.append(op)
        # Do not equate separate casts/values just because their shapes match.
        if any(op.x is not group[0].x for op in group):
            raise ValueError(f'Layer {layer} Q/K/V do not share the identical input variable')
        groups.append(group)

    changes, aliases = [], {}
    with function:
        for layer, group in enumerate(groups):
            first = min(group, key=operations.index)
            weight = np.concatenate([op.weight.val for op in group], axis=0)
            bias = np.concatenate([op.bias.val for op in group], axis=0)
            if weight.shape != (768, 256) or bias.shape != (768,):
                raise ValueError('Packed projection dimensions changed')
            for index, op in enumerate(group):
                if (weight[index * 256:(index + 1) * 256].tobytes() != op.weight.val.tobytes()
                        or bias[index * 256:(index + 1) * 256].tobytes() != op.bias.val.tobytes()):
                    raise ValueError('Packing changed source parameter bytes')
            prefix = f'bwqkv_{layer}'
            packed = mb.linear(x=first.x, weight=weight, bias=bias,
                               name=prefix + '_packed', before_op=first)
            if packed.dtype != types.fp16 or tuple(packed.shape) != (1, 4096, 768):
                raise ValueError('Packed linear interface changed')
            for index, (role, op) in enumerate(zip(('q', 'k', 'v'), group)):
                old = op.outputs[0]
                replacement = mb.slice_by_size(x=packed, begin=[0, 0, index * 256],
                                               size=[1, 4096, 256], name=prefix + '_' + role,
                                               before_op=op)
                if replacement.shape != old.shape or replacement.dtype != old.dtype:
                    raise ValueError('Unpacked projection interface changed')
                function.replace_uses_of_var_after_op(anchor_op=op, old_var=old, new_var=replacement)
                aliases[replacement.name] = old.name
                changes.append({'oldOutput': old.name, 'newOutput': replacement.name,
                                'kind': 'packedSelfProjection', 'layer': layer, 'role': role,
                                'sourceInput': op.x.name, 'packedOutput': packed.name,
                                'sourceWeightSHA256': digest(op.weight.val),
                                'sourceBiasSHA256': digest(op.bias.val),
                                'packedWeightSHA256': digest(weight), 'packedBiasSHA256': digest(bias),
                                'channelRange': [index * 256, (index + 1) * 256],
                                'computationPrecision': 'fp16', 'parameterBytesPreserved': True})
            function.remove_ops(group)
    return changes, aliases
