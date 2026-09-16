"""Isolated memory-attention diagnostics; no normal export changes.

MIL SDPA semantics: softmax(Q K^T / sqrt(d) + mask) V. Every original query
and key is retained. Only the four 4096-query memory attentions change.
https://apple.github.io/coremltools/source/coremltools.converters.mil.mil.ops.defs.html
"""
from __future__ import annotations

from collections import Counter
import sys

import numpy as np

from propagator_convs import signatures

CONTRACT = 'bytewave.propagator-attention.diagnostic.v1'
CHUNK = 256
NATIVE_QUERY_CHUNK = 1024
PADDED_KEY_COUNT = 4096
FP16_BASELINE_VARIANTS = ('memoryfp16q1024', 'memoryfp16k4096')
FP16_VARIANTS = ('memoryfp16', *FP16_BASELINE_VARIANTS)
VARIANTS = ('unchanged', 'chunk256', *FP16_VARIANTS)


def make_variant(package, destination, variant):
    import coremltools as ct
    from coremltools.converters.mil.frontend.milproto.load import load
    from coremltools.converters.mil.mil import Builder as mb, types

    if variant not in VARIANTS or ct.__version__ != '9.0':
        raise ValueError('Requires Core ML Tools 9.0 and a known variant')
    original = ct.models.MLModel(str(package), skip_model_load=True)
    spec = original.get_spec()
    previous = sys.getrecursionlimit()
    try:
        sys.setrecursionlimit(max(previous, 20_000))
        program = load(model_spec=spec, specification_version=spec.specificationVersion,
                       file_weights_dir=original.weights_dir)
        function = program.functions['main']
        before = signatures(function, {})
        selected = [op for op in function.operations
                    if op.op_type == 'scaled_dot_product_attention'
                    and tuple(op.query.shape) == (1, 1, 4096, 256)]
        if len(selected) != 4:
            found = [(op.name, list(op.query.shape), list(op.key.shape))
                     for op in function.operations if op.op_type == 'scaled_dot_product_attention']
            raise ValueError(f'Expected exactly four 4096-query memory attentions; found {found}')
        kinds = Counter()
        descriptions = []
        for op in selected:
            key_count = op.key.shape[-2]
            mask = op.attn_mask
            tensors = (op.query, op.key, op.value, op.outputs[0])
            if (key_count not in (4096, 3648)
                    or tuple(op.key.shape) != (1, 1, key_count, 256)
                    or tuple(op.value.shape) != (1, 1, key_count, 256)
                    or tuple(op.outputs[0].shape) != (1, 1, 4096, 256)
                    or any(v.dtype != types.fp32 for v in tensors)
                    or set(op.inputs) - {'query', 'key', 'value', 'attn_mask'}):
                raise ValueError(f'Unexpected attention shape, precision or semantics: {op.name}')
            if key_count == 4096:
                if mask is not None:
                    raise ValueError('Self-attention unexpectedly has a mask')
                kind = 'self'
            else:
                if mask is None or mask.dtype != types.fp32 or tuple(mask.shape) != (1, 1, 1, 3648):
                    raise ValueError('Cross-attention validity mask changed')
                kind = 'cross'
            kinds[kind] += 1
            descriptions.append({'output': op.outputs[0].name, 'kind': kind,
                                 'queryShape': list(op.query.shape), 'keyShape': list(op.key.shape),
                                 'valueShape': list(op.value.shape), 'dtype': 'fp32',
                                 'maskSource': mask.name if mask is not None else None})
        if kinds != {'self': 2, 'cross': 2}:
            raise ValueError('Expected two self and two cross memory attentions')

        changes, aliases, added_names = [], {}, set()
        if variant == 'chunk256':
            with function:
                for index, op in enumerate(selected):
                    old = op.outputs[0]
                    key_count = int(op.key.shape[-2])
                    prefix = f'bwattention_{index}'
                    chunks = []
                    for begin in range(0, 4096, CHUNK):
                        name = f'{prefix}_{begin // CHUNK}'
                        q = mb.slice_by_size(x=op.query, begin=[0, 0, begin, 0],
                                             size=[1, 1, CHUNK, 256], name=name + '_query', before_op=op)
                        scores = mb.matmul(x=q, y=op.key, transpose_y=True,
                                           name=name + '_scores', before_op=op)
                        scores = mb.mul(x=scores, y=np.float32(1.0 / 16.0),
                                        name=name + '_scale', before_op=op)
                        if op.attn_mask is not None:
                            scores = mb.add(x=scores, y=op.attn_mask, name=name + '_mask', before_op=op)
                        probabilities = mb.softmax(x=scores, axis=-1, name=name + '_softmax', before_op=op)
                        chunk = mb.matmul(x=probabilities, y=op.value, name=name + '_weighted', before_op=op)
                        chunks.append(chunk)
                    replacement = mb.concat(values=chunks, axis=2, name=prefix + '_restore', before_op=op)
                    if replacement.shape != old.shape or replacement.dtype != old.dtype:
                        raise ValueError('Attention output interface changed')
                    function.replace_uses_of_var_after_op(anchor_op=op, old_var=old, new_var=replacement)
                    function.remove_ops([op])
                    aliases[replacement.name] = old.name
                    changes.append({'oldOutput': old.name, 'newOutput': replacement.name,
                                    'queryChunkSize': CHUNK, 'chunks': 4096 // CHUNK,
                                    'keysPerChunk': key_count,
                                    'scoreTensorBytesPerChunk': CHUNK * key_count * 4})
        elif variant in FP16_VARIANTS:
            with function:
                for index, op in enumerate(selected):
                    old = op.outputs[0]
                    key_count = int(op.key.shape[-2])
                    prefix = f'bw{variant}_{index}'
                    inputs = {}
                    for key in ('query', 'key', 'value', 'attn_mask'):
                        value = getattr(op, key)
                        if value is not None:
                            inputs[key] = mb.cast(x=value, dtype='fp16', name=f'{prefix}_{key}', before_op=op)
                    if variant == 'memoryfp16k4096' and key_count == 3648:
                        # Append AFTER rotary/validity construction. Original keys,
                        # values, and additive mask remain in the same order. The
                        # extra score columns are -inf, so they have zero softmax
                        # probability for finite queries and valid original keys.
                        # This is a GPU shape experiment; faster dispatch is not
                        # guaranteed and additional keys/copies can cost more.
                        count = PADDED_KEY_COUNT - key_count
                        zeros = mb.const(val=np.zeros((1, 1, count, 256), dtype=np.float16),
                                         name=prefix + '_padding_zeros', before_op=op)
                        excluded = mb.const(val=np.full((1, 1, 1, count), -np.inf, dtype=np.float16),
                                            name=prefix + '_padding_mask', before_op=op)
                        for key in ('key', 'value'):
                            inputs[key] = mb.concat(values=[inputs[key], zeros], axis=2,
                                                    name=f'{prefix}_{key}_padded', before_op=op)
                        inputs['attn_mask'] = mb.concat(values=[inputs['attn_mask'], excluded], axis=3,
                                                       name=prefix + '_mask_padded', before_op=op)
                        if (any(tuple(inputs[key].shape) != (1, 1, PADDED_KEY_COUNT, 256)
                                for key in ('key', 'value'))
                                or tuple(inputs['attn_mask'].shape) != (1, 1, 1, PADDED_KEY_COUNT)):
                            raise ValueError('Cross-attention padding shape changed')
                    if variant == 'memoryfp16q1024':
                        # SDPA normalizes over keys, independently for each query.
                        # Keep the fused operation, all keys/values, and the exact
                        # broadcast mask/FP16 casts used by the working baseline.
                        # This changes GPU scheduling opportunities, not context.
                        chunks = []
                        for begin in range(0, 4096, NATIVE_QUERY_CHUNK):
                            name = f'{prefix}_{begin // NATIVE_QUERY_CHUNK}'
                            query = mb.slice_by_size(
                                x=inputs['query'], begin=[0, 0, begin, 0],
                                size=[1, 1, NATIVE_QUERY_CHUNK, 256],
                                name=name + '_query', before_op=op)
                            chunks.append(mb.scaled_dot_product_attention(
                                **{**inputs, 'query': query}, name=name + '_attention', before_op=op))
                        half = mb.concat(values=chunks, axis=2, name=prefix + '_concat', before_op=op)
                    else:
                        half = mb.scaled_dot_product_attention(**inputs, name=prefix + '_attention', before_op=op)
                    replacement = mb.cast(x=half, dtype='fp32', name=prefix + '_restore', before_op=op)
                    if replacement.shape != old.shape or replacement.dtype != old.dtype:
                        raise ValueError('Attention output interface changed')
                    function.replace_uses_of_var_after_op(anchor_op=op, old_var=old, new_var=replacement)
                    function.remove_ops([op])
                    aliases[replacement.name] = old.name
                    changes.append({'oldOutput': old.name, 'newOutput': replacement.name,
                                    'attentionPrecision': 'fp16', 'outputInterface': 'fp32',
                                    'maskCast': 'fp16' if 'attn_mask' in inputs else None,
                                    'allQueriesAndKeysRetained': True})
                    if variant == 'memoryfp16q1024':
                        changes[-1].update(queryChunkSize=NATIVE_QUERY_CHUNK,
                                           queryRanges=[[b, b + NATIVE_QUERY_CHUNK]
                                                        for b in range(0, 4096, NATIVE_QUERY_CHUNK)],
                                           chunks=4096 // NATIVE_QUERY_CHUNK,
                                           keysPerChunk=key_count,
                                           nativeSDPARetained=True)
                    if variant == 'memoryfp16k4096':
                        changes[-1].update(originalKeyCount=key_count,
                                           paddedKeyCount=PADDED_KEY_COUNT,
                                           appendedExcludedKeys=PADDED_KEY_COUNT - key_count,
                                           paddingMask='negativeInfinity' if key_count == 3648 else None,
                                           originalKeyOrderPreserved=True,
                                           queryChunking=False, nativeSDPARetained=True)
            if variant == 'memoryfp16k4096' and sorted(c['appendedExcludedKeys'] for c in changes) != [0, 0, 448, 448]:
                raise ValueError('Expected padding only for the two cross-attentions')
        added_names = {op.outputs[0].name for op in function.operations
                       if op.op_type != 'const' and op.outputs[0].name not in before}
        # Freeze the intended graph for conversion: preserve every unrelated op,
        # parameter, mask source and dtype; also compare the new operations exactly.
        expected = signatures(function, {})
        program.validate()
        program.skip_all_passes = True
        candidate = ct.convert(program, source='milinternal', convert_to='mlprogram',
                               minimum_deployment_target=ct.target.iOS18,
                               compute_precision=ct.precision.FLOAT32, skip_model_load=True)
        after_function = candidate._mil_program.functions['main']
        after = signatures(after_function, aliases)
        raw_after = signatures(after_function, {})
        replaced = {row['oldOutput'] for row in changes}
        for name, signature in before.items():
            if signature['type'] == 'const':
                continue  # Constants checked at their consumers by signatures().
            if name in replaced:
                if name in after:
                    raise ValueError('Original memory attention unexpectedly retained')
            elif after.get(name) != signature:
                raise ValueError(f'Unrelated operation changed: {name}')
        actual_added = {name for name in set(after) - set(before) if after[name]['type'] != 'const'}
        if actual_added != added_names:
            raise ValueError('Conversion changed the replacement operation set')
        for name in added_names:
            if raw_after.get(name) != expected[name]:
                raise ValueError(f'Diagnostic attention changed during conversion: {name}')
        for op in after_function.operations:
            name = op.outputs[0].name
            if name in added_names:
                expected_dtype = (types.fp16 if variant in FP16_VARIANTS and not name.endswith('_restore')
                                  else types.fp32)
                if any(v.dtype != expected_dtype for v in op.outputs):
                    raise ValueError(f'Replacement precision changed: {name}')
        actual_spec = candidate.get_spec()
        for direction in ('input', 'output'):
            if ({f.name: f.type for f in getattr(spec.description, direction)} !=
                    {f.name: f.type for f in getattr(actual_spec.description, direction)}):
                raise ValueError(f'External {direction} interface changed')
        candidate.user_defined_metadata.update(dict(spec.description.metadata.userDefined))
        candidate.user_defined_metadata['bytewave.contract'] = CONTRACT
        candidate.user_defined_metadata['bytewave.diagnostic.variant'] = variant
        candidate.save(str(destination))
        return {'variant': variant, 'inspectedAttentions': descriptions, 'modifications': changes,
                'unrelatedOperationsPreserved': True, 'externalInterfacesPreserved': True,
                'attentionPrecision': 'fp16' if variant in FP16_VARIANTS else 'fp32',
                'memoryNote': ('Padding adds 448 excluded key/value rows per cross-attention; no history is removed. '
                               'Peak memory, kernel selection and speed are not established.'
                               if variant == 'memoryfp16k4096' else
                               'Score tensor size is per chunk, not a measured or guaranteed peak allocation.')}
    finally:
        sys.setrecursionlimit(previous)
