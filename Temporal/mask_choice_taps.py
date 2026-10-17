"""Expose existing score/argmax variables without rewriting mask selection."""
import sys

import numpy as np

from diagnose_occlusion import sha
from propagator_convs import signatures


def ancestors(var):
    pending, seen = [var], set()
    while pending:
        value = pending.pop()
        if value.name in seen:
            continue
        seen.add(value.name)
        op = value.op
        if op is not None:
            for item in op.inputs.values():
                if item is not None:
                    pending.extend(item if isinstance(item, (tuple, list)) else [item])
    return seen


def prepare(package, destination, expose):
    import coremltools as ct
    from coremltools.converters.mil.frontend.milproto.load import load
    from coremltools.converters.mil.mil import types
    if ct.__version__ != '9.0':
        raise ValueError('This diagnostic requires the validated coremltools 9.0.')
    source = ct.models.MLModel(str(package), skip_model_load=True)
    spec = source.get_spec()
    previous = sys.getrecursionlimit()
    try:
        sys.setrecursionlimit(max(previous, 20000))
        program = load(model_spec=spec, specification_version=spec.specificationVersion,
                       file_weights_dir=source.weights_dir)
        function = program.functions['main']
        before = signatures(function, {})
        finals = {var.name: var for var in function.outputs}
        choices = [op for op in function.operations if op.op_type == 'reduce_argmax'
                   and tuple(op.x.shape) == (1, 3)]
        if len(choices) != 1:
            found = [(op.name, list(op.x.shape)) for op in function.operations
                     if op.op_type == 'reduce_argmax']
            raise ValueError(f'Expected unique three-candidate argmax; found {found}')
        op = choices[0]
        scores, index = op.x, op.outputs[0]
        if (int(np.asarray(op.axis.val).item()) not in (-1, 1)
                or scores.dtype != types.fp32 or index.dtype != types.int32
                or tuple(index.shape) != (1,)
                or any(index.name not in ancestors(finals[name])
                       for name in ('low_res_mask', 'object_pointer'))
                or scores.name not in ancestors(finals['best_iou'])):
            raise ValueError('Score/selection dtype, axis or mask/pointer ancestry differs from expected graph.')
        mapping = {'scores': scores.name, 'index': index.name,
                   'argmaxOperation': op.name, 'scoreShape': list(scores.shape),
                   'feedsLowMaskAndPointer': True, 'feedsBestIoU': True}
        if expose:
            if scores.name in finals or index.name in finals:
                raise ValueError('Diagnostic outputs already exposed.')
            function.set_outputs(list(function.outputs) + [scores, index])
        program.validate()
        program.skip_all_passes = True
        converted = ct.convert(program, source='milinternal', convert_to='mlprogram',
                               minimum_deployment_target=ct.target.iOS18,
                               compute_precision=ct.precision.FLOAT32, skip_model_load=True)
        after = signatures(converted._mil_program.functions['main'], {})
        # Consumer signatures include constant values. No computation changes
        # or additional operations are allowed, apart from constant bookkeeping.
        if ({k: v for k, v in before.items() if v['type'] != 'const'} !=
                {k: v for k, v in after.items() if v['type'] != 'const'}):
            raise ValueError('Reconversion changed operation inputs, shapes, types or constants.')
        actual = converted.get_spec()
        if ({f.name: f.type for f in spec.description.input} !=
                {f.name: f.type for f in actual.description.input}):
            raise ValueError('Propagator input interface changed.')
        actual_outputs = {f.name: f.type for f in actual.description.output}
        for feature in spec.description.output:
            if actual_outputs.get(feature.name) != feature.type:
                raise ValueError(f'Original output interface changed: {feature.name}')
        expected_names = set(finals) | ({scores.name, index.name} if expose else set())
        if set(actual_outputs) != expected_names:
            raise ValueError('Unexpected diagnostic output interface.')
        converted.user_defined_metadata.update(dict(spec.description.metadata.userDefined))
        converted.user_defined_metadata['bytewave.contract'] = 'bytewave.mask-choice-taps.diagnostic.v1'
        converted.user_defined_metadata['bytewave.diagnostic.variant'] = 'taps' if expose else 'unchanged'
        converted.save(str(destination))
        return {'mapping': mapping, 'operationsPreserved': True,
                'originalInterfacesPreserved': True,
                'files': {str(p.relative_to(destination)): sha(p)
                          for p in sorted(destination.rglob('*')) if p.is_file()}}
    finally:
        sys.setrecursionlimit(previous)
