"""Diagnostic-only replacement of candidate-index uses with one external input.

Keep the original score computation and argmax exposed. One converted model is
used for both same-index control and alternate-index intervention, preventing
different constant-index compilations from confounding the comparison.
"""
import sys

import numpy as np

from diagnose_occlusion import sha
from mask_choice_taps import ancestors
from propagator_convs import signatures

INDEX_INPUT = 'diagnostic_candidate_index'


def prepare(package, destination):
    import coremltools as ct
    from coremltools.converters.mil.frontend.milproto.load import load
    from coremltools.converters.mil.mil import Builder as mb, types

    if ct.__version__ != '9.0':
        raise ValueError('This diagnostic requires coremltools 9.0.')
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
            raise ValueError('Expected unique three-candidate argmax.')
        op = choices[0]
        scores, index = op.x, op.outputs[0]
        affected = ('low_res_mask', 'high_res_mask', 'object_pointer', 'memory_features')
        if (int(np.asarray(op.axis.val).item()) not in (-1, 1)
                or scores.dtype != types.fp32 or index.dtype != types.int32
                or tuple(index.shape) != (1,)
                or any(index.name not in ancestors(finals[name]) for name in affected)
                or scores.name not in ancestors(finals['best_iou'])
                or INDEX_INPUT in function.inputs or INDEX_INPUT in before
                or scores.name in finals or index.name in finals):
            raise ValueError('Unexpected selection topology, dtype or interface.')
        # coremltools 9.0 Function stores placeholders and their Vars separately.
        # Keep both dictionaries and any frontend input-type declarations aligned.
        placeholder = mb.placeholder(shape=(1,), dtype=types.int32, name=INDEX_INPUT)
        function.placeholder_inputs[INDEX_INPUT] = placeholder
        replacement = placeholder.outputs[0]
        function.inputs[INDEX_INPUT] = replacement
        if function.input_types:
            function.set_input_types(tuple(function.input_types) +
                (ct.TensorType(name=INDEX_INPUT, shape=(1,), dtype=np.int32),))
        function.replace_uses_of_var_after_op(anchor_op=op, old_var=index, new_var=replacement)
        # The natural argmax remains observable but no longer selects the masks
        # or pointer. best_iou remains the unmodified maximum predicted score.
        function.set_outputs(list(function.outputs) + [scores, index])
        program.validate()
        program.skip_all_passes = True
        converted = ct.convert(program, source='milinternal', convert_to='mlprogram',
                               minimum_deployment_target=ct.target.iOS18,
                               compute_precision=ct.precision.FLOAT32, skip_model_load=True)
        actual_function = converted._mil_program.functions['main']
        after = signatures(actual_function, {INDEX_INPUT: index.name})
        if ({k: val for k, val in before.items() if val['type'] != 'const'} !=
                {k: val for k, val in after.items() if val['type'] != 'const'}):
            raise ValueError('Operations changed beyond the declared index substitution.')
        actual_finals = {var.name: var for var in actual_function.outputs}
        if any(INDEX_INPUT not in ancestors(actual_finals[name])
               or index.name in ancestors(actual_finals[name]) for name in affected):
            raise ValueError('Override does not exclusively drive expected output selection.')
        actual = converted.get_spec()
        actual_inputs = {f.name: f.type for f in actual.description.input}
        expected_inputs = {f.name: f.type for f in spec.description.input}
        if set(actual_inputs) != set(expected_inputs) | {INDEX_INPUT}:
            raise ValueError('Unexpected input names.')
        if any(actual_inputs[name] != value for name, value in expected_inputs.items()):
            raise ValueError('Original input interface changed.')
        added = actual_inputs[INDEX_INPUT].multiArrayType
        if list(added.shape) != [1] or added.dataType != ct.proto.FeatureTypes_pb2.ArrayFeatureType.INT32:
            raise ValueError('Override input must be int32[1].')
        actual_outputs = {f.name: f.type for f in actual.description.output}
        if set(actual_outputs) != set(finals) | {scores.name, index.name}:
            raise ValueError('Unexpected output names.')
        if any(actual_outputs[f.name] != f.type for f in spec.description.output):
            raise ValueError('Original output interface changed.')
        converted.user_defined_metadata.update(dict(spec.description.metadata.userDefined))
        converted.user_defined_metadata['bytewave.contract'] = 'bytewave.mask-choice-override.diagnostic.v1'
        converted.save(str(destination))
        return {'mapping': {'scores': scores.name, 'index': index.name,
                            'overrideInput': INDEX_INPUT, 'affectedOutputs': list(affected)},
                'onlyIndexUsesReplaced': True, 'originalInterfacesPreserved': True,
                'bestIoUSemantics': 'Unchanged maximum score, not the forced candidate score.',
                'files': {str(p.relative_to(destination)): sha(p)
                          for p in sorted(destination.rglob('*')) if p.is_file()}}
    finally:
        sys.setrecursionlimit(previous)
