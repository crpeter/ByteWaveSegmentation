"""Pinned encoder fanout rewrite validated on the Mac 20-frame fixture.

Keep FP16 convolution paths; feed later FP32 consumers from the corresponding
rounded activation. Source weights and original PyTorch reference are untouched.
"""
from __future__ import annotations

import sys

FANOUT_POLICY = "rounded-encoder-fanout-39.v1"
EXPECTED_EDGES = frozenset((
    ('input_33', 'x', 'input_23', 'input_23_to_fp16'),
    ('input_39', 'y', 'input_33', 'input_33_to_fp16'),
    ('input_49', 'x', 'input_39', 'input_39_to_fp16'),
    ('input_55', 'y', 'input_49', 'input_49_to_fp16'),
    ('x', 'x', 'input_49', 'input_49_to_fp16'),
    ('input_65', 'x', 'input_55', 'input_55_to_fp16'),
    ('x_13', 'y', 'input_83', 'input_83_to_fp16'),
    ('input_105', 'x', 'input_95', 'input_95_to_fp16'),
    ('input_111', 'y', 'input_105', 'input_105_to_fp16'),
    ('input_121', 'x', 'input_111', 'input_111_to_fp16'),
    ('input_127', 'y', 'input_121', 'input_121_to_fp16'),
    ('x_97', 'x', 'input_121', 'input_121_to_fp16'),
    ('input_137', 'x', 'input_127', 'input_127_to_fp16'),
    ('x_25', 'y', 'input_155', 'input_155_to_fp16'),
    ('input_177', 'x', 'input_167', 'input_167_to_fp16'),
    ('input_183', 'y', 'input_177', 'input_177_to_fp16'),
    ('input_193', 'x', 'input_183', 'input_183_to_fp16'),
    ('x_33', 'y', 'input_193', 'input_193_to_fp16'),
    ('input_215', 'x', 'input_205', 'input_205_to_fp16'),
    ('input_221', 'y', 'input_215', 'input_215_to_fp16'),
    ('input_231', 'x', 'input_221', 'input_221_to_fp16'),
    ('x_41', 'y', 'input_231', 'input_231_to_fp16'),
    ('input_253', 'x', 'input_243', 'input_243_to_fp16'),
    ('input_259', 'y', 'input_253', 'input_253_to_fp16'),
    ('input_269', 'x', 'input_259', 'input_259_to_fp16'),
    ('x_49', 'y', 'input_269', 'input_269_to_fp16'),
    ('input_291', 'x', 'input_281', 'input_281_to_fp16'),
    ('input_297', 'y', 'input_291', 'input_291_to_fp16'),
    ('input_307', 'x', 'input_297', 'input_297_to_fp16'),
    ('x_57', 'y', 'input_307', 'input_307_to_fp16'),
    ('input_329', 'x', 'input_319', 'input_319_to_fp16'),
    ('input_335', 'y', 'input_329', 'input_329_to_fp16'),
    ('input_345', 'x', 'input_335', 'input_335_to_fp16'),
    ('x_65', 'y', 'input_345', 'input_345_to_fp16'),
    ('input_367', 'x', 'input_357', 'input_357_to_fp16'),
    ('input_373', 'y', 'input_367', 'input_367_to_fp16'),
    ('input_383', 'x', 'input_373', 'input_373_to_fp16'),
    ('var_1561', 'x', 'input_383', 'input_383_to_fp16'),
    ('x_73', 'y', 'input_383', 'input_383_to_fp16'),
))


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


def apply_encoder_fanout(model, output_names):
    """Apply the tested serialized-graph rewrite and retain the image interface."""
    import coremltools as ct
    from coremltools.converters.mil.debugging_utils import extract_submodel
    from coremltools.converters.mil.mil import Builder as mb

    original = model.get_spec()
    if (len(original.description.input) != 1 or original.description.input[0].name != 'image'
            or original.description.input[0].type.WhichOneof('Type') != 'imageType'):
        raise ValueError('Expected original encoder RGB image interface')
    if {feature.name for feature in original.description.output} != set(output_names):
        raise ValueError('Unexpected original encoder output names')
    previous = sys.getrecursionlimit()
    try:
        sys.setrecursionlimit(max(previous, 20_000))
        # Reload the protobuf, as in the passing package-based diagnostic. Do
        # not extract from the converter's pre-serialization MIL program.
        serialized = ct.models.MLModel(original, weights_dir=model.weights_dir,
                                      compute_units=ct.ComputeUnit.CPU_ONLY, skip_model_load=True)
        extracted = extract_submodel(serialized, outputs=list(output_names))
        program = extracted._mil_program
        if program is None:
            raise ValueError('Extractor did not retain MIL program')
        function = program.functions['main']
        edges = fanout_edges(function)
        identities = [(op.outputs[0].name, operand, source.name, half.name)
                      for op, operand, source, half in edges]
        if len(identities) != 39 or set(identities) != EXPECTED_EDGES:
            raise ValueError('Encoder fanout graph differs from the 39 tested edges; do not export')
        def convolution_signature(function):
            return [(op.outputs[0].name, op.x.dtype, op.weight.dtype, op.outputs[0].dtype)
                    for op in function.operations if op.op_type == 'conv']
        before = convolution_signature(function)
        modifications = []
        with function:
            for index, (op, operand, source, half) in enumerate(edges):
                restored = mb.cast(x=half, dtype='fp32', name=f'diagnostic_rounded_skip_{index}', before_op=op)
                replace_operand(op, operand, source, restored)
                modifications.append({'output': op.outputs[0].name, 'operand': operand,
                                      'oldSource': source.name, 'roundedThrough': half.name,
                                      'newSource': restored.name})
        program.validate()
        program.skip_all_passes = True
        patched = ct.convert(program, source='milinternal', convert_to='mlprogram',
                             minimum_deployment_target=ct.target.iOS18,
                             compute_precision=ct.precision.FLOAT32, skip_model_load=True)
        if convolution_signature(patched._mil_program.functions['main']) != before:
            raise ValueError('Fanout rewrite unexpectedly changed convolution precision/names')
    finally:
        sys.setrecursionlimit(previous)
    spec = patched.get_spec()
    # extract_submodel exposes a raw NCHW tensor. The MIL graph already contains
    # the original image scale and normalization. Restore only the original
    # external feature descriptor; adding another image scale would be wrong.
    original_input = original.mlProgram.functions['main'].inputs[0]
    patched_input = spec.mlProgram.functions['main'].inputs[0]
    if original_input != patched_input:
        raise ValueError('Encoder MIL input type changed; cannot restore image descriptor')
    spec.description.input[0].CopyFrom(original.description.input[0])
    final = ct.models.MLModel(spec, weights_dir=patched.weights_dir, skip_model_load=True)
    return final, {'policy': FANOUT_POLICY, 'modifiedEdges': modifications,
                   'imageInterfaceRestoredFromSource': True}
