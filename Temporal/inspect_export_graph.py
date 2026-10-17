#!/usr/bin/env python3
"""Inspect serialized MIL shapes/indexing without loading or executing models."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


WATCH = {'non_zero', 'gather', 'gather_nd', 'gather_along_axis', 'scatter',
         'scatter_nd', 'scatter_along_axis', 'select', 'concat'}


def tensor_info(value):
    tensor = value.get('tensorType')
    if tensor is None:
        return {'type': value}
    dims = [int(d['constant'].get('size', 0)) if 'constant' in d else 'unknown'
            for d in tensor.get('dimensions', [])]
    return {'dtype': tensor.get('dataType'), 'shape': dims}


def inspect_block(block, location, inherited):
    values = dict(inherited)
    for item in block.get('inputs', []):
        values[item['name']] = tensor_info(item.get('type', {}))
    ops = block.get('operations', [])
    producers = {}
    for index, op in enumerate(ops):
        for item in op.get('outputs', []):
            values[item['name']] = tensor_info(item.get('type', {}))
            producers[item['name']] = index

    def row(index):
        op = ops[index]
        inputs = {}
        for key, argument in op.get('inputs', {}).items():
            inputs[key] = [{'name': b['name'], **values.get(b['name'], {})}
                           if 'name' in b else {'literal': b}
                           for b in argument.get('arguments', [])]
        return {'index': index, 'type': op['type'], 'inputs': inputs,
                'smallAttributes': {k: v for k, v in op.get('attributes', {}).items()
                                    if len(json.dumps(v)) <= 2048},
                'outputs': [{'name': v['name'], **values[v['name']]}
                            for v in op.get('outputs', [])]}

    suspects = set()
    for i, op in enumerate(ops):
        if op['type'] in WATCH or any(
                0 in values[v['name']].get('shape', []) or
                'unknown' in values[v['name']].get('shape', [])
                for v in op.get('outputs', [])):
            suspects.add(i)
    context = set(suspects)
    suspect_values = {v['name'] for i in suspects for v in ops[i].get('outputs', [])}
    for i, op in enumerate(ops):
        names = {b['name'] for a in op.get('inputs', {}).values()
                 for b in a.get('arguments', []) if 'name' in b}
        if i in suspects:
            context.update(producers[n] for n in names if n in producers)
        if names & suspect_values:
            context.add(i)
    result = [{'location': location, 'operationCounts': dict(Counter(o['type'] for o in ops)),
               'suspectIndices': sorted(suspects), 'operationsWithImmediateContext':
               [row(i) for i in sorted(context)]}]
    for i, op in enumerate(ops):
        for j, child in enumerate(op.get('blocks', [])):
            result.extend(inspect_block(child, f'{location}/op{i}/block{j}', values))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    # Parse package bytes directly; never construct MLModel or call predict/convert.
    from coremltools.proto import Model_pb2
    from google.protobuf.json_format import MessageToDict
    args.output.mkdir(parents=True, exist_ok=False)
    report = {'scope': 'Static serialized graph inspection; flagged operations are candidates, not proven failures.',
              'readyForDeviceValidation': False, 'models': {}}
    try:
        for name in ('ImageEncoder', 'Initializer', 'InitialMemoryEncoder', 'Propagator'):
            package = args.models / f'BWTemporal{name}.mlpackage'
            manifest = json.loads((package / 'Manifest.json').read_text())
            entry = manifest['itemInfoEntries'][manifest['rootModelIdentifier']]
            relative = Path(entry['path'])
            if relative.is_absolute() or '..' in relative.parts:
                raise ValueError('Invalid package model path')
            data = (package / 'Data' / relative).read_bytes()
            spec = Model_pb2.Model()
            spec.ParseFromString(data)
            if not spec.HasField('mlProgram'):
                raise ValueError(f'{name} is not an ML program')
            metadata = dict(spec.description.metadata.userDefined)
            if metadata.get('bytewave.contract') != 'bytewave.edgetam-temporal-owned.v2':
                raise ValueError(f'Unexpected contract: {name}')
            program = MessageToDict(spec.mlProgram)
            blocks = []
            for function_name, function in program.get('functions', {}).items():
                inputs = {v['name']: tensor_info(v.get('type', {}))
                          for v in function.get('inputs', [])}
                for key, block in function.get('blockSpecializations', {}).items():
                    blocks.extend(inspect_block(block, f'{function_name}/{key}', inputs))
            if not blocks:
                raise ValueError(f'No MIL blocks found: {name}')
            report['models'][name] = {'modelSpecSHA256': hashlib.sha256(data).hexdigest(),
                                      'metadata': metadata, 'blocks': blocks}
            print(name, 'flagged operations:', sum(len(b['suspectIndices']) for b in blocks), flush=True)
    except Exception as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        (args.output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(f"Report: {args.output / 'report.json'}", flush=True)


if __name__ == '__main__':
    main()
