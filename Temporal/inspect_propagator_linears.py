#!/usr/bin/env python3
"""Map the phone plan's expensive linear outputs to serialized tensor shapes.

Reads protobuf and package hashes only. No compilation, conversion or prediction.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from inspect_encoder_placement import sha, write_json
from inspect_export_graph import tensor_info


def inspect_block(block, location, inherited, targets):
    values = dict(inherited)
    for value in block.get('inputs', []):
        values[value['name']] = tensor_info(value.get('type', {}))
    operations = block.get('operations', [])
    producers = {}
    for index, operation in enumerate(operations):
        for value in operation.get('outputs', []):
            values[value['name']] = tensor_info(value.get('type', {}))
            producers[value['name']] = index

    def input_names(operation):
        return [b['name'] for a in operation.get('inputs', {}).values()
                for b in a.get('arguments', []) if 'name' in b]

    def row(index):
        operation = operations[index]
        return {
            'location': f'{location}/{index}', 'type': operation['type'],
            'inputs': {key: [{'name': b['name'], **values.get(b['name'], {})}
                             if 'name' in b else {'literal': b}
                             for b in argument.get('arguments', [])]
                       for key, argument in operation.get('inputs', {}).items()},
            'outputs': [{'name': value['name'], **values[value['name']]}
                        for value in operation.get('outputs', [])],
            'smallAttributes': {key: value for key, value in operation.get('attributes', {}).items()
                                if len(json.dumps(value)) <= 2048},
        }

    result = []
    for index, operation in enumerate(operations):
        names = {v['name'] for v in operation.get('outputs', [])}
        selected = names & targets
        if selected:
            if operation['type'] != 'linear' or len(selected) != 1:
                raise ValueError(f'Expected one linear output at {location}/{index}')
            entry = row(index)
            entry['selectedOutput'] = next(iter(selected))
            entry['inputProducers'] = [row(i) for i in sorted(
                {producers[n] for n in input_names(operation) if n in producers})]
            entry['outputConsumers'] = [row(i) for i, op in enumerate(operations)
                                        if names.intersection(input_names(op))]
            # MAC count describes the dense arithmetic shape, not device time.
            weight = entry['inputs'].get('weight', [])
            output = entry['outputs'][0].get('shape', [])
            if len(weight) == 1:
                shape = weight[0].get('shape', [])
                if (len(shape) == 2 and output and
                        all(isinstance(n, int) and n > 0 for n in shape + output)):
                    entry['denseMultiplyAccumulateCount'] = math.prod(output) * shape[1]
            result.append(entry)
        for child_index, child in enumerate(operation.get('blocks', [])):
            result.extend(inspect_block(child, f'{location}/{index}/block{child_index}', values, targets))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--plan-summary', type=Path, default=Path(__file__).resolve().parents[1] /
                        'Audit/owned-temporal-iphone17pro-compute-plan-summary.json')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report_path = args.output / 'report.json'
    report = {'scope': __doc__, 'completed': False, 'readyForDeviceValidation': False}
    try:
        manifest_path = args.run / 'models/manifest.json'
        manifest = json.loads(manifest_path.read_text())
        plan = json.loads(args.plan_summary.read_text())
        if plan.get('completed') is not True or plan.get('allPlansLoaded') is not True:
            raise ValueError('Expected a completed device plan')
        if sha(manifest_path) != plan['sourceModelsManifestSHA256']:
            raise ValueError('Run manifest differs from the inspected phone models')
        for key in ('contract', 'graphRevision', 'precisionPolicy'):
            if manifest.get(key) != plan[key]:
                raise ValueError(f'Run/plan mismatch: {key}')
        package = args.run / 'models/BWTemporalPropagator.mlpackage'
        expected = manifest['models']['Propagator']['files']
        actual = {str(p.relative_to(package)): sha(p) for p in package.rglob('*') if p.is_file()}
        if actual != expected:
            raise ValueError('Propagator package bytes changed since validation')
        package_manifest = json.loads((package / 'Manifest.json').read_text())
        root = package_manifest['itemInfoEntries'][package_manifest['rootModelIdentifier']]
        relative = Path(root['path'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Invalid package root path')
        from coremltools.proto import Model_pb2
        from google.protobuf.json_format import MessageToDict
        data = (package / 'Data' / relative).read_bytes()
        spec = Model_pb2.Model()
        spec.ParseFromString(data)
        if not spec.HasField('mlProgram'):
            raise ValueError('Expected ML Program')
        program = MessageToDict(spec.mlProgram)
        targets = set(plan['selectedPropagatorOutputs'])
        if not targets:
            raise ValueError('Plan summary selects no outputs')
        rows = []
        for name, function in program.get('functions', {}).items():
            inputs = {v['name']: tensor_info(v.get('type', {})) for v in function.get('inputs', [])}
            for specialization, block in function.get('blockSpecializations', {}).items():
                rows.extend(inspect_block(block, f'{name}/{specialization}', inputs, targets))
        if len(rows) != len(targets) or {r['selectedOutput'] for r in rows} != targets:
            raise ValueError('Selected plan outputs did not map uniquely to serialized operations')
        for row in rows:
            row['phonePlans'] = [
                {'requestedComputeUnits': p['requestedComputeUnits'], **op}
                for p in plan['plans'] if p['component'] == 'Propagator'
                for op in p['selectedOperations'] if row['selectedOutput'] in op['outputs']]
        report.update(completed=True, sourceManifestSHA256=sha(manifest_path),
                      sourcePlanSummarySHA256=sha(args.plan_summary),
                      sourceAttachmentSHA256=plan['sourceAttachmentSHA256'],
                      metadata=dict(spec.description.metadata.userDefined), files=actual, operations=rows)
        summary = {**report, 'operations': [
            {k: v for k, v in row.items() if k not in ('inputProducers', 'outputConsumers', 'smallAttributes')}
            for row in rows]}
        write_json(args.output / 'summary.json', summary)
    except Exception as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        write_json(report_path, report)
    print(f'Report: {report_path}', flush=True)
    print(f'Summary: {args.output / "summary.json"}', flush=True)


if __name__ == '__main__':
    main()
