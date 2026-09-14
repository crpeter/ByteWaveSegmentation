#!/usr/bin/env python3
"""Restore bundled model weights without downloading anything."""
import hashlib
import json
import os
from pathlib import Path
import tempfile

ROOT = Path(__file__).resolve().parent


def matches(path, record):
    if not path.is_file() or path.stat().st_size != record['size']:
        return False
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest() == record['sha256']


def main():
    records = json.loads((ROOT / 'Audit/model-parts.json').read_text())
    for record in records:
        destination = ROOT / record['path']
        if matches(destination, record):
            print('Already verified:', record['path'])
            continue
        for part in record['parts']:
            if not matches(ROOT / part['path'], part):
                raise RuntimeError('Missing or damaged model part: ' + part['path'])
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as output:
                temporary = Path(output.name)
                for part in record['parts']:
                    with (ROOT / part['path']).open('rb') as source:
                        for block in iter(lambda: source.read(1024 * 1024), b''):
                            output.write(block)
            if not matches(temporary, record):
                raise RuntimeError('Reassembled model checksum failed: ' + record['path'])
            os.replace(temporary, destination)
            print('Restored and verified:', record['path'])
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    manifest = json.loads((ROOT / 'Audit/download-manifest.json').read_text())
    for record in manifest['files']:
        path = record['path']
        local = ROOT / ('Models/' + path[len('models/'):] if path.startswith('models/') else 'ThirdParty/' + path)
        if not matches(local, record):
            raise RuntimeError('Bundled file checksum failed: ' + str(local))
    print('All four model packages verified. Open ByteWaveSegmentationProbe.xcodeproj.')


if __name__ == '__main__':
    main()
