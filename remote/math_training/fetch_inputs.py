"""Download only the 12 pinned training inputs from the user-provided Drive folder."""
import argparse
from pathlib import Path
from core import read, sha


def fetch(destination, downloader=None):
    if downloader is None:
        import gdown
        downloader = gdown.download
    manifest = read(Path(__file__).parent / 'input_manifest.json')
    for item in manifest['files']:
        path = destination / item['path']
        if not path.resolve().is_relative_to(destination.resolve()):
            raise ValueError('Invalid input manifest path')
        if path.exists() and sha(path) == item['sha256']:
            print('Verified:', item['path'], flush=True)
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + '.download')
        result = downloader(id=item['drive_id'], output=str(tmp), quiet=False)
        if not result or sha(tmp) != item['sha256']:
            raise ValueError('Download missing/hash mismatch: ' + item['path'])
        tmp.replace(path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', required=True, type=Path)
    fetch(parser.parse_args().destination.expanduser().resolve())
