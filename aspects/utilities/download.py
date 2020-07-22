import sys

import requests
from tqdm import tqdm


def http_get(url, path):
    file_binary = open(path, "wb")
    req = requests.get(url, stream=True)
    if req.status_code != 200:
        print("Exception when trying to download {}. Response {}".format(
            url, req.status_code), file=sys.stderr)
        req.raise_for_status()

    content_length = req.headers.get('Content-Length')
    total = int(content_length) if content_length is not None else None
    progress = tqdm(unit="B", total=total, unit_scale=True)
    for chunk in req.iter_content(chunk_size=1024):
        if chunk:  # filter out keep-alive new chunks
            progress.update(len(chunk))
            file_binary.write(chunk)
    progress.close()
