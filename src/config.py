import os

_POSSIBLE_ROOT_DIRS = [
    '/mnt/Data/cs5242-project',
    '/mnt/HGST6/cs5242-project'
]

ROOT_DIR = None
for dirname in _POSSIBLE_ROOT_DIRS:
    if os.path.exists(dirname):
        ROOT_DIR = dirname
