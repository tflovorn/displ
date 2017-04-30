import os
import inspect
import json

def _base_dir():
    frame = inspect.getfile(inspect.currentframe())
    this_dir = os.path.dirname(os.path.abspath(frame))
    base_dir = os.path.join(this_dir, "..", "..")
    return os.path.normpath(base_dir)

def _global_config():
    base = _base_dir()
    gconf_path = os.path.join(base, "global_config.json")

    with open(gconf_path, 'r') as fp:
        gconf = json.load(fp)

    return gconf
