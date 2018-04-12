import argparse
import re
import os
from pathlib import Path
from displ.build.util import _global_config
from displ.build.build import get_prefixes

def find_pattern(in_dir, pattern_re, depth):
    # Search in `in_dir` and recursively in each subdirectory up to `depth`.
    p = Path(in_dir)
    for x in p.iterdir():
        if x.is_dir() and depth > 0:
            find_in_subdir = find_pattern(str(x), pattern_re, depth - 1)
            if find_in_subdir is not None:
                return find_in_subdir
        else:
            if pattern_re.search(x.name) is not None:
                return str(x)

    return None

def _main():
    parser = argparse.ArgumentParser("Find calculations with missing files.",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--subdir", type=str, default=None,
            help="Subdirectory under work_base where calculation has been run")
    parser.add_argument("--global_prefix", type=str, default="WSe2_WSe2_WSe2",
            help="Prefix for calculation")
    parser.add_argument("--depth", type=int, default=1,
            help="How many subdirectories to search inside each calculation.")
    parser.add_argument("missing_pattern", type=str,
            help="Regular expression pattern identifying files to search for. Report calculations where no file matches the pattern.")
    args = parser.parse_args()

    gconf = _global_config()
    base_path = os.path.expandvars(gconf["work_base"])
    if args.subdir is not None:
        base_path = os.path.join(base_path, args.subdir)

    prefixes = get_prefixes(base_path, args.global_prefix)

    missing_pattern_re = re.compile(args.missing_pattern)

    missing = []
    for prefix in prefixes:
        in_dir = os.path.join(base_path, prefix)
        if find_pattern(in_dir, missing_pattern_re, args.depth) is None:
            missing.append(prefix)

    for missing_prefix in missing:
        print(missing_prefix)

if __name__ == "__main__":
    _main()
