from __future__ import division
import argparse
import subprocess
from displ.build.alignment import get_all_syms, make_prefix

def get_prefixes():
    all_syms = get_all_syms()
    prefixes = [make_prefix(syms) for syms in all_syms]

    return prefixes

def _main():
    parser = argparse.ArgumentParser("Plot band structure for various TMD stacks",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--subdir", type=str, default=None,
            help="Subdirectory under work_base where calculation was run")
    args = parser.parse_args()

    for prefix in get_prefixes():
        band_cmd = ["python3", "bands.py", prefix, "--subdir", args.subdir, "--band_extrema", "--fermi_shift",
                "--minE", str(-3.0), "--maxE", str(3.0)]
        subprocess.run(band_cmd)

if __name__ == "__main__":
    _main()
