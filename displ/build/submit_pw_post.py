import argparse
import os
from copy import deepcopy
from displ.build.util import _global_config
from displ.build.build import get_prefix_groups
from displ.queue.queuefile import mpi_procs_per_node
from displ.queue.internal import enqueue

def submit_pw_post(base_path, config, prefix_groups):
    config["base_path"] = base_path

    for i in range(len(prefix_groups)):
        dv_config = deepcopy(config)
        dv_config["calc"] = "pw_post_group"
        dv_config["prefix"] = str(i)
        enqueue(dv_config)

def _main():
    parser = argparse.ArgumentParser("Run postprocessing to set up Wannier90 calculation",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--subdir", type=str, default=None,
            help="Subdirectory under work_base to run calculation")
    parser.add_argument("--global_prefix", type=str, default="WSe2_WSe2_WSe2",
            help="Prefix for calculation")
    args = parser.parse_args()

    gconf = _global_config()
    base_path = os.path.expandvars(gconf["work_base"])
    if args.subdir is not None:
        base_path = os.path.join(base_path, args.subdir)

    if "qe_bands" in gconf:
        qe_bands_dir = os.path.expandvars(gconf["qe_bands"])
        qe_bands_path = os.path.join(qe_bands_dir, "bands.x")
    else:
        qe_bands_path = "bands.x"

    calc = "pw_post"
    prefix_groups = get_prefix_groups(base_path, args.global_prefix)

    machine = "stampede2"
    cores_per_node = mpi_procs_per_node(machine)
    config = {"machine": machine, "cores": cores_per_node, "nodes": 1, "queue": "normal",
            "hours": 48, "minutes": 0, "wannier": True, "project": "A-ph9",
            "global_prefix": args.global_prefix, "max_jobs": 24,
            "qe_bands": qe_bands_path}

    submit_pw_post(base_path, config, prefix_groups)

if __name__ == "__main__":
    _main()
