import argparse
import os
from ase import Atoms
import ase.db
from displ.pwscf.build import build_pw2wan, build_bands, build_qe
from displ.wannier.build import Winfile
from displ.queue.queuefile import write_queuefile, write_job_group_files, write_launcherfiles
from displ.build.cell import make_cell
from displ.build.build import (make_qe_config, get_c_sep, get_wann_valence, get_num_bands,
        _write_qe_input, _write_queuefiles, _get_base_path, _get_work)
from displ.build.util import _base_dir, _global_config

def get_all_syms():
    return [["MoS2"], ["WS2"], ["MoSe2"], ["WSe2"],
            ["MoS2", "MoS2"], ["WS2", "WS2"], ["MoS2", "WS2"],
            ["MoSe2", "MoSe2"], ["WSe2", "WSe2"], ["MoSe2", "WSe2"],
            ["MoS2", "MoS2", "MoS2"], ["WS2", "WS2", "WS2"],
            ["MoS2", "MoS2", "WS2"], ["MoS2", "WS2", "MoS2"], ["WS2", "MoS2", "WS2"], ["MoS2", "WS2", "WS2"],
            ["MoSe2", "MoSe2", "MoSe2"], ["WSe2", "WSe2", "WSe2"],
            ["MoSe2", "MoSe2", "WSe2"], ["MoSe2", "WSe2", "MoSe2"], ["WSe2", "MoSe2", "WSe2"], ["MoSe2", "WSe2", "WSe2"]]

def get_stacking(stacking):
    if stacking == 'AB':
        AB_stacking = True
    elif stacking == 'AA':
        AB_stacking = False
    else:
        raise ValueError("unrecognized value for argument 'stacking'")

    return AB_stacking

def set_up_calculation(db, subdir, syms, AB_stacking, soc, vacuum_dist, D, xc, pp):
    # Choose separation between layers as if the system was a bulk system
    # where all layers are the same as the first layer here.
    # TODO -- is there a better strategy for this?
    c_sep = get_c_sep(db, syms[0])

    latvecs, at_syms, cartpos = make_cell(db, syms, c_sep, vacuum_dist, AB_stacking)

    system = Atoms(symbols=at_syms, positions=cartpos, cell=latvecs, pbc=True)
    system.center(axis=2)

    wann_valence, num_wann = get_wann_valence(system.get_chemical_symbols(), soc)
    num_bands = get_num_bands(num_wann)

    holes_per_cell = None
    qe_config = make_qe_config(system, D, holes_per_cell, soc, num_bands, xc, pp)

    prefix = '_'.join(syms)
    work = _get_work(subdir, prefix)

    wannier_dir = os.path.join(work, "wannier")
    if not os.path.exists(wannier_dir):
        os.makedirs(wannier_dir)

    bands_dir = os.path.join(work, "bands")
    if not os.path.exists(bands_dir):
        os.makedirs(bands_dir)

    dirs = {'scf': wannier_dir, 'nscf': wannier_dir, 'bands': bands_dir}

    qe_input = {}
    for calc_type in ['scf', 'nscf', 'bands']:
        qe_input[calc_type] = build_qe(system, prefix, calc_type, qe_config)
        _write_qe_input(prefix, dirs[calc_type], qe_input, calc_type)

    pw2wan_input = build_pw2wan(prefix, soc)
    pw2wan_path = os.path.join(wannier_dir, "{}.pw2wan.in".format(prefix))
    with open(pw2wan_path, 'w') as fp:
        fp.write(pw2wan_input)

    bands_post_input = build_bands(prefix)
    bands_post_path = os.path.join(bands_dir, "{}.bands_post.in".format(prefix))
    with open(bands_post_path, 'w') as fp:
        fp.write(bands_post_input)

    wannier_input = Winfile(system, qe_config, wann_valence, num_wann)
    win_path = os.path.join(wannier_dir, "{}.win".format(prefix))
    with open(win_path, 'w') as fp:
        fp.write(wannier_input)

    return prefix

def _main():
    parser = argparse.ArgumentParser("Build and run calculation over various TMD stacks",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--run", action="store_true",
            help="Run calculation after making inputs")
    parser.add_argument("--subdir", type=str, default=None,
            help="Subdirectory under work_base to run calculation")
    parser.add_argument("--stacking", type=str, default="AB",
            help="Stacking mode: 'AB' (2H) or 'AA' (1H)")
    parser.add_argument("--D", type=float, default=None,
            help="Displacement field (default: no field)")
    parser.add_argument("--xc", type=str, default="lda",
            help="Exchange-correlation functional (lda or pbe)")
    parser.add_argument("--pp", type=str, default="nc",
            help="Pseudopotential type ('nc' or 'paw')")
    args = parser.parse_args()

    soc = True
    all_syms = get_all_syms()
    AB_stacking = get_stacking(args.stacking)

    vacuum_dist = 20.0 # Angstrom

    db_path = os.path.join(_base_dir(), "c2dm.db")
    db = ase.db.connect(db_path)

    prefixes = []
    for syms in all_syms:
        p = set_up_calculation(db, args.subdir, syms, AB_stacking, soc, vacuum_dist, args.D, args.xc, args.pp)
        prefixes.append(p)

    global_prefix = "alignment"
    num_nodes = 2
    num_cores = 24*num_nodes
    queue_config = {"machine": "stampede2", "cores": num_cores, "nodes": num_nodes, "queue": "normal",
            "hours": 12, "minutes": 0, "wannier": True, "project": "A-ph9",
            "global_prefix": global_prefix, "max_jobs": 6,
            "outer_min": -10.0, "outer_max": 5.0,
            "inner_min": -8.0, "inner_max": 3.0,
            "subdir": args.subdir, "qe_bands":_global_config()['qe_bands']}

    _write_queuefiles(_get_base_path(args.subdir), prefixes, queue_config)

if __name__ == '__main__':
    _main()
