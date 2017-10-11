import argparse
import os
from copy import deepcopy
import json
import numpy as np
from ase import Atoms
import ase.db
from displ.pwscf.build import build_pw2wan, build_bands, build_qe
from displ.wannier.build import Winfile
from displ.queue.queuefile import write_queuefile, write_job_group_files, write_launcherfiles
from displ.build.cell import make_cell, get_layer_system, h_from_2H
from displ.build.util import _base_dir, _global_config

def make_qe_config(system, D, holes_per_cell, soc, num_bands, xc, pp):
    latconst = 1.0 # positions in system are given in units of Angstrom

    # Assume we take the electric field to be along the c axis.
    edir = 3
    
    # Choose the descending voltage region of the voltage sawtooth to be
    # 10% of the vacuum.
    min_c, max_c = _get_c_extrema(system)
    # Assume positions have not been wrapped (i.e. min_c and max_c are not
    # separated by vacuum; the vacuum is between max_c and min_c + 1).
    vac = min_c + 1 - max_c
    eopreg = 0.1*vac
    # Choose the center of the descending voltage region to be the center
    # of the vacuum:
    # (min(c) + 1 + max(c))/2 = (emaxpos + emaxpos + eopreg)/2.
    emaxpos = (min_c + 1 + max_c - eopreg)/2

    # D is in V/nm and eamp is in Ha a.u.:
    # 1 a.u. = 5.14220632e2 V/nm --> 1 V/nm = 1.9446905e-3 a.u.
    eamp = 1.9446905e-3 * D

    pseudo = get_pseudo(system.get_chemical_symbols(), soc, pp)
    pseudo_dir = get_pseudo_dir(soc, xc, pp)
    ecutwfc, ecutrho = get_ecut(pp)

    weight = get_weight(system)

    conv_thr = {"scf": 1e-8, "nscf": 1e-8, "bands": 1e-8}

    degauss = 0.02

    Nk_scf = 18
    Nk_nscf = 18
    Nk_bands = 20
    Nk = {"scf": [Nk_scf, Nk_scf, 1], "nscf": [Nk_nscf, Nk_nscf, 1], "bands": Nk_bands}

    band_path = [[0.0, 0.0, 0.0], # Gamma
        [0.5, 0.0, 0.0], # M
        [1/3, 1/3, 0.0], # K
        [0.0, 0.0, 0.0]] # Gamma

    qe_config = {"pseudo_dir": pseudo_dir, "pseudo": pseudo, "soc": soc, "latconst": latconst, 
            "num_bands": num_bands, "weight": weight, "ecutwfc": ecutwfc, "ecutrho": ecutrho,
            "degauss": degauss, "conv_thr": conv_thr, "Nk": Nk, "band_path": band_path,
            "edir": edir, "emaxpos": emaxpos, "eopreg": eopreg, "eamp": eamp,
            "tot_charge": holes_per_cell}

    return qe_config

def band_path_labels():
    return ["$\\Gamma$", "$M$", "$K$", "$\\Gamma$"]

def _get_c_extrema(system):
    all_pos = system.get_scaled_positions()
    cs = [r[2] for r in all_pos]
    return min(cs), max(cs)

def get_pseudo(at_syms, soc=True, pp='nc'):
    pseudo = {}

    for sym in at_syms:
        if sym in pseudo:
            continue

        if pp == 'nc':
            if soc:
                pseudo_name = "{}_r.oncvpsp.upf".format(sym)
            else:
                pseudo_name = "{}.oncvpsp.upf".format(sym)
        elif pp == 'paw': # NOTE/TODO - this breaks for non-lda paw
            if soc:
                pseudo_name = "{}.rel-pz_high_psl.1.0.0.UPF".format(sym)
            else:
                raise ValueError("paw/non-soc not implemented")
        else:
            raise ValueError("unsupported pp value")

        pseudo[sym] = pseudo_name

    return pseudo

def get_pseudo_dir(soc, xc, pp):
    if pp == 'nc':
        if soc:
            if xc == 'lda':
                return os.path.join(_base_dir(), "pseudo", "lda_soc")
            else:
                return os.path.join(_base_dir(), "pseudo", "soc")
        else:
            if xc == 'lda':
                return os.path.join(_base_dir(), "pseudo", "lda_no_soc")
            else:
                return os.path.join(_base_dir(), "pseudo", "no_soc")
    elif pp == 'paw':
        if soc and xc == 'lda':
            return os.path.join(_base_dir(), "pseudo", "pslibrary_lda_soc_paw")
        else:
            raise ValueError("paw/non-soc not implemented")

def get_ecut(pp):
    if pp == 'nc':
        ecutwfc = 60.0
        ecutrho = 240.0
    elif pp == 'paw':
        ecutwfc = 85.0
        ecutrho = 395.0
    else:
        raise ValueError("unsupported pp value")

    return ecutwfc, ecutrho

def get_weight(system):
    syms = system.get_chemical_symbols()
    all_weights = system.get_masses()

    weight = {}
    for sym, sym_weight in zip(syms, all_weights):
        weight[sym] = sym_weight

    return weight

def get_c_sep(db, sym):
    '''Choose separation between layers for a system in which each layer is
    given by the layer `sym`. This separation is assumed to be equal to
    
    (c_bulk - 2*h) / 2

    Where c_bulk is the 2H bulk c-axis lattice constant (which includes 2 layers)
    and h is the separation of chalcogens within the layer (giving the c-axis
    extent of the layer).

    i.e. the bulk cell has the structure:

    X 0
    M h/2
    X h
    --
    X h + c_sep
    M 3*h/2 + c_sep
    X 2*h + c_sep
    --
    X 2*h + 2*c_sep = c_bulk

    ==> 2*c_sep = c_bulk - 2*h ==> c_sep = (c_bulk - 2*h)/2

    Bulk c-axis lattice constants here taken from Table I of:
    Yun et al., PRB 85, 033305 (2012)
    The a and c lattice constants given in this table are experimental values.
    '''
    c_bulk_values = {"MoS2": 12.296, "MoSe2": 12.939,
            "WS2": 12.349, "WSe2": 12.976}
    c_bulk = c_bulk_values[sym]

    system = get_layer_system(db, sym, 'H')
    h = h_from_2H(system)

    return (c_bulk - 2*h)/2

def get_wann_valence(at_syms, soc=True):
    Ms = ["Mo", "W"]
    Xs = ["S", "Se", "Te"]

    # num_wann should equal nspin*(nlayers*9 + 2*nlayers*6)
    num_wann = 0

    wann_valence = {}
    for sym in at_syms:
        if sym in Ms:
            if soc:
                num_wann += 10
            else:
                num_wann += 5

            wann_valence[sym] = ["d"]
        else:
            if soc:
                num_wann += 6
            else:
                num_wann += 3

            wann_valence[sym] = ["p"]

    return wann_valence, num_wann

def get_num_bands(num_wann):
    extra_bands_factor = 2.0
    num_bands = int(np.ceil(extra_bands_factor*num_wann))
    if num_bands % 2 == 1:
        num_bands += 1

    return num_bands

def _extract_syms(syms_str):
    syms = syms_str.split(';')
    return syms

def _get_base_path(subdir):
    work = os.path.expandvars(_global_config()['work_base'])
    if subdir is not None:
        work = os.path.join(work, subdir)

    return work

def _get_work(subdir, prefix):
    work = os.path.join(_get_base_path(subdir), prefix)

    if not os.path.exists(work):
        os.makedirs(work)

    return work

def _write_qe_input(prefix, file_dir, qe_input, calc_type):
    file_path = os.path.join(file_dir, "{}.{}.in".format(prefix, calc_type))
    with open(file_path, 'w') as fp:
        fp.write(qe_input[calc_type])

def group_jobs(config, prefix_list):
    max_jobs = config["max_jobs"]

    groups = []
    for i, prefix in enumerate(prefix_list):
        group_id = i % max_jobs
        if len(groups) <= group_id:
            groups.append([])

        groups[group_id].append(prefix)

    return groups

def _write_queuefiles(base_path, prefixes, config):
    for prefix in prefixes:
        _write_system_queuefile(base_path, prefix, config)

    prefix_groups = group_jobs(config, prefixes)
    _write_prefix_groups(base_path, config["global_prefix"], prefix_groups)
    config["base_path"] = base_path

    wan_setup_group_config = deepcopy(config)
    wan_setup_group_config["calc"] = "wan_setup"
    write_job_group_files(wan_setup_group_config, prefix_groups)

    pw_post_group_config = deepcopy(config)
    pw_post_group_config["calc"] = "pw_post"
    pw_post_group_config["nodes"] = 1
    cores_per_node = int(config["cores"] / config["nodes"])
    pw_post_group_config["cores"] = cores_per_node
    write_job_group_files(pw_post_group_config, prefix_groups)

    launcher_config = deepcopy(config)
    launcher_config["prefix_list"] = prefixes
    launcher_config["calc"] = "wan_run"
    num_systems = len(prefixes)
    # Do 4 Wannier90 runs per core (Wannier90 runtime is quick - minimize time
    # in queue).
    num_wannier_nodes = int(np.ceil(num_systems / (4*cores_per_node)))
    num_wannier_cores = num_wannier_nodes * cores_per_node
    launcher_config["nodes"] = num_wannier_nodes
    launcher_config["cores"] = num_wannier_cores
    write_launcherfiles(launcher_config)

    return prefix_groups

def _write_system_queuefile(base_path, prefix, config):
    config["base_path"] = base_path
    config["prefix"] = prefix

    for calc in ["wan_setup", "pw_post", "wan_run"]:
        _write_queuefile_calc(config, calc)

def _write_queuefile_calc(config, calc):
    calc_config = deepcopy(config)
    calc_config["calc"] = calc
    write_queuefile(calc_config)

def _write_prefix_groups(base_path, global_prefix, prefix_groups):
    groups_path = _prefix_groups_path(base_path, global_prefix)
    with open(groups_path, 'w') as fp:
        json.dump(prefix_groups, fp)

def _prefix_groups_path(base_path, global_prefix):
    groups_path = os.path.join(base_path, "{}_prefix_groups.json".format(global_prefix))
    return groups_path

def _main():
    parser = argparse.ArgumentParser("Build and run calculation over displacement field values",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--run", action="store_true",
            help="Run calculation after making inputs")
    parser.add_argument("--subdir", type=str, default=None,
            help="Subdirectory under work_base to run calculation")
    parser.add_argument("--syms", type=str, default="WSe2;WSe2;WSe2",
            help="Semicolon-separated list of atomic composition of layers. Format example: WSe2;MoSe2;MoS2")
    parser.add_argument("--stacking", type=str, default="AB",
            help="Stacking mode: 'AB' (2H) or 'AA' (1T)")
    parser.add_argument("--minD", type=float, default=0.01,
            help="Minimum displacement field in V/nm")
    parser.add_argument("--maxD", type=float, default=0.5,
            help="Maximum displacement field in V/nm")
    parser.add_argument("--numD", type=int, default=10,
            help="Number of displacement field steps")
    parser.add_argument("--holes", type=float, default=None,
            help="Holes per unit area (cm^{-2}) to add (default = 0)")
    parser.add_argument("--no_soc", action="store_true",
            help="Turn off spin-orbit coupling")
    parser.add_argument("--xc", type=str, default="lda",
            help="Exchange-correlation functional (lda or pbe)")
    parser.add_argument("--pp", type=str, default="nc",
            help="Pseudopotential type ('nc' or 'paw')")
    args = parser.parse_args()

    syms = _extract_syms(args.syms)
    global_prefix = "_".join(syms)

    soc = not args.no_soc

    db_path = os.path.join(_base_dir(), "c2dm.db")
    db = ase.db.connect(db_path)

    # Choose separation between layers as if the system was a bulk system
    # where all layers are the same as the first layer here.
    # TODO -- is there a better strategy for this?
    c_sep = get_c_sep(db, syms[0])

    vacuum_dist = 20.0 # Angstrom

    if args.stacking == 'AB':
        AB_stacking = True
    elif args.stacking == 'AA':
        AB_stacking = False
    else:
        raise ValueError("unrecognized value for argument 'stacking'")

    latvecs, at_syms, cartpos = make_cell(db, syms, c_sep, vacuum_dist, AB_stacking)

    # latvecs units = Angstrom
    cell_area_Angstrom2 = np.linalg.norm(np.cross(latvecs[0], latvecs[1]))
    if args.holes is not None:
        holes_per_cell = args.holes * cell_area_Angstrom2 * 1e-16
    else:
        holes_per_cell = None

    system = Atoms(symbols=at_syms, positions=cartpos, cell=latvecs, pbc=True)
    system.center(axis=2)

    wann_valence, num_wann = get_wann_valence(system.get_chemical_symbols(), soc)
    num_bands = get_num_bands(num_wann)

    Ds = np.linspace(args.minD, args.maxD, args.numD)

    prefixes = []
    for D in Ds:
        qe_config = make_qe_config(system, D, holes_per_cell, soc, num_bands, args.xc, args.pp)

        prefix = "{}_{}".format(global_prefix, str(D))
        prefixes.append(prefix)
        work = _get_work(args.subdir, prefix)

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

    num_nodes = 2
    num_cores = 24*num_nodes
    queue_config = {"machine": "ls5", "cores": num_cores, "nodes": num_nodes, "queue": "normal",
            "hours": 12, "minutes": 0, "wannier": True, "project": "A-ph9",
            "global_prefix": global_prefix, "max_jobs": 1,
            "outer_min": -10.0, "outer_max": 5.0,
            "inner_min": -8.0, "inner_max": 3.0,
            "subdir": args.subdir, "qe_bands":_global_config()['qe_bands']}

    _write_queuefiles(_get_base_path(args.subdir), prefixes, queue_config)

if __name__ == '__main__':
    _main()
