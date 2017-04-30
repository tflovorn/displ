import argparse
import os
import numpy as np
from ase import Atoms
import ase.db
from displ.pwscf.build import build_pw2wan, build_bands, build_qe
from displ.wannier.build import Winfile
from displ.build.cell import make_cell
from displ.build.util import _base_dir, _global_config

def make_qe_config(system, soc, num_bands, xc, pp):
    latconst = 1.0 # positions in system are given in units of Angstrom

    pseudo = get_pseudo(system.get_chemical_symbols(), soc, pp)
    pseudo_dir = get_pseudo_dir(soc, xc, pp)
    ecutwfc, ecutrho = get_ecut(pp)

    weight = get_weight(system)

    conv_thr = {"scf": 1e-8, "nscf": 1e-8, "bands": 1e-8}

    degauss = 0.02

    Nk_scf = 18
    Nk_nscf = 9
    Nk_bands = 20
    Nk = {"scf": [Nk_scf, Nk_scf, 1], "nscf": [Nk_nscf, Nk_nscf, 1], "bands": Nk_bands}

    band_path = [[0.0, 0.0, 0.0], # Gamma
        [0.5, 0.0, 0.0], # M
        [1/3, 1/3, 0.0], # K
        [0.0, 0.0, 0.0]] # Gamma

    qe_config = {"pseudo_dir": pseudo_dir, "pseudo": pseudo, "soc": soc, "latconst": latconst, 
            "num_bands": num_bands, "weight": weight, "ecutwfc": ecutwfc, "ecutrho": ecutrho,
            "degauss": degauss, "conv_thr": conv_thr, "Nk": Nk, "band_path": band_path}

    return qe_config

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

def get_c_bulk(sym):
    '''Bulk c-axis lattice constants from Table I of:
    Yun et al., PRB 85, 033305 (2012)
    The a and c lattice constants given in this table are experimental values.
    '''
    c_bulk_values = {"MoS2": 12.296, "MoSe2": 12.939,
            "WS2": 12.349, "WSe2": 12.976}
    return c_bulk_values[sym]

def get_wann_valence(at_syms, soc=True):
    Ms = ["Mo", "W"]
    Xs = ["S", "Se", "Te"]

    # num_wann should equal nspin*(nlayers*9 + 2*nlayers*6)
    num_wann = 0

    wann_valence = {}
    for sym in at_syms:
        if sym in wann_valence:
            continue

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

def _get_work(subdir, prefix):
    work = os.path.expandvars(_global_config()['work_base'])
    if subdir is not None:
        work = os.path.join(work, subdir)

    work = os.path.join(work, prefix)

    if not os.path.exists(work):
        os.makedirs(work)

    return work

def _write_qe_input(prefix, file_dir, qe_input, calc_type):
    file_path = os.path.join(file_dir, "{}.{}.in".format(prefix, calc_type))
    with open(file_path, 'w') as fp:
        fp.write(qe_input[calc_type])

def _main():
    parser = argparse.ArgumentParser("Build and run calculation over displacement field values",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--run", action="store_true",
            help="Run calculation after making inputs")
    parser.add_argument("--subdir", type=str, default=None,
            help="Subdirectory under work_base to run calculation")
    parser.add_argument("--syms", type=str, default="WSe2;WSe2;WSe2",
            help="Semicolon-separated list of atomic composition of layers. Format example: WSe2;MoSe2;MoS2")
    parser.add_argument("--minD", type=float, default=0.1,
            help="Minimum displacement field")
    parser.add_argument("--maxD", type=float, default=0.1,
            help="Maximum displacement field")
    parser.add_argument("--numD", type=int, default=10,
            help="Number of displacement field steps")
    parser.add_argument("--soc", action="store_true",
            help="Use spin-orbit coupling")
    parser.add_argument("--xc", type=str, default="lda",
            help="Exchange-correlation functional (lda or pbe)")
    parser.add_argument("--pp", type=str, default="nc",
            help="Pseudopotential type ('nc' or 'paw')")
    args = parser.parse_args()

    syms = _extract_syms(args.syms)

    db_path = os.path.join(_base_dir(), "c2dm.db")
    db = ase.db.connect(db_path)

    # Choose separation between layers as the bulk separation between
    # layers of the first type.
    # TODO -- is there a better strategy for this?
    c_bulk = get_c_bulk(syms[0])

    vacuum_dist = 20.0 # Angstrom

    latvecs, at_syms, cartpos = make_cell(db, syms, c_bulk, vacuum_dist)

    system = Atoms(symbols=at_syms, positions=cartpos, cell=latvecs, pbc=True)

    valence, num_wann = get_wann_valence(system.get_chemical_symbols(), args.soc)
    num_bands = get_num_bands(num_wann)

    qe_config = make_qe_config(system, args.soc, num_bands, args.xc, args.pp)

    prefix = "tmp"
    work = _get_work(args.subdir, prefix)

    wannier_dir = os.path.join(work, "wannier")
    if not os.path.exists(wannier_dir):
        os.makedirs(wannier_dir)

    bands_dir = os.path.join(work, "bands")
    if not os.path.exists(bands_dir):
        os.makedirs(bands_dir)

    calc_type = 'scf'
    qe_input = {}
    qe_input['scf'] = build_qe(system, prefix, calc_type, qe_config)
    _write_qe_input(prefix, wannier_dir, qe_input, 'scf')

if __name__ == '__main__':
    _main()
