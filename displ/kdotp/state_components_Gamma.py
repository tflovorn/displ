from __future__ import division
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from displ.build.build import _get_work
from displ.pwscf.parseScf import fermi_from_scf
from displ.wannier.extractHr import extractHr
from displ.wannier.bands import Hk_recip
from displ.kdotp.model_weights_K import top_valence_indices
from displ.kdotp.separability_K import get_layer_orbitals, get_layer_projections
from displ.kdotp.effective_valence_Gamma import get_layer_basis_Gamma

def get_state_weights(layer_orbitals, state):
    weights = {"0": []}

    for z_orb in layer_orbitals:
        pz_up_bot, pz_down_bot = [z_orb[i] for i in range(0, 2)]
        pz_up_top, pz_down_top = [z_orb[i] for i in range(6, 8)]
        dz2_up, dz2_down = [z_orb[i] for i in range(12, 14)]

        lz0_tot = 0.0
        for i in [pz_up_bot, pz_down_bot, pz_up_top, pz_down_top, dz2_up, dz2_down]:
            lz0_tot += abs(state[i, 0])**2

        weights["0"].append(lz0_tot)

    return weights

def print_orbital_weights_eigenstates_Gamma(subdir, prefix):
    num_layers = 3
    assert(num_layers == 3) # != 3 unimplemented

    work = _get_work(subdir, prefix)
    wannier_dir = os.path.join(work, "wannier")
    scf_path = os.path.join(wannier_dir, "scf.out")
    wout_path = os.path.join(wannier_dir, "{}.wout".format(prefix))

    E_F = fermi_from_scf(scf_path)
    layer_orbitals = get_layer_orbitals(wout_path, num_layers)

    Hr_path = os.path.join(wannier_dir, "{}_hr.dat".format(prefix))
    Hr = extractHr(Hr_path)

    Gamma_lat = np.array([0.0, 0.0, 0.0])
    H_TB_Gamma = Hk_recip(Gamma_lat, Hr)
    Es, U = np.linalg.eigh(H_TB_Gamma)
    top = top_valence_indices(E_F, 2*num_layers, Es)

    eigenstate_weights = [get_state_weights(layer_orbitals, U[:, [t]]) for t in top]

    print("eigenstate weights")
    print(eigenstate_weights)

def print_orbital_weights_dm_Gamma(subdir, prefix):
    num_layers = 3
    assert(num_layers == 3) # != 3 unimplemented

    work = _get_work(subdir, prefix)
    wannier_dir = os.path.join(work, "wannier")
    scf_path = os.path.join(wannier_dir, "scf.out")
    wout_path = os.path.join(wannier_dir, "{}.wout".format(prefix))

    E_F = fermi_from_scf(scf_path)
    layer_orbitals = get_layer_orbitals(wout_path, num_layers)
    Pzs = get_layer_projections(wout_path, num_layers)

    Hr_path = os.path.join(wannier_dir, "{}_hr.dat".format(prefix))
    Hr = extractHr(Hr_path)

    Gamma_lat = np.array([0.0, 0.0, 0.0])
    H_TB_Gamma = Hk_recip(Gamma_lat, Hr)
    Es, U = np.linalg.eigh(H_TB_Gamma)
    top = top_valence_indices(E_F, 2*num_layers, Es)

    layer_weights, layer_basis = get_layer_basis_Gamma(U, top, Pzs)

    dm_basis_weights = [get_state_weights(layer_orbitals, s) for s in layer_basis]

    print("dm basis weights")
    print(dm_basis_weights)

def _main():
    parser = argparse.ArgumentParser("Plot orbital contributions at K",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("prefix", type=str,
            help="Prefix for calculation")
    parser.add_argument("--subdir", type=str, default=None,
            help="Subdirectory under work_base where calculation was run")
    args = parser.parse_args()

    print_orbital_weights_eigenstates_Gamma(args.subdir, args.prefix)
    print_orbital_weights_dm_Gamma(args.subdir, args.prefix)

if __name__ == "__main__":
    _main()
