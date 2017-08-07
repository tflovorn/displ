from __future__ import division
import argparse
import os
import itertools
import json
import numpy as np
import matplotlib.pyplot as plt
from displ.build.build import _get_work, _get_base_path, band_path_labels
from displ.pwscf.parseScf import fermi_from_scf, latVecs_from_scf, alat_from_scf
from displ.wannier.extractHr import extractHr
from displ.wannier.bands import Hk, dHk_dk, d2Hk_dk
from displ.kdotp.linalg import nullspace
from displ.kdotp.model_weights_K import vec_linspace, top_valence_indices
from displ.kdotp.separability_K import get_layer_projections, get_total_orbitals
from displ.kdotp.effective_valence_K import (layer_basis_from_dm,
        array_with_rows, layer_Hamiltonian_0th_order, layer_Hamiltonian_ps,
        layer_Hamiltonian_mstar_inverses, correction_Hamiltonian_0th_order,
        correction_Hamiltonian_ps, correction_Hamiltonian_mstar_inverses,
        correction_Hamiltonian_PQ, H_kdotp, effective_mass_band)

def get_prefixes(global_prefix, subdir):
    base = _get_base_path(subdir)

    prefix_groups_path = os.path.join(base, "{}_prefix_groups.json".format(global_prefix))

    with open(prefix_groups_path, 'r') as fp:
        prefix_groups = json.load(fp)

    Es_and_prefixes = []
    for group in prefix_groups:
        for prefix in group:
            E = float(prefix.split('_')[-1])
            Es_and_prefixes.append((E, prefix))

    Es, prefixes = [], []
    for E, prefix in sorted(Es_and_prefixes, key=lambda x: x[0]):
        Es.append(E)
        prefixes.append(prefix)

    return Es, prefixes

def _main():
    np.set_printoptions(threshold=np.inf)

    parser = argparse.ArgumentParser("Plot evolution with electric field",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--global_prefix", type=str, default="WSe2_WSe2_WSe2",
            help="Prefix used for all calculations")
    parser.add_argument("--subdir", type=str, default=None,
            help="Subdirectory under work_base where calculation was run")
    args = parser.parse_args()

    Es, prefixes = get_prefixes(args.global_prefix, args.subdir)
    print(Es)
    print(prefixes)

    Ediffs, mstar_Gammas, mstar_Ks = [], [], []

    for prefix in prefixes:
        work = _get_work(args.subdir, prefix)
        wannier_dir = os.path.join(work, "wannier")
        scf_path = os.path.join(wannier_dir, "scf.out")

        E_F = fermi_from_scf(scf_path)
        latVecs = latVecs_from_scf(scf_path)
        alat_Bohr = 1.0
        R = 2 * np.pi * np.linalg.inv(latVecs.T)

        Gamma_cart = np.array([0.0, 0.0, 0.0])
        K_lat = np.array([1/3, 1/3, 0.0])
        K_cart = np.dot(K_lat, R)

        Hr_path = os.path.join(wannier_dir, "{}_hr.dat".format(prefix))
        Hr = extractHr(Hr_path)

        H_TB_Gamma = Hk(Gamma_cart, Hr, latVecs)
        Es_Gamma, U_Gamma = np.linalg.eigh(H_TB_Gamma)

        H_TB_K = Hk(K_cart, Hr, latVecs)
        Es_K, U_K = np.linalg.eigh(H_TB_K)

        top_Gamma = top_valence_indices(E_F, 2, Es_Gamma)
        top_K = top_valence_indices(E_F, 3, Es_Gamma) # TODO num_layers arg

        E_top_Gamma, E_top_K = Es_Gamma[top_Gamma[0]], Es_K[top_K[0]]

        Ediffs.append(E_top_Gamma - E_top_K)

        def Hfn(k):
            return Hk(k, Hr, latVecs)

        mstar_top_Gamma = effective_mass_band(Hfn, Gamma_cart, top_Gamma[0], alat_Bohr)
        mstar_top_K = effective_mass_band(Hfn, K_cart, top_K[0], alat_Bohr)

        mstar_Gammas.append(mstar_top_Gamma[0])
        mstar_Ks.append(mstar_top_K[0])

    plt.plot(Es, Ediffs, 'k.')
    plt.savefig("Ediffs.png", bbox_inches='tight', dpi=500)
    plt.clf()

    plt.plot(Es, mstar_Gammas, 'k.')
    plt.savefig("mstar_Gammas.png", bbox_inches='tight', dpi=500)
    plt.clf()

    plt.plot(Es, mstar_Ks, 'k.')
    plt.savefig("mstar_Ks.png", bbox_inches='tight', dpi=500)
    plt.clf()

if __name__ == "__main__":
    _main()
