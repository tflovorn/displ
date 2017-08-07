from __future__ import division
import argparse
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from displ.build.build import _get_work, _get_base_path
from displ.pwscf.parseScf import fermi_from_scf, latVecs_from_scf, alat_from_scf
from displ.wannier.extractHr import extractHr
from displ.wannier.bands import Hk
from displ.kdotp.model_weights_K import top_valence_indices
from displ.kdotp.effective_valence_K import effective_mass_band
from displ.kdotp.separability_K import get_layer_projections
from displ.kdotp.efield import _bohr_per_Angstrom, _e_C, hole_distribution, decimal_format

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

    E_V_nms, prefixes = get_prefixes(args.global_prefix, args.subdir)
    print(E_V_nms)
    print(prefixes)

    d_A = 6.488 # Angstrom
    d_bohr = _bohr_per_Angstrom * d_A

    hole_density_cm2 = 8e12
    hole_density_bohr2 = hole_density_cm2 / (10**8 * _bohr_per_Angstrom)**2

    sigma_layer_initial = (1/3) * _e_C * hole_density_bohr2

    sigmas_initial = sigma_layer_initial * np.array([1.0, 1.0, 1.0])

    # Dielectric constant of WSe2:
    # Kim et al., ACS Nano 9, 4527 (2015).
    # http://pubs.acs.org/doi/abs/10.1021/acsnano.5b01114
    epsilon_r = 7.2

    Pzs = get_layer_projections(3)  # TODO num_layers arg

    tol_abs = 1e-6 * sum(sigmas_initial)
    tol_rel = 1e-6

    Ediffs, mstar_Gammas, mstar_Ks = [], [], []
    nh_Gammas_frac, nh_Ks_frac = [], []

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
        top_K = top_valence_indices(E_F, 3, Es_Gamma)

        E_top_Gamma, E_top_K = Es_Gamma[top_Gamma[0]], Es_K[top_K[0]]

        Ediffs.append(E_top_Gamma - E_top_K)

        def Hfn(k):
            return Hk(k, Hr, latVecs)

        mstar_top_Gamma = effective_mass_band(Hfn, Gamma_cart, top_Gamma[0], alat_Bohr)
        mstar_top_K = effective_mass_band(Hfn, K_cart, top_K[0], alat_Bohr)

        mstar_Gammas.append(mstar_top_Gamma[0])
        mstar_Ks.append(mstar_top_K[0])

        nh_Gamma, nh_K = hole_distribution(0.0, R, Hr, latVecs, E_F,
                sigmas_initial, Pzs, hole_density_bohr2, d_bohr, epsilon_r,
                tol_abs, tol_rel)

        nh_Gammas_frac.append(nh_Gamma / hole_density_bohr2)
        nh_Ks_frac.append(nh_K / hole_density_bohr2)

    plt.plot(E_V_nms, Ediffs, 'k.')
    plt.savefig("Ediffs.png", bbox_inches='tight', dpi=500)
    plt.clf()

    plt.plot(E_V_nms, mstar_Gammas, 'k.')
    plt.savefig("mstar_Gammas.png", bbox_inches='tight', dpi=500)
    plt.clf()

    plt.plot(E_V_nms, mstar_Ks, 'k.')
    plt.savefig("mstar_Ks.png", bbox_inches='tight', dpi=500)
    plt.clf()

    plt.xlabel("$E$ [V/nm]")
    plt.ylabel("Occupation fraction")
    plt.xlim(E_V_nms[0], E_V_nms[-1])
    plt.ylim(0.0, 1.0)

    hole_density_note = "$p = $" + decimal_format(hole_density_cm2, 1) + " cm$^{-2}$"

    plt.plot(E_V_nms, nh_Gammas_frac, 'r-', label="$\\Gamma$")
    plt.plot(E_V_nms, nh_Ks_frac, 'b-', label="$K$")
    plt.legend(loc=0, title=hole_density_note)
    plt.savefig("occupations_dft_Efield.png", bbox_inches='tight', dpi=500)
    plt.clf()

if __name__ == "__main__":
    _main()
