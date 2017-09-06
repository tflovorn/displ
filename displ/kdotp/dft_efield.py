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
    parser.add_argument("--holes", type=float, default=8e12,
            help="Hole concentration [cm^{-2}]")
    parser.add_argument("--screened", action='store_true',
            help="Include screening by holes")
    args = parser.parse_args()

    E_V_nms, prefixes = get_prefixes(args.global_prefix, args.subdir)
    print(E_V_nms)
    print(prefixes)

    d_A = 6.488 # Angstrom
    d_bohr = _bohr_per_Angstrom * d_A

    hole_density_cm2 = args.holes
    hole_density_bohr2 = hole_density_cm2 / (10**8 * _bohr_per_Angstrom)**2

    sigma_layer_initial = (1/3) * _e_C * hole_density_bohr2

    sigmas_initial = sigma_layer_initial * np.array([1.0, 1.0, 1.0])

    # Dielectric constant of WSe2:
    # Kim et al., ACS Nano 9, 4527 (2015).
    # http://pubs.acs.org/doi/abs/10.1021/acsnano.5b01114
    epsilon_r = 7.2

    Pzs = get_layer_projections(3)  # TODO num_layers arg

    if hole_density_cm2 > 0.0:
        tol_abs = 1e-6 * sum(sigmas_initial)
    else:
        tol_abs = 1e-12 * _e_C

    tol_rel = 1e-6

    Ediffs, mstar_Gammas, mstar_Ks = [], [], []
    nh_Gammas_frac, nh_Ks_frac = [], []
    E_Gammas, E_Ks = [], []

    for prefix in prefixes:
        print(prefix)
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
        assert(top_Gamma == [41, 40])
        assert(top_K == [41, 40, 39])

        E_top_Gamma, E_top_K = Es_Gamma[top_Gamma[0]], Es_K[top_K[0]]

        Ediffs.append(E_top_Gamma - E_top_K)

        def Hfn(k):
            return Hk(k, Hr, latVecs)

        mstar_top_Gamma = effective_mass_band(Hfn, Gamma_cart, top_Gamma[0], alat_Bohr)
        mstar_top_K = effective_mass_band(Hfn, K_cart, top_K[0], alat_Bohr)

        mstar_Gammas.append(mstar_top_Gamma[0])
        mstar_Ks.append(mstar_top_K[0])

        nh_Gamma, nh_K, E_Gamma, E_K = hole_distribution(0.0, R, Hr, latVecs, E_F,
                sigmas_initial, Pzs, hole_density_bohr2, d_bohr, epsilon_r,
                tol_abs, tol_rel, screened=args.screened)

        if hole_density_bohr2 > 0.0:
            nh_Gammas_frac.append(nh_Gamma / hole_density_bohr2)
            nh_Ks_frac.append(nh_K / hole_density_bohr2)

        E_Gammas.append(E_Gamma)
        E_Ks.append(E_K)

    E_Gamma_Ks, E_Gamma_steps, E_K_steps, E_Gamma_d2s, E_K_d2s = [], [], [], [], []
    for E_index, (E_Gamma, E_K) in enumerate(zip(E_Gammas, E_Ks)):
        E_Gamma_Ks.append(E_Gamma - E_K)

        if E_index > 0:
            last_E_Gamma = E_Gammas[E_index - 1]
            last_E_K = E_Ks[E_index - 1]
            delta_Eperp = E_V_nms[1] - E_V_nms[0]

            E_Gamma_steps.append((E_Gamma - last_E_Gamma) / delta_Eperp)
            E_K_steps.append((E_K - last_E_K) / delta_Eperp)

            if E_index > 1:
                bl_E_Gamma = E_Gammas[E_index - 2]
                bl_E_K = E_Ks[E_index - 2]

                E_Gamma_d2s.append((E_Gamma - 2*last_E_Gamma + bl_E_Gamma) / delta_Eperp**2)
                E_K_d2s.append((E_K - 2*last_E_K + bl_E_K) / delta_Eperp**2)

    # Gamma effective mass
    plt.xlabel("$E$ [V/nm]")
    plt.ylabel("$m^*_{\\Gamma} / m_e$")
    plt.xlim(E_V_nms[0], E_V_nms[-1])

    plt.plot(E_V_nms, mstar_Gammas, 'k.')
    plt.savefig("mstar_Gammas.png", bbox_inches='tight', dpi=500)
    plt.clf()

    # K effective mass
    plt.xlabel("$E$ [V/nm]")
    plt.ylabel("$m^*_K / m_e$")
    plt.xlim(E_V_nms[0], E_V_nms[-1])

    plt.plot(E_V_nms, mstar_Ks, 'k.')
    plt.savefig("mstar_Ks.png", bbox_inches='tight', dpi=500)
    plt.clf()

    if hole_density_bohr2 > 0.0:
        # Band occupations
        plt.xlabel("$E$ [V/nm]")
        plt.ylabel("Occupation fraction")
        plt.xlim(E_V_nms[0], E_V_nms[-1])
        plt.ylim(0.0, 1.0)

        hole_density_note = "$p = $" + decimal_format(hole_density_cm2, 1) + " cm$^{-2}$"

        plt.plot(E_V_nms, nh_Gammas_frac, 'r.', label="$\\Gamma$")
        plt.plot(E_V_nms, nh_Ks_frac, 'b.', label="$K$")
        plt.legend(loc=0, title=hole_density_note)
        plt.savefig("occupations_dft_Efield.png", bbox_inches='tight', dpi=500)
        plt.clf()
    else:
        hole_density_note = ""

    # E_Gamma - E_K
    plt.xlabel("$E$ [V/nm]")
    plt.ylabel("$E_{\\Gamma} - E_K$ [eV]")
    plt.xlim(E_V_nms[0], E_V_nms[-1])

    plt.title(hole_density_note)

    plt.plot(E_V_nms, E_Gamma_Ks, 'r-')
    plt.savefig("Ediffs_dft_Efield.png", bbox_inches='tight', dpi=500)
    plt.clf()

    # Energy shifts
    # NOTE - these may not be meaningful!
    # Only correct if band energies are not shifted overall by the applied
    # electric field. Not sure if this is the case.

    # d E_K / dE ; dE_Gamma / dE
    plt.xlabel("$E$ [V/nm]")
    plt.ylabel("Rate of energy change [eV/(V/nm)]")
    plt.xlim(E_V_nms[0], E_V_nms[-1])

    plt.plot(E_V_nms[1:], E_Gamma_steps, 'r-', label="$dE_{\\Gamma}/dE$")
    plt.plot(E_V_nms[1:], E_K_steps, 'b-', label="$dE_{K}/dE$")
    plt.legend(loc=0, title=hole_density_note)
    plt.savefig("energy_steps_dft_Efield.png", bbox_inches='tight', dpi=500)
    plt.clf()

    # d^2 E_K / dE^2 ; d^2 E_Gamma / dE^2
    plt.xlabel("$E$ [V/nm]")
    plt.ylabel("Second derivative of energy change [eV/(V/nm)$^2$]")
    plt.xlim(E_V_nms[0], E_V_nms[-1])

    plt.plot(E_V_nms[2:], E_Gamma_d2s, 'r-', label="$d^2 E_{\\Gamma}/dE^2$")
    plt.plot(E_V_nms[2:], E_K_d2s, 'b-', label="$d^2 E_{K}/dE^2$")
    plt.legend(loc=0, title=hole_density_note)
    plt.savefig("energy_d2s_dft_Efield.png", bbox_inches='tight', dpi=500)
    plt.clf()

if __name__ == "__main__":
    _main()
