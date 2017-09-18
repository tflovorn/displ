from __future__ import division
import argparse
import os
from multiprocessing import Pool
import numpy as np
from displ.build.build import _get_work
from displ.pwscf.parseScf import fermi_from_scf, latVecs_from_scf, alat_from_scf
from displ.kdotp.effective_valence_K import get_layer_basis_from_dm_K, make_effective_Hamiltonian_K
from displ.kdotp.effective_valence_Gamma import make_effective_Hamiltonian_Gamma
from displ.kdotp.efield import (_bohr_per_Angstrom, _e_C, get_phis, get_sigma_self_consistent,
        H_k0_phis, plot_results)

def get_H_k0(H0, p, mstar_inv):
    def H_k0(q):
        q_part = np.zeros([6, 6], dtype=np.complex128)
        for cp in range(2):
            q_part += q[cp] * p[cp]
            for c in range(2):
                q_part += (1/2) * q[cp] * q[c] * mstar_inv[(cp, c)]

        return H0 + q_part

    return H_k0

def layer_projections():
    Pzs = []
    for l in range(3):
        Pz = np.zeros([6, 6], dtype=np.complex128)
        for i in range(2*l, 2*l + 2):
            Pz[i, i] = 1

        Pzs.append(Pz)

    return Pzs

def hole_distribution(E_V_nm, H0s, ps, mstar_invs, E_F_base, sigmas_initial, Pzs,
        hole_density_bohr2, d_bohr, epsilon_r, tol_abs, tol_rel, screened):
    H_k0s = [get_H_k0(H0, p, mstar_inv) for H0, p, mstar_inv in zip(H0s, ps, mstar_invs)]
    band_indices = [list(range(6)) for i in range(len(H0s))]

    E_V_bohr = E_V_nm / (10 * _bohr_per_Angstrom)

    phis_initial = get_phis(sigmas_initial, d_bohr, E_V_bohr, epsilon_r, screened)

    nh_converged, sigmas_converged = get_sigma_self_consistent(H_k0s, sigmas_initial,
            Pzs, band_indices, hole_density_bohr2, d_bohr, E_V_bohr, epsilon_r,
            tol_abs, tol_rel, curvatures=None, screened=screened, curv_warn=[4, 5])

    phis_converged = get_phis(sigmas_converged, d_bohr, E_V_bohr, epsilon_r, screened)

    E_converged = []
    for H_k0_phi, band_indices_k0 in zip(H_k0_phis(H_k0s, phis_converged, Pzs), band_indices):
        H = H_k0_phi([0.0, 0.0, 0.0])
        Es, U = np.linalg.eigh(H)

        E_converged.append([Es[m] for m in band_indices_k0])

    E_Gamma = max(E_converged[0])
    E_K = max(E_converged[1])

    sum_nh = lambda nh: sum([sum(bands) for bands in nh])

    nh_Gamma = sum_nh(nh_converged[0])
    nh_K = sum_nh(nh_converged[1]) + sum_nh(nh_converged[2])

    return nh_Gamma, nh_K, E_Gamma, E_K

def _main():
    np.set_printoptions(threshold=np.inf)

    parser = argparse.ArgumentParser("TMD multilayer response to electric field",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("prefix", type=str,
            help="Prefix for calculation")
    parser.add_argument("--subdir", type=str, default=None,
            help="Subdirectory under work_base where calculation was run")
    parser.add_argument("--holes", type=float, default=8e12,
            help="Hole concentration [cm^{-2}]")
    parser.add_argument("--screened", action='store_true',
            help="Include screening by holes")
    parser.add_argument("--diag_mass_only", action='store_true',
            help="Include q variation only through qx^2, qy^2 mass terms on diagonal")
    args = parser.parse_args()

    work = _get_work(args.subdir, args.prefix)
    wannier_dir = os.path.join(work, "wannier")
    scf_path = os.path.join(wannier_dir, "scf.out")

    E_F_base = fermi_from_scf(scf_path)

    d_A = 6.488 # Angstrom
    d_bohr = _bohr_per_Angstrom * d_A

    hole_density_cm2 = args.holes
    hole_density_bohr2 = hole_density_cm2 / (10**8 * _bohr_per_Angstrom)**2

    #print("hole_density_bohr2")
    #print(hole_density_bohr2)

    # Choose initial potential assuming holes are distributed uniformally.
    sigma_layer_initial = (1/3) * _e_C * hole_density_bohr2

    sigmas_initial = sigma_layer_initial * np.array([1.0, 1.0, 1.0])
    #print("sigmas_initial [C/bohr^2]")
    #print(sigmas_initial)

    # Dielectric constant of WSe2:
    # Kim et al., ACS Nano 9, 4527 (2015).
    # http://pubs.acs.org/doi/abs/10.1021/acsnano.5b01114
    #epsilon_r = 7.2

    # Effective dielectric constant from DFT (LDA).
    # avg(K^high_{top - bottom}, Gamma_{top - bottom})
    epsilon_r = 7.87

    H0_tot_Gamma, ps_tot_Gamma, mstar_inv_tot_Gamma = make_effective_Hamiltonian_Gamma(args.subdir,
            args.prefix, top_two_only=False, verbose=False)

    K_lat = np.array([1/3, 1/3, 0.0])
    H0_tot_K, ps_tot_K, mstar_inv_tot_K = make_effective_Hamiltonian_K(K_lat, args.subdir,
            args.prefix, get_layer_basis_from_dm_K, verbose=False)

    Kp_lat = np.array([-1/3, -1/3, 0.0])
    H0_tot_Kp, ps_tot_Kp, mstar_inv_tot_Kp = make_effective_Hamiltonian_K(Kp_lat, args.subdir,
            args.prefix, get_layer_basis_from_dm_K, verbose=False)

    H0s = [H0_tot_Gamma, H0_tot_K, H0_tot_Kp]
    if args.diag_mass_only:
        ps = [np.zeros([6, 6], dtype=np.complex128),
                np.zeros([6, 6], dtype=np.complex128),
                np.zeros([6, 6], dtype=np.complex128)]

        def extract_diag(mstar_inv):
            result = {}
            for cp in range(2):
                for c in range(2):
                    result[(cp, c)] = np.zeros([6, 6], dtype=np.complex128)
                    if c == cp:
                        for i in range(6):
                            result[(cp, c)][i, i] = mstar_inv[(cp, c)][i, i]

            return result

        mstar_invs = list(map(extract_diag, [mstar_inv_tot_Gamma, mstar_inv_tot_K, mstar_inv_tot_Kp]))
    else:
        ps = [ps_tot_Gamma, ps_tot_K, ps_tot_Kp]
        mstar_invs = [mstar_inv_tot_Gamma, mstar_inv_tot_K, mstar_inv_tot_Kp]

    Pzs = layer_projections()

    E_V_nms = np.linspace(0.0, 1.2, 84)

    if hole_density_cm2 > 0.0:
        tol_abs = 1e-6 * sum(sigmas_initial)
    else:
        tol_abs = 1e-12 * _e_C

    tol_rel = 1e-6

    distr_args = []
    for E_V_nm in E_V_nms:
        distr_args.append([E_V_nm, H0s, ps, mstar_invs, E_F_base, sigmas_initial, Pzs,
               hole_density_bohr2, d_bohr, epsilon_r, tol_abs, tol_rel, args.screened])

    with Pool() as p:
        nh_Es = p.starmap(hole_distribution, distr_args)

    plot_results(nh_Es, hole_density_bohr2, hole_density_cm2, epsilon_r, args.screened, E_V_nms)

if __name__ == "__main__":
    _main()
