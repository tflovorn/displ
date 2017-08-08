from __future__ import division
import argparse
import os
import itertools
from functools import partial
from multiprocessing import Pool
import numpy as np
import numdifftools as nd
from scipy.optimize import bisect
import matplotlib.pyplot as plt
from displ.build.build import _get_work, band_path_labels
from displ.pwscf.parseScf import fermi_from_scf, latVecs_from_scf, alat_from_scf
from displ.wannier.extractHr import extractHr
from displ.wannier.bands import Hk
from displ.kdotp.model_weights_K import vec_linspace, top_valence_indices
from displ.kdotp.separability_K import get_layer_projections

# Constants (physical and unit relations)
_bohr_per_Angstrom = 1.889726164
_epsilon_0_F_m = 8.8541878e-12 # F/m = C/Vm
_epsilon_0_F_bohr = _epsilon_0_F_m / (10**10 * _bohr_per_Angstrom)
_e_C = 1.60217662e-19 # C

# Avg. electric field above and below the TMD multilayer -- E_V_bohr [V/a_Bohr]
# Distance between W in adjacent TMD laeyers -- d_bohr [a_Bohr]
# Total number of holes per unit area -- hole_density_bohr2 [a_Bohr^{-2}]
# List of Hamiltonian functions for each relevant band maximum -- H_k0s[k0_index](q) [eV], with q given in [a_Bohr^{-1}]
# List of initial guesses for electric potential on each layer -- phis [V]
# List of projection operators onto each layer -- Pzs[layer_index] [unitless]
# List of band indices of bands near top of valence band to include -- band_indices[k0_index][reduced_band_index]

def H_k0_phis(H_k0s, phis, Pzs):
    '''[eV]
    '''
    result = []

    def Hq(H, q):
        H_base = H(q)
        # e * phi has eV units.
        phi_parts = [-phi * Pz for phi, Pz in zip(phis, Pzs)]
        return H_base + sum(phi_parts)

    for H in H_k0s:
        result.append(partial(Hq, H))

    return result

def energies_at_k0(H_k0s, phis, Pzs, band_indices):
    '''E^0_{k0;n0} = result[k0_index][n0]

    [eV]
    '''
    result = []
    for H_k0_phi, band_indices_k0 in zip(H_k0_phis(H_k0s, phis, Pzs), band_indices):
        q0 = np.array([0.0, 0.0, 0.0])
        Es, U = np.linalg.eigh(H_k0_phi(q0))

        E0s = [Es[n0] for n0 in band_indices_k0]
        result.append(E0s)

    return result

def band_curvatures(H_k0s, phis, Pzs, band_indices):
    '''Returns -(d^2 E_{k0;n} / dq_c^2) \equiv [hbar^2 / (2 * mstar_c{k0;n})]^{-1}
    for c = x, y.

    result[k0_index][n0] = [x, y]

    [eV Bohr^2]
    '''
    def band_energy(H_k0_phi, n, c, q):
        q_subst = []
        for cp in range(2):
            if cp == c:
                q_subst.append(q)
            else:
                q_subst.append(0.0)

        q_subst.append(0.0)

        Es, U = np.linalg.eigh(H_k0_phi(np.array(q_subst)))
        return Es[n]

    result = []
    for H_k0_phi, band_indices_k0 in zip(H_k0_phis(H_k0s, phis, Pzs), band_indices):
        result.append([])
        for n in band_indices_k0:
            curvatures = []
            for c in range(2):
                curvatures.append(nd.Derivative(partial(band_energy, H_k0_phi, n, c), n=2, order=16)(0.0))

            cx, cy = curvatures[0], curvatures[1]
            threshold = 0.05
            if abs(cx - cy) > threshold * max(abs(cx), abs(cy)):
                print("WARNING: cx = {}, cy = {}, relative diff = {}".format(cx, cy, abs(cx - cy) / max(abs(cx), abs(cy))))

            result[-1].append([cx, cy])

    return result

def step(x):
    if x > 0.0:
        return 1.0
    else:
        return 0.0

def hole_density_at_E(E0s, curvatures, E):
    '''n_h^A [1/Bohr^2] around k0 in band n = result[k0_index][n0]
    '''
    result = []
    for E0_k0, curvature_k0 in zip(E0s, curvatures):
        result.append([])
        for E0, curvature in zip(E0_k0, curvature_k0):
            cx, cy = curvature
            mass_term = 1/(-cx/2)
            val = (1/(4*np.pi)) * mass_term * (E0 - E) * step(E0 - E)
            result[-1].append(val)

    return result

def get_Fermi_energy(H_k0s, phis, Pzs, band_indices, hole_density, curvatures=None):
    E0s = energies_at_k0(H_k0s, phis, Pzs, band_indices)

    # TODO - E_min choice here is not correct in general.
    # Assumes there is a band somewhere in the supplied bands which is
    # fully unoccupied. The right energy scale to use to choose E_min
    # is not obvious for the k dot p model.
    # If this assumption is wrong, shouldn't fail silently - will have
    # error_fn(E_min) with same sign as error_fn(E_max), causing
    # exception in bisect.
    E_min = min([min(E0s_k0) for E0s_k0 in E0s])
    E_max = max([max(E0s_k0) for E0s_k0 in E0s])

    if curvatures is None:
        curvatures = band_curvatures(H_k0s, phis, Pzs, band_indices)

    def error_fn(E):
        band_hole_density = hole_density_at_E(E0s, curvatures, E)
        return hole_density - sum([sum(hk) for hk in band_hole_density])

    return bisect(error_fn, E_min, E_max), E0s, curvatures

def get_layer_weights(H_k0s, phis, Pzs, band_indices):
    '''Returns list of weight[k0_index][n0][layer_index] [unitless]
    '''
    result = []
    for H_k0_phi, band_indices_k0 in zip(H_k0_phis(H_k0s, phis, Pzs), band_indices):
        result.append([])

        q0 = np.array([0.0, 0.0, 0.0])
        Es, U = np.linalg.eigh(H_k0_phi(q0))

        for n in band_indices_k0:
            result[-1].append([])

            state = U[:, [n]]
            for Pz in Pzs:
                weight = (state.conjugate().T @ Pz @ state)[0, 0].real
                result[-1][-1].append(weight)

    return result

def check_layer_weight(k0s, H_k0s, phis, Pzs, band_indices, E_V_nm):
    '''Check that layer weight doesn't change too quickly in region around k0.
    '''
    weights_q0, weights_near_q0 = [], []
    for k0_index, (k0, H_k0_phi, band_indices_k0) in enumerate(zip(k0s,
            H_k0_phis(H_k0s, phis, Pzs), band_indices)):
        # Check 5% of distance from k0 to all other k0s.
        # TODO - if there is only one k0, this doesn't do anything.
        # Handle that case properly (there, need additional information
        # to choose good distance from k0 since we aren't in reciprocal
        # lattice units here).
        for k0p_index in range(len(k0s)):
            if k0p_index == k0_index:
                continue

            k0p = k0s[k0p_index]
            frac = 0.05
            # H_k0 expects argument near 0. near_q0 = k0 + frac*(k0p - k0) - k0
            near_q0 = frac * (k0p - k0)
            
            q0 = np.array([0.0, 0.0, 0.0])
            Es_q0, U_q0 = np.linalg.eigh(H_k0_phi(q0))

            Es_near_q0, U_near_q0 = np.linalg.eigh(H_k0_phi(near_q0))

            for n in band_indices_k0:
                state_q0 = U_q0[:, [n]]
                state_near_q0 = U_near_q0[:, [n]]

                for z, Pz in enumerate(Pzs):
                    weight_q0 = (state_q0.conjugate().T @ Pz @ state_q0)[0, 0].real
                    weight_near_q0 = (state_near_q0.conjugate().T @ Pz @ state_near_q0)[0, 0].real
                    weight_diff = abs(weight_q0 - weight_near_q0)

                    tolerance = 0.05
                    if weight_diff > tolerance and n == max(band_indices_k0):
                        print("WARNING: weight_diff = {} for E_V_nm = {}, k0, k0p, n, z = {}, {}, {}, {}; weight_q0, weight_near_q0 = {}, {}".format(weight_diff, E_V_nm, k0_index, k0p_index, n, z, weight_q0, weight_near_q0))

def layer_hole_density_at_E(H_k0s, phis, Pzs, band_indices, E, E0s=None, curvatures=None):
    '''n_h^A(l) [1/Bohr^2]
    '''
    if E0s is None:
        E0s = energies_at_k0(H_k0s, phis, Pzs, band_indices)
    if curvatures is None:
        curvatures = band_curvatures(H_k0s, phis, Pzs, band_indices)

    layer_weights = get_layer_weights(H_k0s, phis, Pzs, band_indices)

    result = []
    for E0_k0, curvature_k0, weights_k0 in zip(E0s, curvatures, layer_weights):
        result.append([])
        for E0, curvature, weights_k0_n in zip(E0_k0, curvature_k0, weights_k0):
            result[-1].append([])
            for z, weight in enumerate(weights_k0_n):
                cx, cy = curvature
                # TODO factor out part after weight, same as overall hole density
                mass_term = 1/(-cx/2)
                val = weight * (1/(4*np.pi)) * mass_term * (E0 - E) * step(E0 - E)
                result[-1][-1].append(val)

    result_layer_total = [0.0]*len(Pzs)
    for k0_index, E0_k0 in enumerate(E0s):
        for band_index in range(len(E0_k0)):
            for z in range(len(Pzs)):
                result_layer_total[z] += result[k0_index][band_index][z]

    return result, result_layer_total

def plot_H_k0_phis(H_k0s, phis, Pzs, band_indices, E_F_base):
    qs = vec_linspace(np.array([0.0, 0.0, 0.0]), [0.3, 0.0, 0.0], 100)
    for H_k0_phi, band_indices_k0 in zip(H_k0_phis(H_k0s, phis, Pzs), band_indices):
        Ekms = []
        for q in qs:
            Es, U = np.linalg.eigh(H_k0_phi(q))
            Ekms.append(Es)

        Emks = []
        for band_index in range(len(Ekms[0])):
            Emks.append([])

        for k_index in range(len(qs)):
            for band_index in range(len(Ekms[0])):
                Emks[band_index].append(Ekms[k_index][band_index])

        for band_index, Em in enumerate(Emks):
            if band_index in band_indices_k0:
                plt.plot(range(len(qs)), Em, 'r-')
            else:
                plt.plot(range(len(qs)), Em, 'k-')

        plt.axhline(E_F_base, linestyle='dashed')

        plt.show()
        plt.clf()

def get_phis(sigmas, d_bohr, E_V_bohr, epsilon_r):
    d_eps = d_bohr / (2 * epsilon_r * _epsilon_0_F_bohr)
    phis = [d_bohr * E_V_bohr - d_eps * (sigmas[1] + 2 * sigmas[2]),
            -d_eps * (sigmas[0] + sigmas[2]),
            -d_bohr * E_V_bohr - d_eps * (2 * sigmas[0] + sigmas[1])]
    return phis

def sigma_converged(sigmas, new_sigmas, tol_abs, tol_rel):
    if new_sigmas is None:
        return False

    for s, sp in zip(sigmas, new_sigmas):
        abs_ok = abs(s - sp) < tol_abs

        rel_ok = abs(s - sp) < tol_rel * max(abs(s), abs(sp))

        if not (abs_ok or rel_ok):
            return False

    return True

def get_sigma_self_consistent(H_k0s, sigmas_initial, Pzs, band_indices, hole_density_bohr2, d_bohr, E_V_bohr, epsilon_r, tol_abs, tol_rel, curvatures=None):
    sigmas = sigmas_initial
    new_sigmas = None
    beta = 0.5
    iter_num = 0
    mixing_start = 4

    while not sigma_converged(sigmas, new_sigmas, tol_abs, tol_rel):
        if new_sigmas is not None:
            sigmas = new_sigmas

        phis = get_phis(sigmas, d_bohr, E_V_bohr, epsilon_r)

        E_F, E0s, curvatures = get_Fermi_energy(H_k0s, phis, Pzs, band_indices, hole_density_bohr2, curvatures)

        new_nh, new_nh_layer_total = layer_hole_density_at_E(H_k0s, phis, Pzs, band_indices, E_F, E0s, curvatures)

        suggested_sigmas = [_e_C * n for n in new_nh_layer_total]

        if iter_num >= mixing_start:
            new_sigmas = [sigma + beta * (sigma_prime - sigma) for sigma, sigma_prime in zip(sigmas, suggested_sigmas)]
        else:
            new_sigmas = suggested_sigmas

        iter_num += 1

    return new_nh, new_sigmas

def get_H_k0s(R, Hr, latVecs, E_F_base):
    Gamma_cart = np.array([0.0, 0.0, 0.0])
    K_lat = np.array([1/3, 1/3, 0.0])
    K_cart = np.dot(K_lat, R)
    Kprime_cart = -K_cart

    k0s = [Gamma_cart, K_cart, Kprime_cart]

    H_k0s, band_indices = [], []
    def Hfn(k0, q):
        return Hk(q + k0, Hr, latVecs)

    k0_num_orbitals = [2, 3, 3]
    for k0, num_orbitals in zip(k0s, k0_num_orbitals):
        H_k0s.append(partial(Hfn, k0))

        Es, U = np.linalg.eigh(H_k0s[-1](np.array([0.0, 0.0, 0.0])))
        top = top_valence_indices(E_F_base, num_orbitals, Es)
        band_indices.append(top)

    return k0s, H_k0s, band_indices

def hole_distribution(E_V_nm, R, Hr, latVecs, E_F_base, sigmas_initial, Pzs, hole_density_bohr2, d_bohr, epsilon_r, tol_abs, tol_rel, initial_mass=False):
    # Use full TB model -- TODO k dot p
    # H_k0s is generated here since Hfn can't be pickled for use in multiprocessing
    k0s, H_k0s, band_indices = get_H_k0s(R, Hr, latVecs, E_F_base)

    E_V_bohr = E_V_nm / (10 * _bohr_per_Angstrom)

    #print("unscreened phi_3 - phi_1 [V]")
    #print(-2 * d_bohr * E_V_bohr)

    if initial_mass:
        phi_no_E_no_p = [0.0, 0.0, 0.0]
        curvatures = band_curvatures(H_k0s, phi_no_E_no_p, Pzs, band_indices)
    else:
        curvatures = None

    phis_initial = get_phis(sigmas_initial, d_bohr, E_V_bohr, epsilon_r)
    #print("phis_initial [V]")
    #print(phis_initial)

    nh_converged, sigmas_converged = get_sigma_self_consistent(H_k0s, sigmas_initial, Pzs, band_indices, hole_density_bohr2, d_bohr, E_V_bohr, epsilon_r, tol_abs, tol_rel, curvatures)

    #print("sigmas_converged [C/bohr^2]")
    #print(sigmas_converged)

    phis_converged = get_phis(sigmas_converged, d_bohr, E_V_bohr, epsilon_r)
    #print("phis_converged [V]")
    #print(phis_converged)

    check_layer_weight(k0s, H_k0s, phis_converged, Pzs, band_indices, E_V_nm)

    #print("hole distribution converged")
    #print(nh_converged)

    sum_nh = lambda nh: sum([sum(bands) for bands in nh])

    nh_Gamma = sum_nh(nh_converged[0])
    nh_K = sum_nh(nh_converged[1]) + sum_nh(nh_converged[2])

    #print("fractions nh_Gamma, nh_K")
    #print(nh_Gamma / hole_density_bohr2, nh_K / hole_density_bohr2)

    return nh_Gamma, nh_K

def decimal_format(x, num_decimal):
    num_digits_exp = int(np.floor(np.log10(abs(x))))
    x_reduced = abs(x) / 10**num_digits_exp

    x_front = str(x_reduced)[:2+num_decimal]

    sgn = ""
    if x < 0.0:
        sgn = "-"

    return sgn + x_front + " x 10$^{" + str(num_digits_exp) + "}$"

def _main():
    np.set_printoptions(threshold=np.inf)

    parser = argparse.ArgumentParser("TMD multilayer response to electric field",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("prefix", type=str,
            help="Prefix for calculation")
    parser.add_argument("--subdir", type=str, default=None,
            help="Subdirectory under work_base where calculation was run")
    parser.add_argument("--num_layers", type=int, default=3,
            help="Number of layers")
    parser.add_argument("--initial_mass", action='store_true',
            help="Use effective mass at E_perp = 0, p = 0 instead of recalculating for each phi")
    parser.add_argument("--plot_initial", action='store_true',
            help="Plot initial band structure with max applied field, before charge convergence")
    args = parser.parse_args()

    if args.num_layers != 3:
        assert("num_layers != 3 not implemented")

    work = _get_work(args.subdir, args.prefix)
    wannier_dir = os.path.join(work, "wannier")
    scf_path = os.path.join(wannier_dir, "scf.out")

    E_F_base = fermi_from_scf(scf_path)
    latVecs = latVecs_from_scf(scf_path)
    R = 2 * np.pi * np.linalg.inv(latVecs.T)

    Hr_path = os.path.join(wannier_dir, "{}_hr.dat".format(args.prefix))
    Hr = extractHr(Hr_path)

    d_A = 6.488 # Angstrom
    d_bohr = _bohr_per_Angstrom * d_A

    hole_density_cm2 = 8e12
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
    epsilon_r = 7.2

    Pzs = get_layer_projections(args.num_layers)

    E_V_nms = np.linspace(0.0, 0.6, 42)

    if args.plot_initial:
        E_V_bohr = E_V_nms[-1] / (10 * _bohr_per_Angstrom)
        phis_initial_max = get_phis(sigmas_initial, d_bohr, E_V_bohr, epsilon_r)
        plot_H_k0_phis(H_k0s, phis_initial, Pzs, band_indices, E_F_base)
        return

    tol_abs = 1e-6 * sum(sigmas_initial)
    tol_rel = 1e-6

    distr_args = []
    for E_V_nm in E_V_nms:
        distr_args.append([E_V_nm, R, Hr, latVecs, E_F_base, sigmas_initial,
                Pzs, hole_density_bohr2, d_bohr, epsilon_r, tol_abs, tol_rel, args.initial_mass])

    with Pool() as p:
        nhs = p.starmap(hole_distribution, distr_args)

    nh_Gammas_frac, nh_Ks_frac = [], []
    for nh_Gamma, nh_K in nhs:
        nh_Gammas_frac.append(nh_Gamma / hole_density_bohr2)
        nh_Ks_frac.append(nh_K / hole_density_bohr2)

    plt.xlabel("$E$ [V/nm]")
    plt.ylabel("Occupation fraction")
    plt.xlim(E_V_nms[0], E_V_nms[-1])
    plt.ylim(0.0, 1.0)

    hole_density_note = "$p = $" + decimal_format(hole_density_cm2, 1) + " cm$^{-2}$"

    plt.plot(E_V_nms, nh_Gammas_frac, 'r-', label="$\\Gamma$")
    plt.plot(E_V_nms, nh_Ks_frac, 'b-', label="$K$")
    plt.legend(loc=0, title=hole_density_note)
    plt.savefig("occupations_Efield.png", bbox_inches='tight', dpi=500)

if __name__ == "__main__":
    _main()
