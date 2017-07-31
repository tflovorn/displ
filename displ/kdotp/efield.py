from __future__ import division
import argparse
import os
import itertools
from functools import partial
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
__bohr_per_Angstrom = 1.889726164
__epsilon_0_F_m = 8.8541878e-12 # F/m = C/Vm
__epsilon_0_F_bohr = __epsilon_0_F_m / (10**10 * __bohr_per_Angstrom)
__e_C = 1.60217662e-19 # C

# Electric field below the lowest TMD layer -- E_below [V/a_Bohr]
# Distance between W in adjacent TMD laeyers -- d [a_Bohr]
# Total number of holes per unit area -- hole_density [a_Bohr^{-2}]
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
    for H_k0, band_indices_k0 in zip(H_k0_phis(H_k0s, phis, Pzs), band_indices):
        q0 = np.array([0.0, 0.0, 0.0])
        Es, U = np.linalg.eigh(H_k0(q0))

        E0s = [Es[n0] for n0 in band_indices_k0]
        result.append(E0s)

    return result

def band_curvatures(H_k0s, phis, Pzs, band_indices):
    '''Returns -(d^2 E_{k0;n} / dq_c^2) \equiv [hbar^2 / (2 * mstar_c{k0;n})]^{-1}
    for c = x, y.

    result[k0_index][n0] = [x, y]

    [eV Bohr^2]
    '''
    def band_energy(H_k0, n, c, q):
        q_subst = []
        for cp in range(2):
            if cp == c:
                q_subst.append(q)
            else:
                q_subst.append(0.0)

        q_subst.append(0.0)

        Es, U = np.linalg.eigh(H_k0(np.array(q_subst)))
        return Es[n]

    result = []
    for H_k0, band_indices_k0 in zip(H_k0_phis(H_k0s, phis, Pzs), band_indices):
        result.append([])
        for n in band_indices_k0:
            curvatures = []
            for c in range(2):
                curvatures.append(nd.Derivative(partial(band_energy, H_k0, n, c), n=2, order=16)(0.0))

            print(n, curvatures)
            result[-1].append([curvatures[0], curvatures[1]])

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
            val = np.sqrt(cx/cy) * (1/(4*np.pi)) * (-1/cx) * (E0 - E) * step(E0 - E)
            result[-1].append(val)

    return result

def get_Fermi_energy(H_k0s, phis, Pzs, band_indices, hole_density):
    E0s = energies_at_k0(H_k0s, phis, Pzs, band_indices)

    E_min = min([min(E0s_k0) for E0s_k0 in E0s])
    E_max = max([max(E0s_k0) for E0s_k0 in E0s])

    print("in get_Fermi_energy got E0s = ")
    print(E0s)
    print(E_min, E_max)

    curvatures = band_curvatures(H_k0s, phis, Pzs, band_indices)
    print("curvatures = ")
    print(curvatures)

    def error_fn(E):
        band_hole_density = hole_density_at_E(E0s, curvatures, E)
        #print("E, band_hole_density")
        #print(E)
        #print(band_hole_density)
        return hole_density - sum([sum(hk) for hk in band_hole_density])

    print("n_h error at E_min, E_max")
    print(error_fn(E_min), error_fn(E_max))

    return bisect(error_fn, E_min, E_max), E0s, curvatures

def get_layer_weights(H_k0s, phis, Pzs, band_indices):
    '''Returns list of weight[k0_index][n0][layer_index] [unitless]
    '''
    result = []
    for H_k0, band_indices_k0 in zip(H_k0_phis(H_k0s, phis, Pzs), band_indices):
        result.append([])

        q0 = np.array([0.0, 0.0, 0.0])
        Es, U = np.linalg.eigh(H_k0(q0))

        for n in band_indices_k0:
            result[-1].append([])

            state = U[:, [n]]
            for Pz in Pzs:
                weight = (state.conjugate().T @ Pz @ state)[0, 0]
                result[-1][-1].append(weight)

    return result

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
                val = weight * np.sqrt(cx / cy) * (1/(4*np.pi)) * (-1/cx) * (E0 - E) * step(E0 - E)
                result[-1][-1].append(val)

    result_layer_total = [0.0]*len(Pzs)
    for k0_index, E0_k0 in enumerate(E0s):
        for band_index in range(len(E0_k0)):
            for z in range(len(Pzs)):
                result_layer_total[z] += result[k0_index][band_index][z]

    return result, result_layer_total

def plot_H_k0_phis(H_k0s, phis, Pzs, band_indices, E_F_base):
    qs = vec_linspace(np.array([0.0, 0.0, 0.0]), [0.3, 0.0, 0.0], 100)
    for H_k0, band_indices_k0 in zip(H_k0_phis(H_k0s, phis, Pzs), band_indices):
        Ekms = []
        for q in qs:
            Es, U = np.linalg.eigh(H_k0(q))
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

def get_phis(sigmas, d_bohr, E_below_V_bohr, epsilon_r):
    d_eps = d_bohr / (epsilon_r * __epsilon_0_F_bohr)
    phis = [0.0,
            -d_bohr * E_below_V_bohr - d_eps * sigmas[0],
            -2 * d_bohr * E_below_V_bohr - d_eps * (2 * sigmas[0] + sigmas[1])]
    return phis

def sigma_converged(sigmas, new_sigmas, tol_abs, tol_rel):
    if new_sigmas is None:
        return False

    for s, sp in zip(sigmas, new_sigmas):
        if abs(s - sp) > tol_abs:
            return False

        if abs(s - sp) > tol_rel * max(abs(s), abs(sp)):
            return False

    return True

def get_sigma_self_consistent(H_k0s, sigmas_initial, Pzs, band_indices, hole_density_bohr2, d_bohr, E_below_V_bohr, epsilon_r, tol_abs, tol_rel):
    sigmas = sigmas_initial
    new_sigmas = None

    while not sigma_converged(sigmas, new_sigmas, tol_abs, tol_rel):
        if new_sigmas is not None:
            sigmas = new_sigmas

        phis = get_phis(sigmas, d_bohr, E_below_V_bohr, epsilon_r)
        print("phis [V]")
        print(phis)

        E_F, E0s, curvatures = get_Fermi_energy(H_k0s, phis, Pzs, band_indices, hole_density_bohr2)
        print("E_F")
        print(E_F)

        new_nh, new_nh_layer_total = layer_hole_density_at_E(H_k0s, phis, Pzs, band_indices, E_F, E0s, curvatures)
        print("new_nh")
        print(new_nh)

        print("new_nh_layer_total")
        print(new_nh_layer_total)

        new_sigmas = [__e_C * n for n in new_nh_layer_total]
        print("new_sigmas")
        print(new_sigmas)

    return new_sigmas

def _main():
    np.set_printoptions(threshold=np.inf)

    parser = argparse.ArgumentParser("Plot band structure",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("prefix", type=str,
            help="Prefix for calculation")
    parser.add_argument("--subdir", type=str, default=None,
            help="Subdirectory under work_base where calculation was run")
    parser.add_argument("--num_layers", type=int, default=3,
            help="Number of layers")
    args = parser.parse_args()

    if args.num_layers != 3:
        assert("num_layers != 3 not implemented")

    work = _get_work(args.subdir, args.prefix)
    wannier_dir = os.path.join(work, "wannier")
    scf_path = os.path.join(wannier_dir, "scf.out")

    E_F_base = fermi_from_scf(scf_path)
    latVecs = latVecs_from_scf(scf_path)
    alat_Bohr = 1.0
    R = 2 * np.pi * np.linalg.inv(latVecs.T)

    Gamma_cart = np.array([0.0, 0.0, 0.0])
    K_lat = np.array([1/3, 1/3, 0.0])
    K_cart = np.dot(K_lat, R)
    Kprime_cart = -K_cart

    Hr_path = os.path.join(wannier_dir, "{}_hr.dat".format(args.prefix))
    Hr = extractHr(Hr_path)

    # TODO
    #E_below_V_nm = 0.5 # V/nm
    E_below_V_nm = 0.5
    d_A = 6.488 # Angstrom

    d_bohr = __bohr_per_Angstrom * d_A
    E_below_V_bohr = E_below_V_nm / (10 * __bohr_per_Angstrom)

    #hole_density_cm2 = 8e12
    #hole_density_cm2 = 1e10
    hole_density_cm2 = 1e12
    hole_density_bohr2 = hole_density_cm2 / (10**8 * __bohr_per_Angstrom)**2
    print("hole_density_bohr2")
    print(hole_density_bohr2)

    epsilon_r = 10.0 # TODO relative permittivity felt in trilayer


    # Choose initial potential assuming holes are distributed uniformally.
    sigma_layer_initial = (1/3) * __e_C * hole_density_bohr2
    print("sigma_layer_initial [C/bohr^2]")
    print(sigma_layer_initial)

    sigmas_initial = sigma_layer_initial * np.array([1.0, 1.0, 1.0])

    # Use full TB model -- TODO k dot p
    H_k0s, band_indices = [], []
    def Hfn(k0, q):
        return Hk(q + k0, Hr, latVecs)

    k0s = [Gamma_cart, K_cart, Kprime_cart]
    k0_num_orbitals = [2, 3, 3]
    for k0, num_orbitals in zip(k0s, k0_num_orbitals):
        H_k0s.append(partial(Hfn, k0))

        Es, U = np.linalg.eigh(H_k0s[-1](np.array([0.0, 0.0, 0.0])))
        top = top_valence_indices(E_F_base, num_orbitals, Es)
        band_indices.append(top)

    Pzs = get_layer_projections(args.num_layers)

    #phis_initial = get_phis(sigmas_initial)
    #plot_H_k0_phis(H_k0s, phis_initial, Pzs, band_indices, E_F_base)

    tol_abs = 1e-6 * sum(sigmas_initial)
    tol_rel = 1e-6

    converged_sigma = get_sigma_self_consistent(H_k0s, sigmas_initial, Pzs, band_indices, hole_density_bohr2, d_bohr, E_below_V_bohr, epsilon_r, tol_abs, tol_rel)

if __name__ == "__main__":
    _main()
