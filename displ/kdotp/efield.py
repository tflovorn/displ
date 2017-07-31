from __future__ import division
import argparse
import os
import itertools
import numpy as np
import numdifftools as nd
from scipy.optimize import bisect
import matplotlib.pyplot as plt
from displ.build.build import _get_work, band_path_labels
from displ.pwscf.parseScf import fermi_from_scf, latVecs_from_scf, alat_from_scf
from displ.wannier.extractHr import extractHr
from displ.wannier.bands import Hk
from displ.kdotp.model_weights_K import top_valence_indices
from displ.kdotp.separability_K import get_layer_projections

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
    for H in H_k0s:
        def Hq(q):
            H_base = H(q)
            # e * phi has eV units.
            phi_parts = [-phi * Pz for phi, Pz in zip(phis, Pzs)]
            return H_base + sum(phi_parts)

        result.append(Hq)

    return result

def energies_at_k0(H_k0s, phis, Pzs, band_indices):
    '''E^0_{k0;n0} = result[k0_index][n0]

    [eV]
    '''
    result = []
    for H_k0, band_indices_k0 in zip(H_k0_phis(H_k0s, phis, Pzs), band_indices):
        q0 = [0.0, 0.0, 0.0]
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
    result = []
    for H_k0, band_indices_k0 in zip(H_k0_phis(H_k0s, phis, Pzs), band_indices):
        result.append([])
        for n in band_indices_k0:
            print(n)
            def band_energy(q):
                Es, U = np.linalg.eigh(H_k0(q))
                return Es[n]

            curvatures = nd.Hessian(band_energy)([0.0, 0.0, 0.0])
            # TODO - getting 0 result for this. why?
            result[-1].append([curvatures[0, 0], curvatures[1, 1]])

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
            print(cx, cy)
            val = np.sqrt(cx/cy) * (1/(4*np.pi)) * (1/cx) * (E0 - E) * step(E0 - E)
            result[-1].append(val)

    return result

def get_Fermi_energy(H_k0s, phis, Pzs, band_indices, hole_density):
    E0s = energies_at_k0(H_k0s, phis, Pzs, band_indices)
    curvatures = band_curvatures(H_k0s, phis, Pzs, band_indices)
    print("in get_Fermi_energy got E0s = ")
    print(E0s)
    print("curvatures = ")
    print(curvatures)

    def error_fn(E):
        return hole_density - sum([sum(hk) for hk in hole_density_at_E(E0s, curvatures, E)])

    E_min = min([min(E0s_k0) for E0s_k0 in E0s])
    E_max = max([max(E0s_k0) for E0s_k0 in E0s])

    return bisect(error_fn, E_min, E_max)

def get_layer_weights(H_k0s, phis, Pzs, band_indices):
    '''Returns list of weight[k0_index][n0][layer_index] [unitless]
    '''
    result = []
    for H_k0, band_indices_k0 in zip(H_k0_phis(H_k0s, phis, Pzs), band_indices):
        result.append([])

        q0 = [0.0, 0.0, 0.0]
        Es, U = np.linalg.eigh(H_k0(q0))

        for n in band_indices_k0:
            result[-1].append([])

            state = U[:, [n]]
            for Pz in Pzs:
                weight = state.conjugate().T @ Pz @ state
                result[-1][-1].append(weight)

    return result

def layer_hole_density_at_E(H_k0s, phis, Pzs, band_indices, E):
    '''n_h^A(l) [1/Bohr^2]
    '''
    E0s = energies_at_k0(H_k0s, phis, Pzs, band_indices)
    curvatures = band_curvatures(H_k0s, phis, Pzs, band_indices)

    layer_weights = get_layer_weights(H_k0s, phis, Pzs, band_indices)

    result = [0.0]*len(Pzs)
    for E0_k0, curvature_k0, weights_k0 in zip(E0s, curvatures, layer_weights):
        for E0, curvature, weights_k0_n in zip(E0_k0, curvature_k0, weights_k0):
            for z, weight in enumerate(weights_k0_n):
                cx, cy = curvature
                result[z] += np.sqrt(cx / cy) * weight * (1/(4*np.pi)) * (1/cx) * (E0 - E) * step(E0 - E)

    return result

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
    E_below_V_nm = 0.5 # V/nm
    d_A = 6.488 # Angstrom

    bohr_per_Angstrom = 1.889726164

    d_bohr = bohr_per_Angstrom * d_A
    E_below_V_bohr = E_below_V_nm / (10 * bohr_per_Angstrom)

    hole_density_cm2 = 8e12
    hole_density_bohr2 = hole_density_cm2 / (10**8 * bohr_per_Angstrom)**2
    print("hole_density_bohr2")
    print(hole_density_bohr2)

    epsilon_r = 1.0 # TODO relative permittivity felt in trilayer
    epsilon_0_F_m = 8.8541878e-12 # F/m = C/Vm
    epsilon_0_F_bohr = epsilon_0_F_m / (10**10 * bohr_per_Angstrom)

    e_C = 1.60217662e-19 # C

    # Choose initial potential assuming holes are distributed uniformally.
    sigma_layer_initial = (1/3) * e_C * hole_density_bohr2
    print("sigma_layer_initial [C/bohr^2]")
    print(sigma_layer_initial)

    phis_initial = [0.0,
            -d_bohr * E_below_V_bohr - sigma_layer_initial * d_bohr / (epsilon_r * epsilon_0_F_bohr),
            -2 * d_bohr * E_below_V_bohr - 3 * sigma_layer_initial * d_bohr / (epsilon_r * epsilon_0_F_bohr)]

    print("phis_initial [V]")
    print(phis_initial)

    # Use full TB model -- TODO k dot p
    H_k0s, band_indices = [], []
    for k0 in [Gamma_cart, K_cart, Kprime_cart]:
        def Hfn(q):
            return Hk(k0, Hr, latVecs)

        H_k0s.append(Hfn)

        Es, U = np.linalg.eigh(Hfn([0.0, 0.0, 0.0]))
        top = top_valence_indices(E_F_base, 2*args.num_layers, Es)
        band_indices.append(top)

    Pzs = get_layer_projections(args.num_layers)

    E_F = get_Fermi_energy(H_k0s, phis_initial, Pzs, band_indices, hole_density_bohr2)
    print("E_F")
    print(E_F)

    new_sigmas = layer_hole_density_at_E(H_k0s, phis_initial, Pzs, band_indices, E_F)
    print("new_sigmas")
    print(new_sigmas)

if __name__ == "__main__":
    _main()
