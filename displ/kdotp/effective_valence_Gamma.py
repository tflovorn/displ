from __future__ import division
import argparse
import os
import itertools
import numpy as np
import matplotlib.pyplot as plt
from displ.build.build import _get_work, band_path_labels
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

def get_layer_basis_Gamma(U, top, Pzs, verbose):
    top_states = [U[:, [t]] for t in top]

    layer_weights, layer_basis = layer_basis_from_dm(top_states, Pzs)

    if verbose:
        print("layer weights")
        print(layer_weights)
        print("layer basis")
        for i, v in enumerate(layer_basis):
            print("state ", i)
            for j in range(len(v)):
                print(j, v[j])

    return layer_weights, layer_basis

def make_effective_Hamiltonian_Gamma(subdir, prefix, top_two_only, verbose=False):
    num_layers = 3

    work = _get_work(subdir, prefix)
    wannier_dir = os.path.join(work, "wannier")
    scf_path = os.path.join(wannier_dir, "scf.out")

    E_F = fermi_from_scf(scf_path)
    latVecs = latVecs_from_scf(scf_path)
    alat_Bohr = 1.0
    R = 2 * np.pi * np.linalg.inv(latVecs.T)

    Gamma_cart = np.array([0.0, 0.0, 0.0])
    K_lat = np.array([1/3, 1/3, 0.0])
    K_cart = np.dot(K_lat, R)

    if verbose:
        print(K_cart)
        print(latVecs)

    upto_factor = 0.3
    num_ks = 100

    ks = vec_linspace(Gamma_cart, upto_factor*K_cart, num_ks)
    xs = np.linspace(0.0, upto_factor, num_ks)

    Hr_path = os.path.join(wannier_dir, "{}_hr.dat".format(prefix))
    Hr = extractHr(Hr_path)

    if top_two_only:
        Pzs = [np.eye(get_total_orbitals(num_layers))]
    else:
        Pzs = get_layer_projections(num_layers)

    H_TB_Gamma = Hk(Gamma_cart, Hr, latVecs)
    Es, U = np.linalg.eigh(H_TB_Gamma)

    if top_two_only:
        top = top_valence_indices(E_F, 2, Es)
    else:
        top = top_valence_indices(E_F, 2*num_layers, Es)

    layer_weights, layer_basis = get_layer_basis_Gamma(U, top, Pzs, verbose)

    complement_basis_mat = nullspace(array_with_rows(layer_basis).conjugate())
    complement_basis = []
    for i in range(complement_basis_mat.shape[1]):
        v = complement_basis_mat[:, [i]]
        complement_basis.append(v / np.linalg.norm(v))

    assert(len(layer_basis) + len(complement_basis) == 22*num_layers)

    for vl in [v.conjugate().T for v in layer_basis]:
        for vc in complement_basis:
            assert(abs(np.dot(vl, vc)[0, 0]) < 1e-12)

    # 0th order effective Hamiltonian: H(Gamma) in layer basis.
    H_layer_Gamma = layer_Hamiltonian_0th_order(H_TB_Gamma, layer_basis)

    #E_repr = sum([Es[t] for t in top]) / len(top)
    E_repr = Es[top[0]]
    H_correction = correction_Hamiltonian_0th_order(Gamma_cart, Hr, latVecs, E_repr, complement_basis, layer_basis)

    H0_tot = H_layer_Gamma + H_correction

    H_PQ = correction_Hamiltonian_PQ(K_cart, Hr, latVecs, complement_basis, layer_basis)

    # Momentum expectation values <z_{lp}| dH/dk_{c}|_Gamma |z_l>
    ps = layer_Hamiltonian_ps(Gamma_cart, Hr, latVecs, layer_basis)
    
    ps_correction = correction_Hamiltonian_ps(Gamma_cart, Hr, latVecs, E_repr, complement_basis, layer_basis)

    ps_tot = ps
    for i, v in enumerate(ps_correction):
        ps_tot[i] += v
    
    # Inverse effective masses <z_{lp}| d^2H/dk_{cp}dk_{c}|_Gamma |z_l>
    mstar_invs = layer_Hamiltonian_mstar_inverses(Gamma_cart, Hr, latVecs, layer_basis)

    mstar_invs_correction_base, mstar_invs_correction_other = correction_Hamiltonian_mstar_inverses(Gamma_cart, Hr, latVecs, E_repr, complement_basis, layer_basis)

    mstar_inv_tot = mstar_invs
    for mstar_contrib in [mstar_invs_correction_base, mstar_invs_correction_other]:
        for k, v in mstar_contrib.items():
            mstar_inv_tot[k] += v

    if verbose:
        print("H0")
        print(H_layer_Gamma)

        print("H_correction")
        print(H_correction)
        print("H_correction max")
        print(abs(H_correction).max())

        print("H_PQ max")
        print(abs(H_PQ).max())

        print("p")
        print(ps)

        print("ps max")
        print(max([abs(x).max() for x in ps]))

        print("ps correction")
        print(ps_correction)

        print("ps_correction max")
        print(max([abs(x).max() for x in ps_correction]))

        print("mstar_inv")
        print(mstar_invs)

        print("mstar_inv max")
        print(max([abs(v).max() for k, v in mstar_invs.items()]))

        print("mstar_inv_correction_base")
        print(mstar_invs_correction_base)

        print("mstar_inv_correction_base max")
        print(max([abs(v).max() for k, v in mstar_invs_correction_base.items()]))

        print("mstar_inv_correction_other")
        print(mstar_invs_correction_other)

        print("mstar_inv_correction_other max")
        print(max([abs(v).max() for k, v in mstar_invs_correction_other.items()]))

        # Fit quality plots.
        H_layers = []
        for k in ks:
            q = k - Gamma_cart

            H_layers.append(H_kdotp(q, H_layer_Gamma, H_correction, ps,
                    ps_correction, mstar_invs, mstar_invs_correction_base,
                    mstar_invs_correction_other))

        Emks, Umks = [], []
        for band_index in range(len(layer_basis)):
            Emks.append([])
            Umks.append([])

        for k_index, Hk_layers in enumerate(H_layers):
            Es, U = np.linalg.eigh(Hk_layers)
            #print(k_index)
            #print("U", U)

            for band_index in range(len(layer_basis)):
                Emks[band_index].append(Es)
                Umks[band_index].append(U)

        for band_index in range(len(layer_basis)):
            plt.plot(xs, Emks[band_index])

        TB_Emks = []
        for m in range(len(top)):
            TB_Emks.append([])

        for k in ks:
            this_H_TB_k = Hk(k, Hr, latVecs)
            this_Es, this_U = np.linalg.eigh(this_H_TB_k)

            for m, i in enumerate(top):
                TB_Emks[m].append(this_Es[i])

        for TB_Em in TB_Emks:
            plt.plot(xs, TB_Em, 'k.')

        plt.show()

        # Effective masses.
        print("effective mass, top valence band, TB model: m^*_{xx; yy; xy}")
        print(effective_mass_band(lambda k: Hk(k, Hr, latVecs), Gamma_cart, top[0], alat_Bohr))

        print("effective mass, top valence band, k dot p model: m^*_{xx; yy; xy}")
        print(effective_mass_band(lambda k: H_kdotp(k - Gamma_cart, H_layer_Gamma,
                H_correction, ps, ps_correction, mstar_invs,
                mstar_invs_correction_base, mstar_invs_correction_other),
                Gamma_cart, len(layer_basis) - 1, alat_Bohr))

        # Elements contributing to crossover scale.
        t_Gamma_a = H0_tot[0, 2]
        t_Gamma_b = H0_tot[0, 3]

        print("t_Gamma_a, t_Gamma_b")
        print(t_Gamma_a, t_Gamma_b)

        print("norm(t_Gamma_a)")
        print(abs(t_Gamma_a))

        print("[t_Gamma]_{2, 4}, [t_Gamma]_{2, 5}")
        print(H0_tot[2, 4], H0_tot[2, 5])

        t_Gamma_direct = H0_tot[0, 4]
        print("t_Gamma_direct, norm(t_Gamma_direct)")
        print(t_Gamma_direct, abs(t_Gamma_direct))

        print("E_SL_Gamma")
        print([H0_tot[i, i] for i in range(6)])

        print("H0_tot")
        print(H0_tot)

    return H0_tot, ps_tot, mstar_inv_tot

def _main():
    np.set_printoptions(threshold=np.inf)

    parser = argparse.ArgumentParser("Compute effective Hamiltonian at Gamma for TMD trilayer system with 2H stacking",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("prefix", type=str,
            help="Prefix for calculation")
    parser.add_argument("--subdir", type=str, default=None,
            help="Subdirectory under work_base where calculation was run")
    parser.add_argument("--top_two_only", action='store_true',
            help="Consider only the top two valence states, without layer-projection")
    args = parser.parse_args()

    make_effective_Hamiltonian_Gamma(args.subdir, args.prefix, args.top_two_only, verbose=True)

if __name__ == "__main__":
    _main()
