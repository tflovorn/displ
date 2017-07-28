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
    parser.add_argument("--top_two_only", action='store_true',
            help="Consider only the top two valence states, without layer-projection")
    args = parser.parse_args()

    if args.num_layers != 3:
        assert("num_layers != 3 not implemented")

    work = _get_work(args.subdir, args.prefix)
    wannier_dir = os.path.join(work, "wannier")
    scf_path = os.path.join(wannier_dir, "scf.out")

    E_F = fermi_from_scf(scf_path)
    latVecs = latVecs_from_scf(scf_path)
    alat_Bohr = 1.0
    R = 2 * np.pi * np.linalg.inv(latVecs.T)

    Gamma_cart = np.array([0.0, 0.0, 0.0])
    K_lat = np.array([1/3, 1/3, 0.0])
    K_cart = np.dot(K_lat, R)
    print(K_cart)
    print(latVecs)

    upto_factor = 0.3
    num_ks = 100

    ks = vec_linspace(Gamma_cart, upto_factor*K_cart, num_ks)
    xs = np.linspace(0.0, upto_factor, num_ks)

    Hr_path = os.path.join(wannier_dir, "{}_hr.dat".format(args.prefix))
    Hr = extractHr(Hr_path)

    if args.top_two_only:
        Pzs = [np.eye(get_total_orbitals(args.num_layers))]
    else:
        Pzs = get_layer_projections(args.num_layers)

    H_TB_Gamma = Hk(Gamma_cart, Hr, latVecs)
    Es, U = np.linalg.eigh(H_TB_Gamma)

    if args.top_two_only:
        top = top_valence_indices(E_F, 2, Es)
    else:
        top = top_valence_indices(E_F, 2*args.num_layers, Es)

    top_states = [U[:, [t]] for t in top]

    layer_weights, layer_basis = layer_basis_from_dm(top_states, Pzs)
    print("layer weights")
    print(layer_weights)
    print("layer basis")
    for i, v in enumerate(layer_basis):
        print("state ", i)
        for j in range(len(v)):
            print(j, v[j])

    complement_basis_mat = nullspace(array_with_rows(layer_basis).conjugate())
    complement_basis = []
    for i in range(complement_basis_mat.shape[1]):
        v = complement_basis_mat[:, [i]]
        complement_basis.append(v / np.linalg.norm(v))

    assert(len(layer_basis) + len(complement_basis) == 22*args.num_layers)

    for vl in [v.conjugate().T for v in layer_basis]:
        for vc in complement_basis:
            assert(abs(np.dot(vl, vc)[0, 0]) < 1e-12)

    # 0th order effective Hamiltonian: H(Gamma) in layer basis.
    H_layer_Gamma = layer_Hamiltonian_0th_order(H_TB_Gamma, layer_basis)

    print("H0")
    print(H_layer_Gamma)

    #E_repr = sum([Es[t] for t in top]) / len(top)
    E_repr = Es[top[0]]
    H_correction = correction_Hamiltonian_0th_order(Gamma_cart, Hr, latVecs, E_repr, complement_basis, layer_basis)
    print("H_correction")
    print(H_correction)
    print("H_correction max")
    print(abs(H_correction).max())

    H_PQ = correction_Hamiltonian_PQ(K_cart, Hr, latVecs, complement_basis, layer_basis)
    print("H_PQ max")
    print(abs(H_PQ).max())

    # Momentum expectation values <z_{lp}| dH/dk_{c}|_Gamma |z_l>
    ps = layer_Hamiltonian_ps(Gamma_cart, Hr, latVecs, layer_basis)
    
    print("p")
    print(ps)

    print("ps max")
    print(max([abs(x).max() for x in ps]))

    ps_correction = correction_Hamiltonian_ps(Gamma_cart, Hr, latVecs, E_repr, complement_basis, layer_basis)
    
    print("ps correction")
    print(ps_correction)

    print("ps_correction max")
    print(max([abs(x).max() for x in ps_correction]))

    # Inverse effective masses <z_{lp}| d^2H/dk_{cp}dk_{c}|_Gamma |z_l>
    mstar_invs = layer_Hamiltonian_mstar_inverses(Gamma_cart, Hr, latVecs, layer_basis)

    print("mstar_inv")
    print(mstar_invs)

    print("mstar_inv max")
    print(max([abs(v).max() for k, v in mstar_invs.items()]))

    mstar_invs_correction_base, mstar_invs_correction_other = correction_Hamiltonian_mstar_inverses(Gamma_cart, Hr, latVecs, E_repr, complement_basis, layer_basis)

    print("mstar_inv_correction_base")
    print(mstar_invs_correction_base)

    print("mstar_inv_correction_base max")
    print(max([abs(v).max() for k, v in mstar_invs_correction_base.items()]))

    print("mstar_inv_correction_other")
    print(mstar_invs_correction_other)

    print("mstar_inv_correction_other max")
    print(max([abs(v).max() for k, v in mstar_invs_correction_other.items()]))

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

    print("effective mass, top valence band, TB model: m^*_{xx; yy; xy}")
    print(effective_mass_band(lambda k: Hk(k, Hr, latVecs), Gamma_cart, top[0], alat_Bohr))

    print("effective mass, top valence band, k dot p model: m^*_{xx; yy; xy}")
    print(effective_mass_band(lambda k: H_kdotp(k - Gamma_cart, H_layer_Gamma,
            H_correction, ps, ps_correction, mstar_invs,
            mstar_invs_correction_base, mstar_invs_correction_other),
            Gamma_cart, len(layer_basis) - 1, alat_Bohr))

if __name__ == "__main__":
    _main()
