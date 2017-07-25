from __future__ import division
import argparse
import os
import itertools
import numpy as np
import matplotlib.pyplot as plt
from displ.build.build import _get_work, band_path_labels
from displ.pwscf.parseScf import fermi_from_scf, latVecs_from_scf
from displ.wannier.extractHr import extractHr
from displ.wannier.bands import Hk, dHk_dk, d2Hk_dk
from displ.kdotp.linalg import nullspace
from displ.kdotp.model_weights_K import vec_linspace, top_valence_indices
from displ.kdotp.separability_K import get_layer_projections, density_matrix

def layer_basis_from_dm(states, Pzs):
    dm = density_matrix(states, [1]*len(states))

    states_per_layer = len(states) // len(Pzs)
    layer_weights, layer_basis = [], []

    for Pz in Pzs:
        proj_dm = np.dot(Pz, np.dot(dm, Pz))
        proj_dm_evals, proj_dm_evecs = np.linalg.eigh(proj_dm)

        for i in range(states_per_layer):
            state_index = len(proj_dm_evals) - i - 1
            layer_weights.append(proj_dm_evals[state_index])
            layer_basis.append(proj_dm_evecs[:, [state_index]])

    return layer_weights, layer_basis

def array_with_rows(xs):
    A = np.zeros([len(xs), len(xs[0])], dtype=np.complex128)

    for i, x in enumerate(xs):
        A[i, :] = x.T

    return A

def layer_Hamiltonian_0th_order(H_TB_k0, layer_basis):
    H_layer_k0 = np.zeros([len(layer_basis), len(layer_basis)], dtype=np.complex128)

    for zp, zp_state in enumerate([v.conjugate().T for v in layer_basis]):
        for z, z_state in enumerate(layer_basis):
            H_layer_k0[zp, z] = np.dot(zp_state, np.dot(H_TB_k0, z_state))[0, 0]

    return H_layer_k0

def layer_Hamiltonian_ps(k0_cart, Hr, latVecs, layer_basis):
    dHk_dk_k0 = dHk_dk(k0_cart, Hr, latVecs)

    ps = [] # ps[c][zp, z]
    for c in range(2):
        ps.append(np.zeros([len(layer_basis), len(layer_basis)], dtype=np.complex128))

        for zp, zp_state in enumerate([v.conjugate().T for v in layer_basis]):
            for z, z_state in enumerate(layer_basis):
                ps[c][zp, z] = np.dot(zp_state, np.dot(dHk_dk_k0[c], z_state))[0, 0]

    return ps

def layer_Hamiltonian_mstar_inverses(k0_cart, Hr, latVecs, layer_basis):
    d2Hk_dk_k0 = d2Hk_dk(k0_cart, Hr, latVecs)

    mstar_invs = {} # mstar_invs[(cp, c)][zp, z]
    for cp in range(2):
        for c in range(2):
            mstar_invs[(cp, c)] = np.zeros([len(layer_basis), len(layer_basis)], dtype=np.complex128)

            for zp, zp_state in enumerate([v.conjugate().T for v in layer_basis]):
                for z, z_state in enumerate(layer_basis):
                    mstar_invs[(cp, c)][zp, z] = np.dot(zp_state, np.dot(d2Hk_dk_k0[(cp, c)], z_state))[0, 0]

    return mstar_invs

def correction_Hamiltonian_QQ(k0_cart, Hr, latVecs, complement_basis):
    Hk_k0 = Hk(k0_cart, Hr, latVecs)

    H_QQ = np.zeros([len(complement_basis), len(complement_basis)], dtype=np.complex128)
    for zp, zp_state in enumerate([v.conjugate().T for v in complement_basis]):
        for z, z_state in enumerate(complement_basis):
            H_QQ[zp, z] = np.dot(zp_state, np.dot(Hk_k0, z_state))[0, 0]

    return H_QQ

def correction_Hamiltonian_PQ(k0_cart, Hr, latVecs, complement_basis, layer_basis):
    Hk_k0 = Hk(k0_cart, Hr, latVecs)

    H_PQ = np.zeros([len(layer_basis), len(complement_basis)], dtype=np.complex128)

    for zp, zp_state in enumerate([v.conjugate().T for v in layer_basis]):
        for z, z_state in enumerate(complement_basis):
            H_PQ[zp, z] = np.dot(zp_state, np.dot(Hk_k0, z_state))[0, 0]

    return H_PQ

def correction_Hamiltonian_derivs_PQ(k0_cart, Hr, latVecs, complement_basis, layer_basis):
    dHk_dk_k0 = dHk_dk(k0_cart, Hr, latVecs)

    derivs_PQ = []
    for c in range(2):
        dH_PQ = np.zeros([len(layer_basis), len(complement_basis)], dtype=np.complex128)

        for zp, zp_state in enumerate([v.conjugate().T for v in layer_basis]):
            for z, z_state in enumerate(complement_basis):
                dH_PQ[zp, z] = np.dot(zp_state, np.dot(dHk_dk_k0[c], z_state))[0, 0]

        derivs_PQ.append(dH_PQ)

    return derivs_PQ

def correction_Hamiltonian_0th_order(k0_cart, Hr, latVecs, E_repr, complement_basis, layer_basis):
    H_QQ = correction_Hamiltonian_QQ(k0_cart, Hr, latVecs, complement_basis)
    H_PQ = correction_Hamiltonian_PQ(k0_cart, Hr, latVecs, complement_basis, layer_basis)

    center = np.linalg.inv(E_repr*np.eye(H_QQ.shape[0]) - H_QQ)
    Hprime = np.dot(H_PQ, np.dot(center, H_PQ.conjugate().T))

    return Hprime

def correction_Hamiltonian_mstar_inverses(k0_cart, Hr, latVecs, E_repr, complement_basis, layer_basis):
    H_QQ = correction_Hamiltonian_QQ(k0_cart, Hr, latVecs, complement_basis)
    derivs_PQ = correction_Hamiltonian_derivs_PQ(k0_cart, Hr, latVecs, complement_basis, layer_basis)

    mstar_invs = {}
    for cp in range(2):
        for c in range(2):
            center = np.linalg.inv(E_repr*np.eye(H_QQ.shape[0]) - H_QQ)
            Hprime = (np.dot(derivs_PQ[cp], np.dot(center, derivs_PQ[c].conjugate().T))
                    + np.dot(derivs_PQ[c], np.dot(center, derivs_PQ[cp].conjugate().T)))

            mstar_invs[(cp, c)] = Hprime

    return mstar_invs

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
    parser.add_argument("--states_from_dm", action='store_true',
            help="Choose states from layer-projected density matrix")
    args = parser.parse_args()

    if args.num_layers != 3:
        assert("num_layers != 3 not implemented")

    work = _get_work(args.subdir, args.prefix)
    wannier_dir = os.path.join(work, "wannier")
    scf_path = os.path.join(wannier_dir, "scf.out")

    E_F = fermi_from_scf(scf_path)
    latVecs = latVecs_from_scf(scf_path)
    R = 2 * np.pi * np.linalg.inv(latVecs.T)

    K_lat = np.array([1/3, 1/3, 0.0])
    K_cart = np.dot(K_lat, R)
    print(K_cart)
    print(latVecs)

    reduction_factor = 0.7
    num_ks = 100

    ks = vec_linspace(K_cart, reduction_factor*K_cart, num_ks)
    xs = np.linspace(1, reduction_factor, num_ks)

    Hr_path = os.path.join(wannier_dir, "{}_hr.dat".format(args.prefix))
    Hr = extractHr(Hr_path)

    Pzs = get_layer_projections(args.num_layers)

    H_TB_K = Hk(K_cart, Hr, latVecs)
    Es, U = np.linalg.eigh(H_TB_K)

    top = top_valence_indices(E_F, 2*args.num_layers, Es)

    if args.states_from_dm:
        top_states = [U[:, [t]] for t in top]

        layer_weights, layer_basis = layer_basis_from_dm(top_states, Pzs)
        print("layer weights")
        print(layer_weights)
        print("layer basis")
        for i, v in enumerate(layer_basis):
            print("state ", i)
            for j in range(len(v)):
                print(j, v[j])
    else:
        # Basis states for the effective Hamiltonian:
        # |P_{z = 0} m_0> (m_0 = highest valence state);
        # |P_{z = 1} m_1>
        # |P_{z = 2} m_0>.
        # TODO support arbitrary layer number
        layer_basis_indices = [(0, 0), (0, 4), (1, 1), (1, 5), (2, 0), (2, 4)]
        layer_basis = []

        for z, m in layer_basis_indices:
            Pz = Pzs[z]
        
            band_index = top[m]
            eigenstate = U[:, [band_index]]

            proj_state = np.dot(Pz, eigenstate)
            proj_state_normed = proj_state / np.linalg.norm(proj_state)

            layer_basis.append(proj_state_normed)

    complement_basis_mat = nullspace(array_with_rows(layer_basis).conjugate())
    complement_basis = []
    for i in range(complement_basis_mat.shape[1]):
        v = complement_basis_mat[:, [i]]
        complement_basis.append(v / np.linalg.norm(v))

    assert(len(layer_basis) + len(complement_basis) == 22*args.num_layers)

    for vl in [v.conjugate().T for v in layer_basis]:
        for vc in complement_basis:
            assert(abs(np.dot(vl, vc)[0, 0]) < 1e-12)

    # Mirror operation:
    # M|m_0> = |m_0>; M|m_2> = -|m_2>; M|m_1> = |m_1> (? depends on M|p_z>).
    # Want to preserve this is TB basis.

    # |z_0> = |P_{z = 0}, m0> + |P_{z = 0}, m2>
    # |z_1> = |P_{z = 1}, m1>
    # |z_2> = |P_{z = 2}, m0> - |P_{z = 2}, m2>
    #layer_states = [
    #        U[:, top[0]] + U[:, top[2]],
    #        U[:, top[1]],
    #        U[:, top[0]] - U[:, top[2]]
    #]
    #proj_states = [np.dot(Pzs[z], v) for z, v in enumerate(layer_states)]
    #layer_basis = [v / np.linalg.norm(v) for v in proj_states]

    # 0th order effective Hamiltonian: H(K) in layer basis.
    H_layer_K = layer_Hamiltonian_0th_order(H_TB_K, layer_basis)

    print("H0")
    print(H_layer_K)

    #E_repr = sum([Es[t] for t in top]) / len(top)
    E_repr = Es[top[0]]
    H_correction = correction_Hamiltonian_0th_order(K_cart, Hr, latVecs, E_repr, complement_basis, layer_basis)
    print("H_correction")
    print(H_correction)
    print("H_correction max")
    print(abs(H_correction).max())

    H_PQ = correction_Hamiltonian_PQ(K_cart, Hr, latVecs, complement_basis, layer_basis)
    print("H_PQ max")
    print(abs(H_PQ).max())

    # Momentum expectation values <z_{lp}| dH/dk_{c}|_K |z_l>
    ps = layer_Hamiltonian_ps(K_cart, Hr, latVecs, layer_basis)
    
    print("p")
    print(ps)

    # Inverse effective masses <z_{lp}| d^2H/dk_{cp}dk_{c}|_K |z_l>
    mstar_invs = layer_Hamiltonian_mstar_inverses(K_cart, Hr, latVecs, layer_basis)

    print("mstar_inv")
    print(mstar_invs)

    mstar_invs_correction = correction_Hamiltonian_mstar_inverses(K_cart, Hr, latVecs, E_repr, complement_basis, layer_basis)

    print("mstar_inv_correction")
    print(mstar_invs_correction)

    H_layers = []
    for k in ks:
        q = k - K_cart

        first_order = [q[c] * ps[c] for c in range(2)]

        second_order = []
        for cp in range(2):
            for c in range(2):
                mstar_eff = mstar_invs[(cp, c)] + mstar_invs_correction[(cp, c)]
                second_order.append((1/2) * q[cp] * q[c] * mstar_eff)

        #H_layers.append(H_layer_K + sum(first_order) + sum(second_order))
        H_layers.append(H_layer_K + H_correction + sum(first_order) + sum(second_order))

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

if __name__ == "__main__":
    _main()
