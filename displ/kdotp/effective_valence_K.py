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
from displ.kdotp.model_weights_K import vec_linspace, top_valence_indices
from displ.kdotp.separability_K import get_layer_projections

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

    # Momentum expectation values <z_{lp}| dH/dk_{c}|_K |z_l>
    ps = layer_Hamiltonian_ps(K_cart, Hr, latVecs, layer_basis)
    
    print("p")
    print(ps)

    # Inverse effective masses <z_{lp}| d^2H/dk_{cp}dk_{c}|_K |z_l>
    mstar_invs = layer_Hamiltonian_mstar_inverses(K_cart, Hr, latVecs, layer_basis)

    print("mstar_inv")
    print(mstar_invs)

    H_layers = []
    for k in ks:
        q = k - K_cart

        first_order = [q[c] * ps[c] for c in range(2)]

        second_order = []
        for cp in range(2):
            for c in range(2):
                second_order.append((1/2) * q[cp] * q[c] * mstar_invs[(cp, c)])

        H_layers.append(H_layer_K + sum(first_order) + sum(second_order))
        #H_layers.append(H_layer_K)

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

    plt.show()

if __name__ == "__main__":
    _main()
