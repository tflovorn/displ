from __future__ import division
import argparse
from copy import deepcopy
import os
import itertools
import numpy as np
import matplotlib.pyplot as plt
import numdifftools as nd
from displ.build.build import _get_work, band_path_labels
from displ.pwscf.parseScf import fermi_from_scf, latVecs_from_scf, alat_from_scf
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

def correction_Hamiltonian_derivs(k0_cart, Hr, latVecs, complement_basis, layer_basis):
    dHk_dk_k0 = dHk_dk(k0_cart, Hr, latVecs)

    derivs_PQ, derivs_QQ = [], []
    for c in range(2):
        dH_PQ = np.zeros([len(layer_basis), len(complement_basis)], dtype=np.complex128)
        dH_QQ = np.zeros([len(complement_basis), len(complement_basis)], dtype=np.complex128)

        for zp, zp_state in enumerate([v.conjugate().T for v in layer_basis]):
            for z, z_state in enumerate(complement_basis):
                dH_PQ[zp, z] = np.dot(zp_state, np.dot(dHk_dk_k0[c], z_state))[0, 0]

        derivs_PQ.append(dH_PQ)

        for zp, zp_state in enumerate([v.conjugate().T for v in complement_basis]):
            for z, z_state in enumerate(complement_basis):
                dH_QQ[zp, z] = np.dot(zp_state, np.dot(dHk_dk_k0[c], z_state))[0, 0]

        derivs_QQ.append(dH_QQ)

    return derivs_PQ, derivs_QQ

def correction_Hamiltonian_second_derivs(k0_cart, Hr, latVecs, complement_basis, layer_basis):
    d2Hk_dk_k0 = d2Hk_dk(k0_cart, Hr, latVecs)

    second_derivs_PQ, second_derivs_QQ = {}, {} # second_derivs[(cp, c)][zp, z]
    for cp in range(2):
        for c in range(2):
            second_derivs_PQ[(cp, c)] = np.zeros([len(layer_basis), len(complement_basis)], dtype=np.complex128)
            second_derivs_QQ[(cp, c)] = np.zeros([len(complement_basis), len(complement_basis)], dtype=np.complex128)

            for zp, zp_state in enumerate([v.conjugate().T for v in layer_basis]):
                for z, z_state in enumerate(complement_basis):
                    second_derivs_PQ[(cp, c)][zp, z] = np.dot(zp_state, np.dot(d2Hk_dk_k0[(cp, c)], z_state))[0, 0]

            for zp, zp_state in enumerate([v.conjugate().T for v in complement_basis]):
                for z, z_state in enumerate(complement_basis):
                    second_derivs_QQ[(cp, c)][zp, z] = np.dot(zp_state, np.dot(d2Hk_dk_k0[(cp, c)], z_state))[0, 0]

    return second_derivs_PQ, second_derivs_QQ

def correction_Hamiltonian_0th_order(k0_cart, Hr, latVecs, E_repr, complement_basis, layer_basis):
    H_QQ = correction_Hamiltonian_QQ(k0_cart, Hr, latVecs, complement_basis)
    H_PQ = correction_Hamiltonian_PQ(k0_cart, Hr, latVecs, complement_basis, layer_basis)

    center = np.linalg.inv(E_repr*np.eye(H_QQ.shape[0]) - H_QQ)
    Hprime = np.dot(H_PQ, np.dot(center, H_PQ.conjugate().T))

    return Hprime

def correction_Hamiltonian_ps(k0_cart, Hr, latVecs, E_repr, complement_basis, layer_basis):
    H_QQ = correction_Hamiltonian_QQ(k0_cart, Hr, latVecs, complement_basis)
    H_PQ = correction_Hamiltonian_PQ(k0_cart, Hr, latVecs, complement_basis, layer_basis)
    H_QP = H_PQ.conjugate().T

    derivs_PQ, derivs_QQ = correction_Hamiltonian_derivs(k0_cart, Hr, latVecs, complement_basis, layer_basis)
    derivs_QP = [M.conjugate().T for M in derivs_PQ]

    ps = []
    for c in range(2):
         Dinv = np.linalg.inv(E_repr*np.eye(H_QQ.shape[0]) - H_QQ)

         ps.append(derivs_PQ[c] @ Dinv @ H_QP
                 + H_PQ @ Dinv @ derivs_QP[c]
                 + H_PQ @ Dinv @ derivs_QQ[c] @ Dinv @ H_QP)

    return ps

def correction_Hamiltonian_mstar_inverses(k0_cart, Hr, latVecs, E_repr, complement_basis, layer_basis):
    H_QQ = correction_Hamiltonian_QQ(k0_cart, Hr, latVecs, complement_basis)
    H_PQ = correction_Hamiltonian_PQ(k0_cart, Hr, latVecs, complement_basis, layer_basis)
    H_QP = H_PQ.conjugate().T

    derivs_PQ, derivs_QQ = correction_Hamiltonian_derivs(k0_cart, Hr, latVecs, complement_basis, layer_basis)
    derivs_QP = [M.conjugate().T for M in derivs_PQ]

    d2s_PQ, d2s_QQ = correction_Hamiltonian_second_derivs(k0_cart, Hr, latVecs, complement_basis, layer_basis)
    d2s_QP = {k: M.conjugate().T for (k, M) in d2s_PQ.items()}

    mstar_invs_base, mstar_invs_other = {}, {}
    for cp in range(2):
        for c in range(2):
            Dinv = np.linalg.inv(E_repr*np.eye(H_QQ.shape[0]) - H_QQ)
            # Terms which are finite even if H_PQ = 0.
            base = (derivs_PQ[cp] @ Dinv @ derivs_QP[c]
                  + derivs_PQ[c] @ Dinv @ derivs_QP[cp])
            # Remaining terms.
            other = (d2s_PQ[(cp, c)] @ Dinv @ H_QP
                   + H_PQ @ Dinv @ d2s_QP[(cp, c)]
                   + derivs_PQ[c] @ Dinv @ derivs_QQ[cp] @ Dinv @ H_QP
                   + H_PQ @ Dinv @ derivs_QQ[cp] @ Dinv @ derivs_QP[c]
                   + derivs_PQ[cp] @ Dinv @ derivs_QQ[c] @ Dinv @ H_QP
                   + H_PQ @ Dinv @ d2s_QQ[(cp, c)] @ Dinv @ H_QP
                   + H_PQ @ Dinv @ derivs_QQ[c] @ Dinv @ derivs_QP[cp]
                   + H_PQ @ Dinv @ derivs_QQ[cp] @ Dinv @ derivs_QQ[c] @ Dinv @ H_QP
                   + H_PQ @ Dinv @ derivs_QQ[c] @ Dinv @ derivs_QQ[cp] @ Dinv @ H_QP)

            mstar_invs_base[(cp, c)] = base
            mstar_invs_other[(cp, c)] = other

    return mstar_invs_base, mstar_invs_other

def H_kdotp(q, H_layer_k0, H_correction, ps, ps_correction, mstar_invs, mstar_invs_correction_base,
        mstar_invs_correction_other):
    first_order = [q[c] * (ps[c] + ps_correction[c]) for c in range(2)]

    second_order = []
    for cp in range(2):
        for c in range(2):
            mstar_eff = (mstar_invs[(cp, c)]
                    + mstar_invs_correction_base[(cp, c)]
                    + mstar_invs_correction_other[(cp, c)])
            second_order.append((1/2) * q[cp] * q[c] * mstar_eff)

    return H_layer_k0 + H_correction + sum(first_order) + sum(second_order)

def effective_mass_band(Hfn, k0, band_index, alat_Bohr):
    def band_energy(k):
        Es, U = np.linalg.eigh(Hfn(k))
        return Es[band_index]

    curvature = nd.Hessian(band_energy)(k0)

    hbar_eV_s = 6.582119514e-16
    me_eV_per_c2 = 0.5109989461e6
    c_m_per_s = 2.99792458e8
    Bohr_m = 0.52917721067e-10
    fac = hbar_eV_s**2 / (me_eV_per_c2 * (c_m_per_s)**(-2) * (Bohr_m)**2 * alat_Bohr**2)

    mstars = [-fac/curvature[0, 0], -fac/curvature[1, 1], -fac/curvature[0, 1]]

    return mstars

def get_layer_basis_from_dm_K(U, top, Pzs, verbose):
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

def get_layer_basis_direct_K(U, top, Pzs, verbose):
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

    return layer_weights, layer_basis

def make_effective_Hamiltonian_K(k0_lat, subdir, prefix, get_layer_basis, verbose=False):
    num_layers = 3

    work = _get_work(subdir, prefix)
    wannier_dir = os.path.join(work, "wannier")
    scf_path = os.path.join(wannier_dir, "scf.out")
    wout_path = os.path.join(wannier_dir, "{}.wout".format(prefix))

    E_F = fermi_from_scf(scf_path)
    latVecs = latVecs_from_scf(scf_path)
    alat_Bohr = 1.0
    R = 2 * np.pi * np.linalg.inv(latVecs.T)

    K_cart = np.dot(k0_lat, R)

    if verbose:
        print(K_cart)
        print(latVecs)

    reduction_factor = 0.7
    num_ks = 100

    ks = vec_linspace(K_cart, reduction_factor*K_cart, num_ks)
    xs = np.linspace(1, reduction_factor, num_ks)

    Hr_path = os.path.join(wannier_dir, "{}_hr.dat".format(prefix))
    Hr = extractHr(Hr_path)

    Pzs = get_layer_projections(wout_path, num_layers)

    H_TB_K = Hk(K_cart, Hr, latVecs)
    Es, U = np.linalg.eigh(H_TB_K)

    top = top_valence_indices(E_F, 2*num_layers, Es)

    layer_weights, layer_basis = get_layer_basis(U, top, Pzs, verbose)

    complement_basis_mat = nullspace(array_with_rows(layer_basis).conjugate())
    complement_basis = []
    for i in range(complement_basis_mat.shape[1]):
        v = complement_basis_mat[:, [i]]
        complement_basis.append(v / np.linalg.norm(v))

    assert(len(layer_basis) + len(complement_basis) == 22*num_layers)

    for vl in [v.conjugate().T for v in layer_basis]:
        for vc in complement_basis:
            assert(abs(np.dot(vl, vc)[0, 0]) < 1e-12)

    # 0th order effective Hamiltonian: H(K) in layer basis.
    H_layer_K = layer_Hamiltonian_0th_order(H_TB_K, layer_basis)

    #E_repr = sum([Es[t] for t in top]) / len(top)
    E_repr = Es[top[0]]
    H_correction = correction_Hamiltonian_0th_order(K_cart, Hr, latVecs, E_repr, complement_basis, layer_basis)

    H0_tot = H_layer_K + H_correction

    H_PQ = correction_Hamiltonian_PQ(K_cart, Hr, latVecs, complement_basis, layer_basis)

    # Momentum expectation values <z_{lp}| dH/dk_{c}|_K |z_l>
    ps = layer_Hamiltonian_ps(K_cart, Hr, latVecs, layer_basis)

    ps_correction = correction_Hamiltonian_ps(K_cart, Hr, latVecs, E_repr, complement_basis, layer_basis)

    ps_tot = deepcopy(ps)
    for i, v in enumerate(ps_correction):
        ps_tot[i] += v

    # Inverse effective masses <z_{lp}| d^2H/dk_{cp}dk_{c}|_K |z_l>
    mstar_invs = layer_Hamiltonian_mstar_inverses(K_cart, Hr, latVecs, layer_basis)

    mstar_invs_correction_base, mstar_invs_correction_other = correction_Hamiltonian_mstar_inverses(K_cart, Hr, latVecs, E_repr, complement_basis, layer_basis)

    mstar_inv_tot = deepcopy(mstar_invs)
    for mstar_contrib in [mstar_invs_correction_base, mstar_invs_correction_other]:
        for k, v in mstar_contrib.items():
            mstar_inv_tot[k] += v

    if verbose:
        print("H0")
        print(H_layer_K)

        print("H_correction")
        print(H_correction)
        print("H_correction max")
        print(abs(H_correction).max())

        print("H_PQ max")
        print(abs(H_PQ).max())

        print("ps")
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
            q = k - K_cart

            H_layers.append(H_kdotp(q, H_layer_K, H_correction, ps, ps_correction,
                    mstar_invs, mstar_invs_correction_base,
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
        plt.clf()

        # Effective masses.
        print("effective mass, top valence band, TB model: m^*_{xx; yy; xy} / m_e")
        mstar_TB = effective_mass_band(lambda k: Hk(k, Hr, latVecs), K_cart, top[0], alat_Bohr)
        print(mstar_TB)

        print("effective mass, top valence band, k dot p model: m^*_{xx; yy; xy} / m_e")
        mstar_kdotp = effective_mass_band(lambda k: H_kdotp(k - K_cart, H_layer_K,
                H_correction, ps, ps_correction, mstar_invs,
                mstar_invs_correction_base, mstar_invs_correction_other),
                K_cart, len(layer_basis) - 1, alat_Bohr)
        print(mstar_kdotp)

        # Elements contributing to crossover scale.
        t_K = H0_tot[1, 2]
        t_K_25 = H0_tot[2, 5]
        Delta_K = H0_tot[1, 1] - H0_tot[2, 2]
        lambda_SO = H0_tot[1, 1] - H0_tot[0, 0]

        t_K_direct = H0_tot[1, 5]

        print("t_K, Delta_K, lambda_SO")
        print(t_K, Delta_K, lambda_SO)

        print("t_K_25")
        print(t_K_25)

        print("norm(t_K), norm(t_K_25)")
        print(abs(t_K), abs(t_K_25))

        print("t_K_direct, norm(t_K_direct)")
        print(t_K_direct, abs(t_K_direct))

        print("E_SL_K top")
        print([H0_tot[i, i] for i in [1, 3, 5]])

        print("E_SL_K bottom")
        print([H0_tot[i, i] for i in [0, 2, 4]])

        # Elements contributing to effective dielectric constant.
        print("Layer on-site energy differences:")
        print("Top group of bands: middle layer - bottom layer; top layer - middle layer")
        print(H0_tot[3, 3] - H0_tot[1, 1], H0_tot[5, 5] - H0_tot[3, 3])
        print("Bottom group of bands: middle layer - bottom layer; top layer - middle layer")
        print(H0_tot[2, 2] - H0_tot[0, 0], H0_tot[4, 4] - H0_tot[2, 2])

    return H0_tot, ps_tot, mstar_inv_tot

def _main():
    np.set_printoptions(threshold=np.inf)

    parser = argparse.ArgumentParser("Compute effective Hamiltonian at K for TMD trilayer system with 2H stacking",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("prefix", type=str,
            help="Prefix for calculation")
    parser.add_argument("--subdir", type=str, default=None,
            help="Subdirectory under work_base where calculation was run")
    parser.add_argument("--states_from_dm", action='store_true',
            help="Choose states from layer-projected density matrix")
    args = parser.parse_args()

    if args.states_from_dm:
        get_layer_basis = get_layer_basis_from_dm_K
    else:
        get_layer_basis = get_layer_basis_direct_K

    K_lat = np.array([1/3, 1/3, 0.0])

    make_effective_Hamiltonian_K(K_lat, args.subdir, args.prefix, get_layer_basis, verbose=True)

if __name__ == "__main__":
    _main()
