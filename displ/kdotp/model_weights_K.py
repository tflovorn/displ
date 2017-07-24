from __future__ import division
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from displ.build.build import _get_work, band_path_labels
from displ.pwscf.parseScf import fermi_from_scf
from displ.wannier.extractHr import extractHr
from displ.wannier.bands import Hk_recip

def vec_linspace(start, stop, N):
    step = (stop - start) / (N - 1)

    vecs = []
    for i in range(N):
        vecs.append(start + i*step)

    return vecs

def top_valence_indices(E_F, N, Es):
    '''Return the index of the top N valence bands.
    '''
    first_above_E_F = None
    for i, E in enumerate(Es):
        if E > E_F:
            first_above_E_F = i
            break

    if first_above_E_F - N < 0:
        raise ValueError("fewer than N bands in valence bands")

    return [first_above_E_F - i for i in range(1, N+1)]

def basis_state_labels(num_layers):
    p_orbitals = ["pz", "px", "py"]
    d_orbitals = ["dz2", "dxz", "dyz", "dx2-y2", "dxy"]

    labels = []
    for z in range(num_layers):
        for pos in ["low", "high"]:
            for orb in p_orbitals:
                for spin in ["up", "down"]:
                    labels.append("z_{}_{}_{}_{}".format(z, pos, orb, spin))

    for z in range(num_layers):
        for orb in d_orbitals:
            for spin in ["up", "down"]:
                labels.append("z_{}_{}_{}".format(z, orb, spin))

    return labels

def mirror_op():
    z_map = [2, 1, 0]
    p_pos_map = [1, 0]
    p_orb_sign = [-1, 1, 1]
    d_orb_sign = [1, -1, -1, 1, 1]

    total_orbitals = 66
    M = np.zeros([total_orbitals, total_orbitals], dtype=np.complex128)
    i = 0
    # p orbitals
    for z in range(3):
        for p_pos in range(2):
            for p_orb in range(3):
                for spin in range(2):
                    ip = orbital_pos(z_map[z], "p", p_pos_map[p_pos], p_orb, spin)
                    M[ip, i] = p_orb_sign[p_orb]
                    print("p", i, ip, z, p_pos, p_orb, spin)
                    i += 1

    # d orbitals
    for z in range(3):
        for d_orb in range(5):
            for spin in range(2):
                ip = orbital_pos(z_map[z], "d", None, d_orb, spin)
                M[ip, i] = d_orb_sign[d_orb]
                print("d", i, ip, z, d_orb, spin)
                i += 1

    assert((np.dot(M, M) == np.eye(M.shape[0])).all())
    assert((M.conjugate().T == M).all())

    basis = basis_state_labels(3)
    for ip in range(total_orbitals):
        for i in range(total_orbitals):
            if M[ip, i] != 0:
                print(ip, i, basis[ip], basis[i], M[ip, i].real)

    return M

def orbital_pos(z, p_or_d, p_pos, orb, spin):
    if p_or_d == "p":
        z_base = (3*2*2) * z
        return z_base + (3*2) * p_pos + 2 * orb + spin
    else:
        z_base = (3*2*2) * 3 + (5*2) * z
        return z_base + 2*orb + spin

def _main():
    parser = argparse.ArgumentParser("Plot band structure",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("prefix", type=str,
            help="Prefix for calculation")
    parser.add_argument("--subdir", type=str, default=None,
            help="Subdirectory under work_base where calculation was run")
    parser.add_argument("--num_layers", type=int, default=3,
            help="Number of layers (required if group_layer_* options given)")
    args = parser.parse_args()

    if args.num_layers != 3:
        raise ValueError("mirror check not implemented for num_layers != 3")

    work = _get_work(args.subdir, args.prefix)
    wannier_dir = os.path.join(work, "wannier")
    scf_path = os.path.join(wannier_dir, "scf.out")

    E_F = fermi_from_scf(scf_path)

    Hr_path = os.path.join(wannier_dir, "{}_hr.dat".format(args.prefix))
    Hr = extractHr(Hr_path)

    K = np.array([1/3, 1/3, 0.0])

    reduction_factor = 0.7
    num_ks = 100

    ks = vec_linspace(K, reduction_factor*K, num_ks)
    xs = np.linspace(1, reduction_factor, num_ks)

    basis = basis_state_labels(args.num_layers)
    M = mirror_op()

    # Assume SOC present and that model has 2*3 X(p) orbitals per layer
    # and 2*5 M(d) in canonical Wannier90 order.
    # Assumes atoms are ordered with all Xs first, then all Ms, and within
    # M/X groups the atoms are in layer order.
    orbitals_per_X = 6
    orbitals_per_M = 10

    # Base index for orbitals of each layer:
    base_orbitals = [args.num_layers * 2 * orbitals_per_X + z * orbitals_per_M for z in range(args.num_layers)]

    # Orbitals contributing to j = +5/2 state:
    x2y2_up = [base + 6 for base in base_orbitals]
    xy_up = [base + 8 for base in base_orbitals]
    x2y2_dn = [base + 7 for base in base_orbitals]
    xy_dn = [base + 9 for base in base_orbitals]

    num_top_bands = 2*args.num_layers
    weights = []
    for i in range(num_top_bands):
        weights.append([])

    for k in ks:
        Hk = Hk_recip(k, Hr)
        Es, U = np.linalg.eigh(Hk)

        top = top_valence_indices(E_F, num_top_bands, Es)

        for i, band in enumerate(top):
            total = 0

            state = U[:, band]

            mirror_eval = np.dot(state.conjugate().T, np.dot(M, state))[0, 0]
            print("band {}; mirror <v|M|v> = {}".format(band, mirror_eval))
            print("mirror deviation elements")
            #Mdev = np.dot(M, state) - mirror_signs[i] * state
            #for n, v in enumerate(Mdev):
            #    if abs(v)**2 > 1e-3:
            #        print(n, basis[n], v, abs(v)**2)

            print("band, orb, orb_label, weight, evec comp")
            #for n, v in enumerate(state):
            #    if abs(v)**2 > 1e-2:
            #        print(band, n, basis[n], abs(v)**2, v)
            for n, v in enumerate(state):
                print(band, n, basis[n], abs(v)**2, v)

            for z, (n1_up, n2_up, n1_dn, n2_dn) in enumerate(zip(x2y2_up, xy_up, x2y2_dn, xy_dn)):
                # Appropriate arrangement for K.
                # TODO - swap for -K.
                # U[n, i] = <n|i> --> <i|n> = U[n, i].conjugate()
                if z % 2 == 0:
                    evec_comp = (1/np.sqrt(2)) * (U[n1_dn, band].conjugate() - 1j*U[n2_dn, band].conjugate())
                else:
                    evec_comp = (1/np.sqrt(2)) * (U[n1_up, band].conjugate() + 1j*U[n2_up, band].conjugate())

                total += abs(evec_comp)**2

            weights[i].append(1 - total)

    for i, band_weights in enumerate(weights):
        plt.plot(xs, band_weights, label="Band {}".format(i))

    plt.legend(loc=0)
    plt.xlabel("k / K", fontsize='large')
    plt.ylabel("Weight outside l_z = +/- 2, up/down group")
    plt.savefig("model_weights_K.png", bbox_inches='tight', dpi=500)

if __name__ == "__main__":
    _main()
