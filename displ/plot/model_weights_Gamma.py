from __future__ import division
import argparse
import os
import itertools
import numpy as np
import matplotlib.pyplot as plt
from displ.build.build import _get_work, band_path_labels
from displ.pwscf.parseScf import fermi_from_scf
from displ.wannier.extractHr import extractHr
from displ.wannier.bands import Hk_recip
from displ.plot.model_weights_K import (vec_linspace, top_valence_indices,
        basis_state_labels, mirror_op)

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

    Gamma = np.array([0.0, 0.0, 0.0])
    K = np.array([1/3, 1/3, 0.0])

    max_K_factor = 0.3
    num_ks = 2

    ks = vec_linspace(Gamma, max_K_factor*K, num_ks)
    xs = np.linspace(0.0, max_K_factor, num_ks)

    basis = basis_state_labels(args.num_layers)
    M = mirror_op()

    # Assume SOC present and that model has 2*3 X(p) orbitals per layer
    # and 2*5 M(d) in canonical Wannier90 order.
    # Assumes atoms are ordered with all Xs first, then all Ms, and within
    # M/X groups the atoms are in layer order.
    orbitals_per_X = 6
    orbitals_per_M = 10

    # Base index for orbitals of each layer:
    X_base_orbitals = [z * orbitals_per_X for z in range(1, 2*args.num_layers - 1)]
    M_base_orbitals = [args.num_layers * 2 * orbitals_per_X + z * orbitals_per_M for z in range(args.num_layers)]

    pz_up = [n for n in X_base_orbitals]
    pz_dn = [n + 1 for n in X_base_orbitals]
    dz2_up = [n for n in M_base_orbitals]
    dz2_dn = [n + 1 for n in M_base_orbitals]
    #print(pz_up)
    #print(pz_dn)
    #print(dz2_up)
    #print(dz2_dn)

    orbital_group = list(itertools.chain(pz_up, pz_dn, dz2_up, dz2_dn))

    num_top_bands = 2*args.num_layers

    weights = []
    for i in range(num_top_bands):
        weights.append([])

    for k in ks:
        print("k = ", k)
        Hk = Hk_recip(k, Hr)
        Es, U = np.linalg.eigh(Hk)

        top = top_valence_indices(E_F, num_top_bands, Es)
        print("top valence", top, [Es[t] for t in top])

        mirror_signs = [1, 1, -1, -1, 1, 1]

        for i, band in enumerate(top):
            state = U[:, band]

            mirror_eval = np.dot(state.conjugate().T, np.dot(M, state))[0, 0]
            print("band {}; mirror <v|M|v> = {}".format(band, mirror_eval))
            print("mirror deviation elements")
            Mdev = np.dot(M, state) - mirror_signs[i] * state
            for n, v in enumerate(Mdev):
                if abs(v)**2 > 1e-3:
                    print(n, basis[n], v, abs(v)**2)

            print("band, orb, orb_label, weight, evec comp")
            #for n, v in enumerate(state):
            #    if abs(v)**2 > 1e-2:
            #        print(band, n, basis[n], abs(v)**2, v)
            for n, v in enumerate(state):
                print(band, n, basis[n], abs(v)**2, v)

            total = 0

            for n in orbital_group:
                evec_comp = U[n, band].conjugate()
                total += abs(evec_comp)**2

            #print(total)
            weights[i].append(1 - total)

    #for i, band_weights in enumerate(weights):
    #    plt.plot(xs, band_weights, label="Band {}".format(i))
    plt.plot(xs, weights[0], 'b.', label="Band 0")
    plt.plot(xs, weights[1], 'g--', label="Band 1")

    plt.legend(loc=0)
    plt.xlabel("k / K", fontsize='large')
    plt.ylabel("Weight outside inner p_z, dz^2")
    plt.savefig("model_weights_Gamma.png", bbox_inches='tight', dpi=500)

if __name__ == "__main__":
    _main()
