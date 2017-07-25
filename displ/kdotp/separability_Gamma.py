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
from displ.kdotp.model_weights_K import vec_linspace, top_valence_indices
from displ.kdotp.separability_K import (density_matrix, expectation_normalized,
        get_total_orbitals, get_layer_projections)

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

    work = _get_work(args.subdir, args.prefix)
    wannier_dir = os.path.join(work, "wannier")
    scf_path = os.path.join(wannier_dir, "scf.out")

    E_F = fermi_from_scf(scf_path)

    Hr_path = os.path.join(wannier_dir, "{}_hr.dat".format(args.prefix))
    Hr = extractHr(Hr_path)

    Gamma = np.array([0.0, 0.0, 0.0])
    K = np.array([1/3, 1/3, 0.0])

    upto_factor = 0.3
    num_ks = 100

    ks = vec_linspace(Gamma, upto_factor*K, num_ks)
    xs = np.linspace(0.0, upto_factor, num_ks)

    assert(Hr[(0, 0, 0)][0].shape[0] == get_total_orbitals(args.num_layers))

    Pzs = get_layer_projections(args.num_layers)

    num_top_bands = 2*args.num_layers

    for k in ks:
        print("k = {}".format(k))

        Hk = Hk_recip(k, Hr)
        Es, U = np.linalg.eigh(Hk)

        top = top_valence_indices(E_F, num_top_bands, Es)
        print("top orbitals = ", top, "energies = ", [Es[t] for t in top])

        tb_states = []
        for restricted_index_n, band_n in enumerate(top):
            state_n = U[:, [band_n]]
            print("orbital", band_n)
            for i, v in enumerate(state_n[:, 0]):
                print(i, v)

            tb_states.append(state_n)

        dm = density_matrix(tb_states, [1]*len(tb_states))

        for z, Pz in enumerate(Pzs):
            print("z = {}".format(z))

            proj_dm = np.dot(Pz, np.dot(dm, Pz))

            proj_dm_evals, proj_dm_evecs = np.linalg.eigh(proj_dm)

            print("proj_dm_evals")
            print(proj_dm_evals)

            print("proj_dm_evecs")

            for i in range(proj_dm_evecs.shape[1]):
                for j in range(proj_dm_evecs.shape[0]):
                    print(j, proj_dm_evecs[j, i])

if __name__ == "__main__":
    _main()
