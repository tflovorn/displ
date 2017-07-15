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
from displ.plot.model_weights_K import vec_linspace, top_valence_indices

def density_matrix(states, probabilities):
    eps_abs, eps_rel = 1e-12, 1e-12
    assert(near_equal(sum(probabilities), 1.0, eps_abs, eps_rel))

    dimension = states[0].shape[0] # assume states are row vectors
    dm = np.zeros([dimension, dimension], dtype=np.complex128)
    for v, p in zip(states, probabilities):
        for ip in range(dimension):
            for i in range(dimension):
                # dm = \sum_v p_v |v> <v|
                # --> <ip|dm|i> = \sum_v p_v <ip|v> <v|i> = \sum_v p_v v_ip v_i^*
                dm[ip, i] += p * v[ip, 0] * v[i, 0].conjugate()

    assert_diagonal_real(dm, eps_abs)
    assert(near_equal(np.trace(dm), 1.0, eps_abs, eps_rel))

    return dm

def assert_diagonal_real(M, eps_abs):
    assert(M.shape[0] == M.shape[1])
    dimension = M.shape[0]
    for i in range(dimension):
        assert(near_zero(M[i, i].imag, eps_abs))

def near_zero(x, eps_abs):
    return abs(x) < eps_abs

def near_equal(x, y, eps_abs, eps_rel):
    if abs(x - y) < eps_abs:
        return True
    else:
        return abs(x - y) < eps_rel * max(abs(x), abs(y))

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

    # Assume SOC present and that model has 2*3 X(p) orbitals per layer
    # and 2*5 M(d) in canonical Wannier90 order.
    # Assumes atoms are ordered with all Xs first, then all Ms, and within
    # M/X groups the atoms are in layer order.
    orbitals_per_X = 6
    orbitals_per_M = 10

    total_orbitals = args.num_layers * 2 * orbitals_per_X + args.num_layers * orbitals_per_M
    assert(Hr[(0, 0, 0)][0].shape[0] == total_orbitals)

    M_base = args.num_layers * 2 * orbitals_per_X

    layer_orbitals = [list(itertools.chain(range(z * 2 * orbitals_per_X, (z + 1) * 2 * orbitals_per_X),
            range(M_base + z * orbitals_per_M, M_base + (z + 1) * orbitals_per_M)))
            for z in range(args.num_layers)]

    # For each layer, make a projection onto all orbitals in that layer.
    Pzs = []
    for z_basis_elements in layer_orbitals:
        print(z_basis_elements)
        Pz = np.zeros([total_orbitals, total_orbitals], dtype=np.complex128)
        
        for i in z_basis_elements:
            Pz[i, i] = 1

        Pzs.append(Pz)

    # Consider only orthogonal projections: projection matrices should be Hermitian.
    for Pz in Pzs:
        assert((Pz == Pz.T.conjugate()).all())

    num_top_bands = args.num_layers

    deviations = []
    for z in range(len(Pzs)):
        deviations.append([])

    for k in ks:
        print("k = {}".format(k))
        for z, Pz in enumerate(Pzs):
            Hk = Hk_recip(k, Hr)
            Es, U = np.linalg.eigh(Hk)

            top = top_valence_indices(E_F, num_top_bands, Es)
            proj_dms = [] # "dm" == density matrix

            for restricted_index_n, band_n in enumerate(top):
                state_n = U[:, [band_n]]
                dm_n = density_matrix([state_n], [1])

                proj_dm = np.dot(Pz, np.dot(dm_n, Pz))
                proj_dms.append(proj_dm)

            proj_overlap = np.zeros([num_top_bands, num_top_bands], dtype=np.complex128)

            for restricted_index_np in range(len(top)):
                for restricted_index_n in range(len(top)):
                    dm_np, dm_n = proj_dms[restricted_index_np], proj_dms[restricted_index_n]
                    dm_np_norm = dm_np / np.trace(dm_np)
                    dm_n_norm = dm_n / np.trace(dm_n)

                    print("np, n, Tr rho_np^z, Tr rho_n^z", restricted_index_np, restricted_index_n, np.trace(dm_np), np.trace(dm_n), np.trace(dm_np_norm), np.trace(dm_n_norm))

                    threshold = 1e-4
                    if np.trace(dm_np) < threshold or np.trace(dm_n) < threshold:
                        proj_overlap[restricted_index_np, restricted_index_n] = 1
                    else:
                        proj_overlap[restricted_index_np, restricted_index_n] = np.trace(np.dot(dm_np_norm, dm_n_norm))

            print("z = {}".format(z))
            print("overlap = ")
            print(proj_overlap)

            # Compute deviation from desired 'all ones' value of proj_overlap.
            deviation = sum([abs(1 - v) for v in np.nditer(proj_overlap)])

            print("sum of abs(deviation) = {}".format(deviation))

            deviations[z].append(deviation)

    for z in range(len(Pzs)):
        plt.plot(xs, deviations[z], label="deviations for layer {}".format(z))

    plt.legend(loc=0)
    plt.savefig("{}_separability_K.png".format(args.prefix), bbox_inches='tight', dpi=500)

if __name__ == "__main__":
    _main()
