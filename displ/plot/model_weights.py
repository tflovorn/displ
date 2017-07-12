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

    # Base index for orbitals of each layer:
    base_orbitals = [args.num_layers * orbitals_per_X + z * orbitals_per_M for z in range(args.num_layers)]

    # Orbitals contributing to j = +5/2 state:
    x2y2_up = [base + 6 for base in base_orbitals]
    xy_up = [base + 8 for base in base_orbitals]

    weights = []
    for z in range(args.num_layers):
        weights.append([])

    for k in ks:
        Hk = Hk_recip(k, Hr)
        Es, U = np.linalg.eigh(Hk)

        top = top_valence_indices(E_F, args.num_layers, Es)

        for z, (n1, n2) in enumerate(zip(x2y2_up, xy_up)):
            total = 0

            for i in top:
                evec_comp = (1/np.sqrt(2)) * (U[n1, i] - 1j*U[n2, i])
                total += abs(evec_comp)**2

            weights[z].append(1 - total)

    for z, layer_weights in enumerate(weights):
        plt.plot(xs, layer_weights, 'k-', label="Layer {}".format(z))

    plt.show()

if __name__ == "__main__":
    _main()
