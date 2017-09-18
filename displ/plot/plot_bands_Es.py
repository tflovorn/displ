from __future__ import division
import argparse
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from displ.build.build import _get_work
from displ.pwscf.parseScf import latVecs_from_scf
from displ.wannier.extractHr import extractHr
from displ.wannier.bands import Hk

def get_ks(R, band_path_lat, Nk_per_panel):
    # Each panel includes the ending symmetry point, but not the beginning one.
    # Need to start with the first symmetry point so that it is included.
    ks_cart = [np.dot(band_path_lat[0], R)]
    dists = [0.0]

    # Construct each panel on the interval (sym_lat, next_sym_lat].
    for sym_lat, next_sym_lat in zip(band_path_lat, band_path_lat[1:]):
        sym_cart = np.dot(sym_lat, R)
        next_sym_cart = np.dot(next_sym_lat, R)

        panel_delta_k = next_sym_cart - sym_cart
        dists.append(np.linalg.norm(panel_delta_k))

        step_k = panel_delta_k / Nk_per_panel

        for i in range(Nk_per_panel):
            k = sym_cart + (i + 1) * step_k
            ks_cart.append(k)

    # Scale xs to correspond to the Cartesian distance travelled along the band path.
    total_dist = sum(dists)
    special_xs = [sum(dists[:i+1]) / total_dist for i in range(len(dists))]

    xs = [0.0]
    for special_x, next_special_x in zip(special_xs, special_xs[1:]):
        step_x = (next_special_x - special_x) / Nk_per_panel

        for i in range(Nk_per_panel):
            x = special_x + (i + 1) * step_x
            xs.append(x)

    return xs, special_xs, ks_cart

def transpose_lists(x_ijs):
    x_jis = []

    n_js = len(x_ijs[0])
    for j in range(n_js):
        x_jis.append([])

    for x_i in x_ijs:
        for j, x in enumerate(x_i):
            x_jis[j].append(x)

    return x_jis

def calculate_bands(Hr, latVecs, band_path_lat, Nk_per_panel):
    D = latVecs.T
    R = 2.0 * np.pi * np.linalg.inv(D)

    xs, special_xs, ks_cart = get_ks(R, band_path_lat, Nk_per_panel)

    Ekms = []
    for k_cart in ks_cart:
        this_Hk = Hk(k_cart, Hr, latVecs)
        Es = np.linalg.eigvalsh(this_Hk)
        Ekms.append(Es)

    Emks = transpose_lists(Ekms)

    return xs, special_xs, Emks

def _main():
    parser = argparse.ArgumentParser("Plot band structure for multiple electric field values",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("prefix_E_0", type=str,
            help="Prefix for E = 0 calculation")
    parser.add_argument("prefix_E_neq_0", type=str,
            help="Prefix for E != 0 calculation")
    parser.add_argument("--subdir", type=str, default=None,
            help="Subdirectory under work_base where calculation was run")
    parser.add_argument("--minE", type=float, default=None,
            help="Minimum energy to plot (using original energy zero)")
    parser.add_argument("--maxE", type=float, default=None,
            help="Maximum energy to plot (using original energy zero)")
    args = parser.parse_args()

    E_neq_0_val = args.prefix_E_neq_0.split('_')[-1]

    Gamma_lat = np.array([0.0, 0.0, 0.0])
    M_lat = np.array([1/2, 0.0, 0.0])
    K_lat = np.array([1/3, 1/3, 0.0])
    band_path_lat = [Gamma_lat, M_lat, K_lat, Gamma_lat]
    band_path_labels = ["$\\Gamma$", "$M$", "$K$", "$\\Gamma$"]
    Nk_per_panel = 400

    prefixes = [args.prefix_E_0, args.prefix_E_neq_0]
    labels = ["$E = 0$", "$E = {}$ V/nm".format(E_neq_0_val)]
    styles = ['r-', 'k--']

    xs, special_xs, Emks = [], [], []
    for prefix in prefixes:
        work = _get_work(args.subdir, prefix)
        wannier_dir = os.path.join(work, "wannier")
        scf_path = os.path.join(wannier_dir, "scf.out")

        latVecs = latVecs_from_scf(scf_path)

        Hr_path = os.path.join(wannier_dir, "{}_hr.dat".format(prefix))
        Hr = extractHr(Hr_path)

        this_xs, this_special_xs, this_Emks = calculate_bands(Hr, latVecs, band_path_lat, Nk_per_panel)

        xs.append(this_xs)
        special_xs.append(this_special_xs)
        Emks.append(this_Emks)

    # TODO - output data

    for this_xs, this_Emks, label, style in zip(xs, Emks, labels, styles):
        for m, Emk in enumerate(this_Emks):
            if m == 0:
                m_label = label
            else:
                m_label = '_' # exclude from legend

            plt.plot(this_xs, Emk, style, label=m_label)

    # TODO - change zero
    plt.ylim(args.minE, args.maxE)

    plt.xlim(0.0, 1.0)

    for special_x in special_xs[0]:
        plt.axvline(special_x, color='k', alpha=0.5, linestyle='--')

    plt.xticks(special_xs[0], band_path_labels)

    plt.legend(loc=0)

    plt.savefig("Efield_bands.png", bbox_inches='tight', dpi=500)

if __name__ == "__main__":
    _main()
