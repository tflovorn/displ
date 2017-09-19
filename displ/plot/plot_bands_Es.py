from __future__ import division
import argparse
import os
import json
import csv
import numpy as np
import matplotlib.pyplot as plt
from displ.build.build import _get_work
from displ.pwscf.parseScf import latVecs_from_scf, fermi_from_scf
from displ.wannier.extractHr import extractHr
from displ.wannier.bands import Hk, Hk_recip
from displ.kdotp.model_weights_K import top_valence_indices

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

def get_E_Gamma(Hr, E_F):
    Gamma_lat = np.array([0.0, 0.0, 0.0])
    this_Hk = Hk_recip(Gamma_lat, Hr)
    Es = np.linalg.eigvalsh(this_Hk)

    top_Gamma_index = top_valence_indices(E_F, 1, Es)[0]
    return Es[top_Gamma_index]

def shift_Emks(Emks, E_base):
    new_Emks = []
    for Em in Emks:
        new_Emks.append([])
        for E in Em:
            new_Emks[-1].append(E - E_base)

    return new_Emks

def write_bands_csv(xs, Emks, fpath):
    num_bands = len(Emks)
    row_keys = ["x"]
    for m in range(num_bands):
        row_keys.append("m_{}".format(m))

    Ekms = transpose_lists(Emks)

    with open(fpath, 'w') as fp:
        writer = csv.writer(fp)
        writer.writerow(row_keys)

        for x, Ek in zip(xs, Ekms):
            row = [x]
            for E in Ek:
                row.append(E)

            writer.writerow(row)

def write_bands_aux(special_xs, band_path_label, fpath):
    row_keys = ["special_x_value", "special_x_label"]

    with open(fpath, 'w') as fp:
        writer = csv.writer(fp)
        writer.writerow(row_keys)

        for x, label in zip(special_xs, band_path_label):
            writer.writerow([x, label])

def plot_bands(xs, special_xs, band_path_labels, Emks, labels, styles, xlim, ylim, out_prefix):
    for this_xs, this_Emks, label, style in zip(xs, Emks, labels, styles):
        for m, Emk in enumerate(this_Emks):
            if m == 0:
                m_label = label
            else:
                m_label = '_' # exclude from legend

            plt.plot(this_xs, Emk, style, label=m_label)

    plt.ylabel("$E - E_{\\Gamma}^0$ [eV]", fontsize='large')
    plt.ylim(ylim[0], ylim[1])

    plt.xlim(xlim[0], xlim[1])

    plt.axhline(0.0, color='k', alpha=0.25, linestyle='-')

    for special_x in special_xs:
        plt.axvline(special_x, color='k', alpha=0.25, linestyle='-')

    plt.xticks(special_xs, band_path_labels, fontsize='large')

    plt.legend(loc=0)

    plt.savefig("{}_Efield_bands.png".format(out_prefix), bbox_inches='tight', dpi=500)
    plt.clf()

def _main():
    parser = argparse.ArgumentParser("Plot band structure for multiple electric field values",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("prefix_E_0", type=str,
            help="Prefix for E = 0 calculation")
    parser.add_argument("prefix_E_mid", type=str,
            help="Prefix for E != 0 calculation, middle value")
    parser.add_argument("prefix_E_high", type=str,
            help="Prefix for E != 0 calculation, high value")
    parser.add_argument("--subdir", type=str, default=None,
            help="Subdirectory under work_base where calculation was run")
    parser.add_argument("--minE", type=float, default=1.85,
            help="Minimum energy to plot (using original energy zero)")
    parser.add_argument("--maxE", type=float, default=5.75,
            help="Maximum energy to plot (using original energy zero)")
    parser.add_argument("--load", action='store_true',
            help="Load stored band data instead of recalculating")
    args = parser.parse_args()

    E_mid_val = args.prefix_E_mid.split('_')[-1]
    E_high_val = args.prefix_E_high.split('_')[-1]

    Gamma_lat = np.array([0.0, 0.0, 0.0])
    M_lat = np.array([1/2, 0.0, 0.0])
    K_lat = np.array([1/3, 1/3, 0.0])
    band_path_lat = [Gamma_lat, M_lat, K_lat, Gamma_lat]
    band_path_labels = ["$\\Gamma$", "$M$", "$K$", "$\\Gamma$"]
    Nk_per_panel = 400

    prefixes = [args.prefix_E_0, args.prefix_E_mid, args.prefix_E_high]
    labels = ["$E = 0$", "$E = {}$ V/nm".format(E_mid_val), "$E = {}$ V/nm".format(E_high_val)]
    styles = ['r-', 'b-.', 'k--']


    xs, special_xs, Emks = [], [], []
    if args.load:
        with open("Efield_bands.json", 'r') as fp:
            in_data = json.load(fp)

        xs, special_xs, Emks = list(map(lambda k: [d[k] for d in in_data],
                ["xs", "special_xs", "Emks"]))

        E_Gamma_Eperp_0 = in_data[0]["E_Gamma_Eperp_0"]
    else:
        out_data = []

        for prefix_index, prefix in enumerate(prefixes):
            work = _get_work(args.subdir, prefix)
            wannier_dir = os.path.join(work, "wannier")
            scf_path = os.path.join(wannier_dir, "scf.out")

            latVecs = latVecs_from_scf(scf_path)

            Hr_path = os.path.join(wannier_dir, "{}_hr.dat".format(prefix))
            Hr = extractHr(Hr_path)

            if prefix_index == 0:
                E_F = fermi_from_scf(scf_path)
                E_Gamma_Eperp_0 = get_E_Gamma(Hr, E_F)

            this_xs, this_special_xs, this_Emks = calculate_bands(Hr, latVecs, band_path_lat, Nk_per_panel)

            this_Emks = shift_Emks(this_Emks, E_Gamma_Eperp_0)

            xs.append(this_xs)
            special_xs.append(this_special_xs)
            Emks.append(this_Emks)

            out_data.append({"prefix": prefix, "xs": this_xs, "special_xs": this_special_xs,
                    "band_path_labels": band_path_labels, "Emks": this_Emks,
                    "E_Gamma_Eperp_0": E_Gamma_Eperp_0})

        with open("Efield_bands.json", 'w') as fp:
            json.dump(out_data, fp)

    full_min_x, full_max_x = 0.0, 1.0
    full_minE, full_maxE = args.minE - E_Gamma_Eperp_0, args.maxE - E_Gamma_Eperp_0

    print("full_minE = ", full_minE)
    print("full_maxE = ", full_maxE)

    for this_xs, this_Emks, path_suffix in zip(xs, Emks, ["0.0", "0.5", "1.0"]):
        out_path = "Efield_bands_E_{}.csv".format(path_suffix)
        write_bands_csv(this_xs, this_Emks, out_path)

    write_bands_aux(special_xs[0], band_path_labels, "Efield_bands_aux.csv")

    plot_bands([xs[0], xs[-1]], special_xs[0], band_path_labels, [Emks[0], Emks[-1]],
            [labels[0], labels[-1]], [styles[0], styles[-1]], [full_min_x, full_max_x],
            [full_minE, full_maxE], "full")

    x_K = special_xs[0][2]
    zoom_xlim = [x_K - 0.05, x_K + 0.05]
    zoom_special_xs = [zoom_xlim[0], x_K, zoom_xlim[-1]]
    zoom_band_path_labels = ["$\\leftarrow M$", "$K$", "$\\rightarrow \\Gamma$"]
    zoom_ylim = [-0.3, 0.1]

    print("zoom_xlim = ", zoom_xlim)
    print("zoom_ylim = ", zoom_ylim)

    plot_bands(xs, zoom_special_xs, zoom_band_path_labels, Emks,
            labels, styles, zoom_xlim, zoom_ylim, "zoom")

if __name__ == "__main__":
    _main()
