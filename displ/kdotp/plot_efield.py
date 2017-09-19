from __future__ import division
import argparse
from collections import OrderedDict
import json
import csv
import matplotlib.pyplot as plt
from displ.kdotp.efield import decimal_format

def check_consistency(data, epsilon_r, expected_hole_density):
    if epsilon_r is None:
        epsilon_r = data["epsilon_r"]
    else:
        assert(epsilon_r == data["epsilon_r"])

    if expected_hole_density is not None:
        assert(expected_hole_density == data["hole_density_cm2"])

    return epsilon_r

def plot_ediffs(results):
    keys = ["dft_no_holes", "kdotp_no_holes", "dft_screened", "kdotp_screened"]
    styles = ['rs', 'r--', 'ko', 'k-']
    Ediff_labels = ["$p$ = 0 ($E$ by DFT)", "$p$ = 0 ($E$ by $k \cdot p$)", "$p$ = {} ($E$ by DFT)", "$p$ = {} ($E$ by $k \cdot p$)"]
    expected_hole_densities = [0.0, 0.0, None, None]
    epsilon_r = None

    for k, expected_hole_density, style, label in zip(keys, expected_hole_densities, styles, Ediff_labels):
        epsilon_r = check_consistency(results[k], epsilon_r, expected_hole_density)

        E_V_nms = results[k]["E_V_nms"]
        E_Gamma_Ks = results[k]["E_Gamma_Ks"]

        if results[k]["hole_density_cm2"] != 0.0:
            label = label.format(decimal_format(results[k]["hole_density_cm2"], 0) + " cm$^{-2}$")

        plt.plot(E_V_nms, E_Gamma_Ks, style, label=label)

    plt.xlabel("$E$ [V/nm]", fontsize='large')
    plt.ylabel("$E_{\\Gamma} - E_K$ [eV]", fontsize='large')
    plt.legend(loc=0)
    plt.axhline(0.0, color='k', linestyle='-', alpha=0.25)

    plt.savefig("collected_Ediffs_Efield.png", bbox_inches='tight', dpi=500)
    plt.clf()

def write_csv_from_dict(table, row_keys, out_path):
    with open(out_path, 'w') as fp:
        fieldnames = ['E_V_nm']
        fieldnames.extend(row_keys)

        writer = csv.DictWriter(fp, fieldnames=fieldnames)

        writer.writeheader()
        for k, v in table.items():
            row = {'E_V_nm': k}
            for kr, vr in v.items():
                row[kr] = vr

            writer.writerow(row)

def write_ediffs_csv(results):
    table_unsorted = {}

    result_keys = ["dft_no_holes", "kdotp_no_holes", "dft_screened", "kdotp_screened"]
    row_keys = ["Ediff_p_0_DFT", "Ediff_p_0_kdotp", "Ediff_p_gt_0_DFT", "Ediff_p_gt_0_kdotp"]

    for result_key, row_key in zip(result_keys, row_keys):
        E_V_nms = results[result_key]["E_V_nms"]
        E_Gamma_Ks = results[result_key]["E_Gamma_Ks"]

        for E_V_nm, E_Gamma_K in zip(E_V_nms, E_Gamma_Ks):
            if E_V_nm in table_unsorted:
                table_unsorted[E_V_nm][row_key] = E_Gamma_K
            else:
                table_unsorted[E_V_nm] = {row_key: E_Gamma_K}

    table = OrderedDict(sorted(table_unsorted.items(), key=lambda kv: kv[0]))

    write_csv_from_dict(table, row_keys, "collected_Ediffs_Efield.csv")

def plot_occupations(results):
    result_keys = ["dft_screened", "kdotp_screened", "dft_screened", "kdotp_screened"]
    data_keys = ["nh_Gammas_frac", "nh_Gammas_frac", "nh_Ks_frac", "nh_Ks_frac"]
    styles = ['ks', 'k-', 'bo', 'b--']
    labels = ["$\\Gamma$ ($E$ by DFT)", "$\\Gamma$ ($E$ by $k \cdot p$)", "$K$ ($E$ by DFT)", "$K$ ($E$ by $k \cdot p$)"]

    for rk, dk, style, label in zip(result_keys, data_keys, styles, labels):
        E_V_nms = results[rk]["E_V_nms"]
        ys = results[rk][dk]
        hole_density_cm2 = results[rk]["hole_density_cm2"]

        plt.plot(E_V_nms, ys, style, label=label)

    plt.xlabel("$E$ [V/nm]", fontsize='large')
    plt.ylabel("Occupation fraction", fontsize='large')
    plt.ylim(0.0, 1.0)
    plt.legend(loc=0, title="$p$ = {}".format(decimal_format(hole_density_cm2, 0) + " cm$^{-2}$"))

    plt.savefig("collected_occupations_Efield.png", bbox_inches='tight', dpi=500)
    plt.clf()

def write_occupations_csv(results):
    table_unsorted = {}

    result_keys = ["dft_screened", "kdotp_screened", "dft_screened", "kdotp_screened"]
    data_keys = ["nh_Gammas_frac", "nh_Gammas_frac", "nh_Ks_frac", "nh_Ks_frac"]
    row_keys = ["occ_Gamma_DFT", "occ_Gamma_kdotp", "occ_K_DFT", "occ_K_kdotp"]

    for result_key, data_key, row_key in zip(result_keys, data_keys, row_keys):
        E_V_nms = results[result_key]["E_V_nms"]
        occs = results[result_key][data_key]

        for E_V_nm, occ in zip(E_V_nms, occs):
            if E_V_nm in table_unsorted:
                table_unsorted[E_V_nm][row_key] = occ
            else:
                table_unsorted[E_V_nm] = {row_key: occ}

    table = OrderedDict(sorted(table_unsorted.items(), key=lambda kv: kv[0]))

    write_csv_from_dict(table, row_keys, "collected_occupations_Efield.csv")

def _main():
    parser = argparse.ArgumentParser("Plot TMD multilayer response to electric field",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("dft_no_holes", type=str,
            help="Data file path for (E_perp from DFT, no holes) result")
    parser.add_argument("kdotp_no_holes", type=str,
            help="Data file path for (E_perp from k dot p model, no holes) result")
    parser.add_argument("dft_screened", type=str,
            help="Data file path for (E_perp from DFT, holes, screened) result")
    parser.add_argument("kdotp_screened", type=str,
            help="Data file path for (E_perp from k dot p model, holes, screened) result")
    args = parser.parse_args()

    results = {}
    keys = ["dft_no_holes", "kdotp_no_holes", "dft_screened", "kdotp_screened"]
    for k in keys:
        path = getattr(args, k)
        with open(path, 'r') as fp:
            k_data = json.load(fp)
            results[k] = k_data

    write_ediffs_csv(results)
    write_occupations_csv(results)

    plot_ediffs(results)
    plot_occupations(results)

if __name__ == "__main__":
    _main()
