from __future__ import division
import argparse
import os
import json
from multiprocessing import Pool
import numpy as np
import yaml
from displ.pwscf.parseScf import fermi_from_scf, D_from_scf, alat_from_scf
from displ.pwscf.extractQEBands import extractQEBands
from displ.wannier.bands import Hk_recip
from displ.wannier.bands import Hk as Hk_Cart
from displ.build.build import get_prefixes
from displ.build.util import _global_config
from displ.plot.shift_plot_ds import (get_atom_order, orbital_index, ds_from_prefixes,
        wrap_cell, sorted_d_group, plot_d_vals, get_Hr, filter_to_D)
from displ.plot.shift_gap import (get_relax_data, plot_relax_data, get_layer_indices,
        bracket_indices, get_layer_contribs, select_layer_contrib)

def get_extrema(work, prefix, k, k_prefix, num_layers, num_valence_bands, num_conduction_bands):
    wannier_dir = os.path.join(work, prefix, "wannier")
    scf_path = os.path.join(wannier_dir, "scf.out")
    E_F = fermi_from_scf(scf_path)

    Hr = get_Hr(work, prefix)
    Hk = Hk_recip(k, Hr)

    Es, U = np.linalg.eigh(Hk)
    below_fermi, above_fermi = bracket_indices(Es, E_F)

    if num_layers != 2:
        raise ValueError("get_layer_indices() unimplemented for num_layers != 2")

    layer_indices_up = get_layer_indices(work, prefix, 'up')
    layer_indices_down = get_layer_indices(work, prefix, 'down')
    layer_contribs_up = get_layer_contribs(layer_indices_up, U)
    layer_contribs_down = get_layer_contribs(layer_indices_down, U)

    extrema = {}

    def add_extrema_entries(i, band_index, vc_label):
        extrema["energy_{}_{}".format(vc_label, i)] = Es[band_index]

        for l in range(num_layers):
            layer_contrib = select_layer_contrib(layer_contribs_up, layer_contribs_down, None,
                    l, band_index)

            extrema["contrib_layer_{}_{}_{}".format(l, vc_label, i)] = layer_contrib

        for spin in ['up', 'down']:
            spin_contrib = sum([select_layer_contrib(layer_contribs_up, layer_contribs_down, spin,
                    l, band_index) for l in range(num_layers)])

            extrema["contrib_spin_{}_{}_{}".format(spin, vc_label, i)] = spin_contrib

    for i, band_index in enumerate(range(below_fermi, below_fermi - num_valence_bands, -1)):
        add_extrema_entries(i, band_index, 'valence')

    for i, band_index in enumerate(range(above_fermi, above_fermi + num_conduction_bands)):
        add_extrema_entries(i, band_index, 'conduction')

    return extrema

def get_extrema_data(work, dps, k, k_prefix, num_layers, num_valence_bands, num_conduction_bands):
    get_extrema_args = []
    for d, prefix in dps:
        get_extrema_args.append([work, prefix, k, k_prefix, num_layers, num_valence_bands, num_conduction_bands])

    with Pool() as p:
        all_extrema = p.starmap(get_extrema, get_extrema_args)

    extrema_data = []
    # For JSON output, use same format as plot_ds.
    json_extrema_data = {"_ds": []}

    for (d, prefix), extrema in zip(dps, all_extrema):
        extrema_data.append([list(d), extrema])

        json_extrema_data["_ds"].append(d)
        for k, v in extrema.items():
            if k not in json_extrema_data:
                json_extrema_data[k] = []

            json_extrema_data[k].append(v)

    with open("{}_extrema_data.json".format(k_prefix), 'w') as fp:
        json.dump(json_extrema_data, fp)

    return extrema_data

def plot_extrema_data(extrema_data, dps, k_prefix, k_label, layer_labels):
    ds_extrema_data = {}
    for d, extrema in extrema_data:
        for k, v in extrema.items():
            if k in ds_extrema_data:
                ds_extrema_data[k].append(v)
            else:
                ds_extrema_data[k] = [v]

    for k, v in ds_extrema_data.items():
        fname = "{}_{}_plot".format(k_prefix, k)
        plot_label = "{} {}".format(k_prefix, k)

        plot_d_vals(fname, plot_label, dps, v)

def _main():
    parser = argparse.ArgumentParser("Report band extrema and their layer / spin weights",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--subdir", type=str, default=None,
            help="Subdirectory under work_base where calculation was run")
    parser.add_argument("--num_layers", type=int, default=2,
            help="Number of layers in system (determines number of band extrema to consider)")
    parser.add_argument("--D", type=float, default=None,
            help="Perpendicular electric field value [V/nm]")
    parser.add_argument('global_prefix', type=str,
            help="Calculation name")
    args = parser.parse_args()

    gconf = _global_config()
    work = os.path.expandvars(gconf["work_base"])
    if args.subdir is not None:
        work = os.path.join(work, args.subdir)

    prefixes = get_prefixes(work, args.global_prefix)
    if args.D is not None:
        prefixes = filter_to_D(prefixes, args.D)

    ds = ds_from_prefixes(prefixes)
    ds, prefixes = wrap_cell(ds, prefixes)
    dps = sorted_d_group(ds, prefixes)

    K = (1/3, 1/3, 0.0)
    layer_labels = ["bot.", "top"]

    relax_data = get_relax_data(work, dps)
    plot_relax_data(relax_data, dps)

    K_num_valence_bands = args.num_layers
    K_num_conduction_bands = 2 * args.num_layers

    Gamma_num_valence_bands = 2
    Gamma_num_conduction_bands = 4

    extrema_data_K = get_extrema_data(work, dps, K, "K", args.num_layers, K_num_valence_bands, K_num_conduction_bands)
    plot_extrema_data(extrema_data_K, dps, "K", "$K$", layer_labels)

    Gamma = (0.0, 0.0, 0.0)

    extrema_data_Gamma = get_extrema_data(work, dps, Gamma, "Gamma", args.num_layers, Gamma_num_valence_bands, Gamma_num_conduction_bands)
    plot_extrema_data(extrema_data_Gamma, dps, "Gamma", r"$\Gamma$", layer_labels)

if __name__ == "__main__":
    _main()
