from __future__ import division
import argparse
import os
import json
from multiprocessing import Pool
import numpy as np
import yaml
import numdifftools
from displ.pwscf.parseScf import fermi_from_scf, D_from_scf, alat_from_scf
from displ.pwscf.extractQEBands import extractQEBands
from displ.wannier.bands import Hk_recip
from displ.wannier.bands import Hk as Hk_Cart
from displ.build.build import get_prefixes
from displ.build.util import _global_config
from displ.plot.shift_plot_ds import (get_atom_order, orbital_index, ds_from_prefixes,
        wrap_cell, sorted_d_group, plot_d_vals, get_Hr, filter_to_D)

def _close(k, q, eps):
    for i in range(len(k)):
        if abs(k[i] - q[i]) > eps:
            return False

    return True

def get_layer_indices(work, prefix, fixed_spin):
    atom_order = get_atom_order(work, prefix)

    syms = [["X1", "M", "X2"], ["X1p", "Mp", "X2p"]]
    orbitals = {"X1": ["pz", "px", "py"], "M": ["dz2", "dxz", "dyz", "dx2-y2", "dxy"]}
    orbitals["X2"] = orbitals["X1"]
    orbitals["X1p"] = orbitals["X1"]
    orbitals["X2p"] = orbitals["X1"]
    orbitals["Mp"] = orbitals["M"]

    spins = ["up", "down"]

    layer_indices = []
    for layer_syms in syms:
        layer_indices.append([])
        for sym in layer_syms:
            for orb in orbitals[sym]:
                for spin in spins:
                    if spin != fixed_spin:
                        continue

                    index = orbital_index(atom_order, sym, orb, spin, soc=True)
                    layer_indices[-1].append(index)

    return layer_indices

def get_layer_contribs(layer_indices, U):
    layer_contribs = [[], []]
    num_states = U.shape[0]
    for n in range(num_states):
        for l, l_indices in enumerate(layer_indices):
            contrib = 0.0
            for index in l_indices:
                contrib += abs(U[index, n])**2

            layer_contribs[l].append(contrib)

    return layer_contribs

def bracket_indices(w, E_F):
    for i, val in enumerate(w):
        if i == len(w) - 1:
            return None

        if val <= E_F and w[i+1] > E_F:
            return i, i+1

def select_layer_contrib(layer_contribs_up, layer_contribs_down, spin, l, n):
    contrib_up = layer_contribs_up[l][n]
    contrib_down = layer_contribs_down[l][n]

    if spin is None:
        contrib = contrib_up + contrib_down
    elif spin == 'up':
        contrib = contrib_up
    elif spin == 'down':
        contrib = contrib_down
    else:
        raise ValueError("unrecognized spin value")

    return contrib

def get_curvature(D, Hr, k, n):
    '''Calculate d^2 E / d k^2 along kx and ky directions at band n.
    Assumes there are no band crossings in the region sampled, so that
    the single index n can be used for all sampled ks.
    '''
    curvature = []
    for d in range(2):
        def Er_d(kd):
            kr = []
            for dp in range(3):
                if dp == d:
                    kr.append(kd)
                else:
                    kr.append(k[dp])

            H_kr = Hk_Cart(kr, Hr, D.T)
            w, U = np.linalg.eigh(H_kr)
            Er = w[n]
            return Er

        fd = numdifftools.Derivative(Er_d, n=2)
        curvature_d = fd(k[d])

        curvature.append(curvature_d)

    return curvature

def layer_band_extrema(Es, U, E_F, layer_indices_up, layer_indices_down, layer_threshold,
        spin_valence=None, spin_conduction=None):
    conduction = [None, None]
    valence = [None, None]

    layer_contribs_up = get_layer_contribs(layer_indices_up, U)
    layer_contribs_down = get_layer_contribs(layer_indices_down, U)

    below_fermi, above_fermi = bracket_indices(Es, E_F)

    n = below_fermi
    while n >= 0 and any([valence[l] is None for l in [0, 1]]):
        for l in [0, 1]:
            contrib = select_layer_contrib(layer_contribs_up, layer_contribs_down, spin_valence, l, n)

            if contrib > layer_threshold and valence[l] is None:
                valence[l] = n

        n -= 1

    n = above_fermi
    while n < len(Es) and any([conduction[l] is None for l in [0, 1]]):
        for l in [0, 1]:
            contrib = select_layer_contrib(layer_contribs_up, layer_contribs_down, spin_conduction, l, n)

            if contrib > layer_threshold and conduction[l] is None:
                conduction[l] = n

        n += 1

    if any([c is None for c in conduction]) or any([v is None for v in valence]):
        raise ValueError("failed to find all band extrema")

    return conduction, valence

def get_gaps(work, prefix, layer_threshold, k, spin_valence=None, spin_conduction=None,
        use_curvature=True):
    wannier_dir = os.path.join(work, prefix, "wannier")
    scf_path = os.path.join(wannier_dir, "scf.out")
    E_F = fermi_from_scf(scf_path)
    alat_Bohr = alat_from_scf(scf_path)
    D = D_from_scf(scf_path)
    R = 2*np.pi*np.linalg.inv(D)
    k_Cart = np.dot(np.array(k), R)

    layer_indices_up = get_layer_indices(work, prefix, 'up')
    layer_indices_down = get_layer_indices(work, prefix, 'down')

    Hr = get_Hr(work, prefix)
    # note:
    # rotated K 2pi/3: K_R2 = (-2/3, 1/3, 0.0)
    # rotated K 4pi/3: K_R4 = (1/3, -2/3, 0.0)
    Hk = Hk_recip(k, Hr)

    w, U = np.linalg.eigh(Hk)

    conduction, valence = layer_band_extrema(w, U, E_F, layer_indices_up, layer_indices_down,
            layer_threshold, spin_valence, spin_conduction)

    conduction_curvature = [None, None]
    valence_curvature = [None, None]

    for l in [0, 1]:
        valence_curvature[l] = get_curvature(D, Hr, k_Cart, valence[l])
        conduction_curvature[l] = get_curvature(D, Hr, k_Cart, conduction[l])

    gaps = {}
    layer_labels = ["0", "1"]
    # Generate gap for each layer pair. Keys have the form: "(0, 1)/(0, 1)".
    for cond_layer_index, cond_layer_label in enumerate(layer_labels):
        for valence_layer_index, valence_layer_label in enumerate(layer_labels):
            k = "{}/{}".format(cond_layer_label, valence_layer_label)
            gaps[k] = float(w[conduction[cond_layer_index]] - w[valence[valence_layer_index]])

    # Generate band extremum energy for each choice of layer and valence or conduction.
    vc_labels = ["valence", "conduction"]
    all_vc_data = [valence, conduction]
    for vc_label, vc_data in zip(vc_labels, all_vc_data):
        for layer_index, layer_label in enumerate(layer_labels):
            k = "{}_{}".format(layer_label, vc_label)
            gaps[k] = float(w[vc_data[layer_index]])

    # Minimum of the spin-orbit split partner of the lowest conduction band.
    # If layer gap is smaller than SO splitting, need to skip a band to find partner.
    conduction_min = min(conduction[0], conduction[1])
    if abs(conduction[0] - conduction[1]) == 1:
        conduction_min_partner = conduction_min + 2
    else:
        conduction_min_partner = conduction_min + 1

    gaps["conduction_min_partner"] = float(w[conduction_min_partner])

    if use_curvature:
        add_curvature(gaps, valence_curvature, conduction_curvature, alat_Bohr)

    return gaps

def reduced_mass(m1, m2):
    return m1 * m2 / (m1 + m2)

def add_curvature(gaps, valence_curvature, conduction_curvature, alat_Bohr):
    hbar_eV_s = 6.582119514e-16
    me_eV_per_c2 = 0.5109989461e6
    c_m_per_s = 2.99792458e8
    Bohr_m = 0.52917721067e-10
    unit_fac = hbar_eV_s**2 / (me_eV_per_c2 * (c_m_per_s)**(-2) * (Bohr_m)**2 * alat_Bohr**2)

    vc_labels = ["valence", "conduction"]
    all_curv_vals = [valence_curvature, conduction_curvature]
    vc_factors = [-1, 1]

    Cart_labels = ["kx", "ky"]
    layer_labels = ["0", "1"]

    # Generate effective mass fields for each choice of layer, valence or conduction,
    # and Cartesian direction. Keys have the form:
    # "(0, 1)_(valence, conduction)_effmass_(kx, ky)".
    for vc_label, vc_factor, curv_vals in zip(vc_labels, vc_factors, all_curv_vals):
        for Cart_index, Cart_label in enumerate(Cart_labels):
            for layer_index, layer_label in enumerate(layer_labels):
                k = "{}_{}_effmass_{}".format(layer_label, vc_label, Cart_label)
                gaps[k] = float(vc_factor * unit_fac / curv_vals[layer_index][Cart_index])

    rvc_labels = ["reduced"]
    rvc_labels.extend(vc_labels)

    # Generate effective mass fields for each choice of layer and Cartesian direction.
    # Keys have the form: "(0, 1)_reduced_effmass_(kx, ky)".
    for layer_label in layer_labels:
        for Cart_label in Cart_labels:
            ks = {rvc: "{}_{}_effmass_{}".format(layer_label, rvc, Cart_label) for rvc in rvc_labels}
            gaps[ks["reduced"]] = reduced_mass(gaps[ks["valence"]], gaps[ks["conduction"]])

def get_gap_data(work, dps, threshold, spin_valence, spin_conduction, k, gap_label, use_curvature=True):
    get_gaps_args = []
    for d, prefix in dps:
        get_gaps_args.append([work, prefix, threshold, k, spin_valence, spin_conduction, use_curvature])

    with Pool() as p:
        all_gaps = p.starmap(get_gaps, get_gaps_args)

    gap_data = []
    # For JSON output, use same format as plot_ds.
    json_gap_data = {"_ds": []}

    for (d, prefix), gaps in zip(dps, all_gaps):
        gap_data.append([list(d), gaps])

        json_gap_data["_ds"].append(d)
        for k, v in gaps.items():
            if k not in json_gap_data:
                json_gap_data[k] = []

            json_gap_data[k].append(v)

    with open("{}_gap_data.json".format(gap_label), 'w') as fp:
        json.dump(json_gap_data, fp)

    return gap_data

def plot_gap_data(gap_data, dps, gap_label, gap_label_tex, layer_labels, use_curvature=True):
    # Band extremum energy (keys, filenames, plot labels).
    ds_data_keys = ["0/0", "1/1", "0/1", "1/0", "0_valence", "1_valence", "0_conduction",
            "1_conduction", "conduction_min_partner"]

    out_filenames = ["{}_layer0_gaps", "{}_layer1_gaps", "{}_interlayer_01_gaps",
            "{}_interlayer_10_gaps", "{}_layer0_valence", "{}_layer1_valence",
            "{}_layer0_conduction", "{}_layer1_conduction", "{}_conduction_min_partner"]
    out_filenames = list(map(lambda s: s.format(gap_label), out_filenames))

    layer0_label, layer1_label = layer_labels

    plot_labels = ["{} {} gap [eV]".format(gap_label_tex, layer0_label),
            "{} {} gap [eV]".format(gap_label_tex, layer1_label),
            "{} {} - {} gap [eV]".format(gap_label_tex, layer0_label, layer1_label),
            "{} {} - {} gap [eV]".format(gap_label_tex, layer1_label, layer0_label),
            "{} {} valence maximum [eV]".format(gap_label_tex, layer0_label),
            "{} {} valence maximum [eV]".format(gap_label_tex, layer1_label),
            "{} {} conduction minimum [eV]".format(gap_label_tex, layer0_label),
            "{} {} conduction minimum [eV]".format(gap_label_tex, layer1_label),
            "{} conduction minimum partner [eV]".format(gap_label_tex)]

    # Effective mass (keys, filenames, plot labels).
    if use_curvature:
        vcr_labels = ["valence", "conduction", "reduced"]
        vcr_tex_labels = ["valence", "conduction", ""]
        vcr_mass_labels = ["m", "m", "\\mu"]

        for layer_label, layer_label_tex in zip(["0", "1"], [layer0_label, layer1_label]):
            for vcr_label, vcr_label_tex, vcr_mass_label in zip(vcr_labels, vcr_tex_labels, vcr_mass_labels):
                for kxy_label, xy_label in zip(["kx", "ky"], ["x", "y"]):
                    ds_data_keys.append("{}_{}_effmass_{}".format(layer_label, vcr_label, kxy_label))
                    out_filenames.append("{}_layer{}_{}_effmass_{}".format(gap_label, layer_label, vcr_label, kxy_label))
                    plot_labels.append("{} {} {} ${}^*_{}/m_e$".format(gap_label_tex, layer_label_tex, vcr_label_tex, vcr_mass_label, xy_label))

    # Collect and plot data.
    ds_band_data = {}
    for d, gaps in gap_data:
        for k in ds_data_keys:
            if k in ds_band_data:
                ds_band_data[k].append(gaps[k])
            else:
                ds_band_data[k] = [gaps[k]]

    for k, fname, plot_label in zip(ds_data_keys, out_filenames, plot_labels):
        plot_d_vals(fname, plot_label, dps, ds_band_data[k])

def _main():
    parser = argparse.ArgumentParser("Calculation of gaps",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--subdir", type=str, default=None,
            help="Subdirectory under work_base where calculation was run")
    parser.add_argument("--threshold_K", type=float, default=0.9,
            help="Threshold for deciding if a state is dominated by one layer")
    parser.add_argument("--threshold_Gamma", type=float, default=0.6,
            help="Threshold for deciding if a state is dominated by one layer")
    parser.add_argument("--spin_valence", type=str, default=None,
            help="Set 'up' or 'down' to choose valence band spin type; closest to E_F is used if not set")
    parser.add_argument("--spin_conduction", type=str, default=None,
            help="Set 'up' or 'down' to choose conduction band spin type; closest to E_F is used if not set")
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

    gap_data_K = get_gap_data(work, dps, args.threshold_K, args.spin_valence, args.spin_conduction,
            K, "K")
    plot_gap_data(gap_data_K, dps, "K", "$K$", layer_labels)

    Gamma = (0.0, 0.0, 0.0)

    gap_data_Gamma = get_gap_data(work, dps, args.threshold_Gamma, args.spin_valence, args.spin_conduction,
            Gamma, "Gamma", use_curvature=False)
    plot_gap_data(gap_data_Gamma, dps, "Gamma", r"$\Gamma$", layer_labels, use_curvature=False)

if __name__ == "__main__":
    _main()
