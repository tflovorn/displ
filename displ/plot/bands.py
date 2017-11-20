from __future__ import division
import argparse
import os
from displ.build.build import _get_work, band_path_labels
from displ.pwscf.extractQEBands import extractQEBands
from displ.pwscf.parseScf import fermi_from_scf, alat_from_scf, latVecs_from_scf
from displ.wannier.extractHr import extractHr
from displ.wannier.plotBands import plotBands

def make_comp_groups(orbitals_per_X, orbitals_per_M, num_layers):
    # Assume orbital basis is arranged such that all X's come before all M's,
    # and X's and M's are individually sorted by z position.
    Xs_per_layer = orbitals_per_X * 2
    Ms_per_layer = orbitals_per_M

    groups = []
    for layer in range(num_layers):
        X_base = layer*Xs_per_layer
        layer_orbitals = list(range(X_base, X_base + Xs_per_layer))

        M_base = num_layers*Xs_per_layer + layer*Ms_per_layer
        layer_orbitals.extend(list(range(M_base, M_base + Ms_per_layer)))

        groups.append(layer_orbitals)

    return groups

def _main():
    parser = argparse.ArgumentParser("Plot band structure",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("prefix", type=str,
            help="Prefix for calculation")
    parser.add_argument("--subdir", type=str, default=None,
            help="Subdirectory under work_base where calculation was run")
    parser.add_argument("--DFT_only", action='store_true',
            help="Use DFT bands only (no Wannier)")
    parser.add_argument("--fermi_shift", action='store_true',
            help="Shift plotted energies so that E_F = 0")
    parser.add_argument("--band_extrema", action='store_true',
            help="Add lines at VBM/CBM")
    parser.add_argument("--minE", type=float, default=None,
            help="Minimum energy to plot")
    parser.add_argument("--maxE", type=float, default=None,
            help="Maximum energy to plot")
    parser.add_argument("--plot_evecs", action='store_true',
            help="Plot eigenvector decomposition")
    parser.add_argument("--num_layers", type=int, default=3,
            help="Number of layers (required if group_layer_* options given)")
    parser.add_argument("--group_layer_evecs_soc", action='store_true',
            help="Group eigenvectors weights by layer, assuming SOC")
    parser.add_argument("--group_layer_evecs_no_soc", action='store_true',
            help="Group eigenvectors weights by layer, assuming no SOC")
    parser.add_argument("--show", action='store_true',
            help="Show band plot instead of outputting file")
    args = parser.parse_args()

    work = _get_work(args.subdir, args.prefix)
    wannier_dir = os.path.join(work, "wannier")
    scf_path = os.path.join(wannier_dir, "scf.out")

    E_F = fermi_from_scf(scf_path)
    if args.minE is not None and args.maxE is not None:
        if args.fermi_shift:
            minE_plot = E_F + args.minE
            maxE_plot = E_F + args.maxE
        else:
            minE_plot = args.minE
            maxE_plot = args.maxE
    else:
        minE_plot, maxE_plot = None, None

    alat = alat_from_scf(scf_path)
    latVecs = latVecs_from_scf(scf_path)

    if args.DFT_only:
        Hr = None
    else:
        Hr_path = os.path.join(wannier_dir, "{}_hr.dat".format(args.prefix))
        Hr = extractHr(Hr_path)

    bands_dir = os.path.join(work, "bands")
    bands_path = os.path.join(bands_dir, "{}_bands.dat".format(args.prefix))

    num_bands, num_ks, qe_bands = extractQEBands(bands_path)
    outpath = args.prefix

    if args.group_layer_evecs_soc:
        orbitals_per_X = 6
        orbitals_per_M = 10
        comp_groups = make_comp_groups(orbitals_per_X, orbitals_per_M, args.num_layers)
    elif args.group_layer_evecs_no_soc:
        orbitals_per_X = 3
        orbitals_per_M = 5
        comp_groups = make_comp_groups(orbitals_per_X, orbitals_per_M, args.num_layers)
    else:
        comp_groups = None

    plotBands(qe_bands, Hr, alat, latVecs, minE_plot, maxE_plot, outpath, show=args.show,
            symList=band_path_labels(), fermi_energy=E_F, plot_evecs=args.plot_evecs,
            comp_groups=comp_groups, fermi_shift=args.fermi_shift, plot_band_extrema=args.band_extrema)

if __name__ == "__main__":
    _main()
