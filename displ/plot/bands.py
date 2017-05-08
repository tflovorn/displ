from __future__ import division
import argparse
import os
from displ.build.build import _get_work, band_path_labels
from displ.pwscf.extractQEBands import extractQEBands
from displ.pwscf.parseScf import fermi_from_scf, alat_from_scf, latVecs_from_scf
from displ.wannier.extractHr import extractHr
from displ.wannier.plotBands import plotBands

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
    parser.add_argument("--minE", type=float, default=None,
            help="Minimum energy to plot, relative to E_F")
    parser.add_argument("--maxE", type=float, default=None,
            help="Maximum energy to plot, relative to E_F")
    parser.add_argument("--plot_evecs", action='store_true',
            help="Plot eigenvector decomposition")
    parser.add_argument("--show", action='store_true',
            help="Show band plot instead of outputting file")
    args = parser.parse_args()

    work = _get_work(args.subdir, args.prefix)
    wannier_dir = os.path.join(work, "wannier")
    scf_path = os.path.join(wannier_dir, "scf.out")

    E_F = fermi_from_scf(scf_path)
    if args.minE is not None and args.maxE is not None:
        minE_plot = E_F + args.minE
        maxE_plot = E_F + args.maxE
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

    # TODO?
    comp_groups = None

    plotBands(qe_bands, Hr, alat, latVecs, minE_plot, maxE_plot, outpath, show=args.show,
            symList=band_path_labels(), fermi_energy=E_F, plot_evecs=args.plot_evecs,
            comp_groups=comp_groups, fermi_shift=args.fermi_shift)

if __name__ == "__main__":
    _main()
