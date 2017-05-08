from __future__ import division
import os
import json
from argparse import ArgumentParser
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from displ.pwscf.parseScf import fermi_from_scf, alat_from_scf, D_from_scf
from displ.plot.util import _base_dir, _global_config

def _rlist(row):
    r_str = "{} {} {}".format(row[0], row[1], row[2])
    return r_str

def PartialDos(Hr_path, R, minE, maxE, num_dos, sigma, n0, cwannier_path, pdos_outpath):
    run_pdos_path = os.path.join(cwannier_path, "RunPartialDos.out")
    run_pdos_call = [run_pdos_path, Hr_path, pdos_outpath, str(minE), str(maxE),
            str(num_dos), str(sigma), str(n0), str(n0), str(n0),
            _rlist(R[0, :]), _rlist(R[1, :]), _rlist(R[2, :])]

    subprocess.call(run_pdos_call)
    pdos_vals, E_vals = _extract_pdos_vals(pdos_outpath)

    return pdos_vals, E_vals

def PartialNum(Hr_path, R, minE, maxE, num_dos, num_electrons, n0, cwannier_path, pnum_outpath):
    run_pnum_path = os.path.join(cwannier_path, "RunPartialNum.out")
    run_pnum_call = [run_pnum_path, Hr_path, pnum_outpath, str(minE), str(maxE), 
            str(num_dos), str(num_electrons), str(n0), str(n0), str(n0),
            _rlist(R[0, :]), _rlist(R[1, :]), _rlist(R[2, :])]

    subprocess.call(run_pnum_call)
    pnum_vals, E_vals, n_fermi, fermi_from_num = _extract_pnum_vals(pnum_outpath)

    return pnum_vals, E_vals, n_fermi, fermi_from_num

def pdos_from_num(this_pnum, this_Es):
    # Approximate DOS(E) = (n(E_i) - n(E_{i-1})) / (E_i - E_{i-1})
    num_Es = len(this_pnum[0])

    pdos = []
    for band_index, band_pnum in enumerate(this_pnum):
        pdos.append([])
        for E_index in range(num_Es):
            if E_index == 0:
                pdos[band_index].append(0.0)
                continue
            ndiff = band_pnum[E_index] - band_pnum[E_index - 1]
            pdos_val = ndiff / (this_Es[E_index] - this_Es[E_index - 1])
            pdos[band_index].append(pdos_val)

    return pdos

def _extract_pdos_vals(dosPath):
    fp = open(dosPath)
    lines = fp.readlines()
    fp.close()

    pdos_vals, E_vals = [], []
    for i, line in enumerate(lines):
        # Skip header.
        if i == 0:
            continue
        # If we get here, on a data line.
        line_split = line.strip().split('\t')
        # Need to initialize pdos_vals inner lists.
        if len(pdos_vals) == 0:
            num_bands = len(line_split) - 1
            for band_index in range(num_bands):
                pdos_vals.append([])

        E_vals.append(float(line_split[0]))

        for band_index in range(num_bands):
            pdos_vals[band_index].append(float(line_split[band_index+1]))

    return pdos_vals, E_vals

def _extract_pnum_vals(numPath):
    fp = open(numPath)
    lines = fp.readlines()
    fp.close()

    pnum_vals, E_vals = [], []
    n_fermi = []
    fermi_from_num = 0.0

    for i, line in enumerate(lines):
        # Skip headers.
        if i == 0 or i == 2:
            continue

        # Fermi energy line.
        if i == 1:
            line_split = line.strip().split('\t')
            fermi_from_num = float(line_split[0])
            for num_val in line_split[1:]:
                n_fermi.append(float(num_val))
            continue

        # If we get here, on a normal data line.
        line_split = line.strip().split('\t')
        # Need to initialize pnum_vals inner lists.
        if len(pnum_vals) == 0:
            num_bands = len(line_split) - 1
            for band_index in range(num_bands):
                pnum_vals.append([])

        E_vals.append(float(line_split[0]))

        for band_index in range(num_bands):
            pnum_vals[band_index].append(float(line_split[band_index+1]))

    return pnum_vals, E_vals, n_fermi, fermi_from_num

def plot_pvals(pdos, Es, fermi, outpath, ylabel, minE=None, maxE=None, group_labels=None):
    spin_polarized = len(pdos) > 1
    num_bands = len(pdos[0])

    plt.axvline(0.0, color='k')

    cvals = np.linspace(0.0, 0.9, num_bands, endpoint=True)

    pdos_max = 0.0
    for band_index in range(num_bands):
        xs = [E - fermi for E in Es[0]]

        if group_labels is not None:
            label = group_labels[band_index]
        else:
            label = "Band {}".format(str(band_index))

        color = cm.Accent(cvals[band_index], 1)

        plt.plot(xs, pdos[0][band_index], label=label, c=color)

        for E, pdos_val in zip(Es[0], pdos[0][band_index]):
            E_shifted = E - fermi
            if minE <= E_shifted <= maxE and pdos_val > pdos_max:
                pdos_max = pdos_val

    if spin_polarized:
        plt.axhline(0.0, color='k')
        neg_pdos = []
        for band_index, band in enumerate(pdos[1]):
            neg_pdos.append([])
            for val in band:
                neg_pdos[band_index].append(-val)

            xs = [E - fermi for E in Es[1]]

            if group_labels is not None:
                label = group_labels[band_index]
            else:
                label = "Band {}".format(str(band_index))

            color = cm.Accent(cvals[band_index], 1)

            plt.plot(Es[1] - fermi, pdos[1][band_index], label=label, c=color)
    else:
        plt.ylim(0.0, pdos_max)

    if minE is None or maxE is None:
        plt.xlim(Es[0][0], Es[0][-1])
    else:
        plt.xlim(minE, maxE)

    plt.xlabel("E [eV]")
    plt.ylabel(ylabel)
    plt.legend(loc=0)

    plt.savefig(outpath + '.png', bbox_inches='tight', dpi=500)
    plt.clf()

def export_pdos_data(prefix, Es_shifted, group_pdos, group_labels_split):
    out = {}
    out["E"] = Es_shifted
    out["PDOS"] = {}
    for pdos, label in zip(group_pdos, group_labels_split):
        out["PDOS"][label] = pdos

    fname = "{}_export.json".format(prefix)
    with open(fname, 'w') as fp:
        json.dump(out, fp)

def _main():
    parser = ArgumentParser(description="Plot partial DOS from Wannier Hamiltonian.")
    parser.add_argument('prefix', type=str, help="Name of the system.")
    parser.add_argument('--num', action='store_true',
            help="Calculate number of states from tetrahedron method and DOS from n(E).")
    parser.add_argument('--num_electrons', type=float, default=None,
            help="Number of valence electrons in Wannier bands (must be specified if --num is used)")
    parser.add_argument('--load', action='store_true',
            help="Load pre-calculated DOS/n data instead of calculating.")
    parser.add_argument('--spin_polarized', help="Use spin-polarized system.",
            action='store_true')
    parser.add_argument('--groups', type=str, default=None,
            help="Partial DOS values to be added together (ex: '0,1,2;3,4,5' reports pdos 0+1+2 and 3+4+5)")
    parser.add_argument('--group_labels', type=str, default=None,
            help="Labels for groups specified by --groups")
    parser.add_argument('--minE', type=float, help="Minimum energy to plot.", default=None)
    parser.add_argument('--maxE', type=float, help="Maximum energy to plot.", default=None)
    parser.add_argument('--num_dos', type=int, default=3000,
            help="Number of energy values at which to calculate density of states")
    parser.add_argument('--n0', type=int, default=8,
            help="Number of k-points in each direction")
    parser.add_argument('--sigma', type=float, default=0.1,
            help="Gaussian delta-function broadening constant")
    parser.add_argument('--cwannier_path', type=str, default=None,
            help="Path to cwannier/ directory")
    args = parser.parse_args()

    base_dir = _base_dir()
    cwannier_path = os.path.join(base_dir, "cwannier")
    work_base = os.path.expandvars(_global_config()["work_base"])
    wannier_dir = os.path.join(work_base, args.prefix, "wannier")
    outPath = "{}_{}".format(args.prefix, str(args.n0))

    # Get Fermi energy from SCF output.
    scfPath = os.path.join(wannier_dir, "scf.out")
    fermi = fermi_from_scf(scfPath)
    alat = alat_from_scf(scfPath)
    D = alat * D_from_scf(scfPath)
    R = 2.0 * np.pi * np.linalg.inv(D)

    # Load Wannier Hamiltonian and calculate DOS.
    if not args.spin_polarized:
        spins = [""]
    else:
        spins = ["_up", "_dn"]

    # Calculate PDOS values.
    pdos, Es = [], []
    pnum, n_fermi = [], []
    fermi_from_num = 0.0
    for sp in spins:
        Hr_path = os.path.join(wannier_dir, "{}{}_hr.dat".format(args.prefix, sp))
        if args.num:
            pnum_outpath = outPath + "_pnum_data"
            if not args.load:
                this_pnum, this_Es, n_fermi, fermi_from_num = PartialNum(Hr_path, R, args.minE, args.maxE, args.num_dos, args.num_electrons, args.n0, cwannier_path, pnum_outpath)
            else:
                this_pnum, this_Es, n_fermi, fermi_from_num = _extract_pnum_vals(pnum_outpath)

            this_pdos = pdos_from_num(this_pnum, this_Es)
            pnum.append(this_pnum)
            pdos.append(this_pdos)
            Es.append(this_Es)
        else:
            pdos_outpath = outPath + "_pdos_data"
            if not args.load:
                this_pdos, this_Es = PartialDos(Hr_path, R, args.minE, args.maxE, args.num_dos, args.sigma, args.n0, cwannier_path, pdos_outpath)
            else:
                this_pdos, this_Es = _extract_pdos_vals(pdos_outpath)

            pdos.append(this_pdos)
            Es.append(this_Es)

    # Aggregate groups.
    group_pdos, group_pnum = [], []
    if args.groups != None:
        group_vals = args.groups.split(";")
        num_groups = len(group_vals)
        for sp_index, sp_pdos in enumerate(pdos):
            sp_pnum = pnum[sp_index]
            group_pdos.append([])
            if args.num:
                group_pnum.append([])

            for group_index, group in enumerate(group_vals):
                group_pdos[sp_index].append([])
                if args.num:
                    group_pnum[sp_index].append([])

                group_band_indices = list(map(int, group.split(",")))
                for E_index in range(len(Es[sp_index])):
                    total_dos, total_num = 0.0, 0.0
                    for band_index in group_band_indices:
                        total_dos += sp_pdos[band_index][E_index]
                        if args.num:
                            total_num += sp_pnum[band_index][E_index]

                    group_pdos[sp_index][group_index].append(total_dos)
                    if args.num:
                        group_pnum[sp_index][group_index].append(total_num)
    else:
        group_pdos = pdos
        group_pnum = pnum

    if args.group_labels is not None:
        group_labels_split = args.group_labels.split(";")
    else:
        group_labels_split = None

    if args.num:
        # Echo n(E_Fermi)
        print("Got Fermi energy from n(E) = {}; scf Fermi energy = {}".format(str(fermi_from_num), str(fermi)))
        print("n(E) at the E_F from n(E) = {}".format(str(n_fermi)))
        # Make pnum plot.
        plot_pnum_outpath = outPath + "_pnum"
        plot_pvals(group_pnum, Es, fermi_from_num, plot_pnum_outpath, "$n(E)$", args.minE, args.maxE, group_labels_split)

    # Make PDOS plot.
    plot_pdos_outpath = outPath + "_pdos"
    plot_pvals(group_pdos, Es, fermi, plot_pdos_outpath, "PDOS [eV$^{-1}$]", args.minE, args.maxE, group_labels_split)

    # Export data to allow plotting with different styling.
    # NOTE - only supports case with non-spin-polarized or noncollinear spin cases.
    # TODO - support collinear spin-polarized case.
    Es_shifted = [E - fermi for E in Es[0]]
    export_pdos_data(outPath, Es_shifted, group_pdos[0], group_labels_split)

if __name__ == "__main__":
    _main()
