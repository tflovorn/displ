import argparse
import json
import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt
from displ.wannier.bands import Hk, Hk_recip
from displ.wannier.extractHr import extractHr
from displ.pwscf.extractQEBands import extractQEBands
from displ.pwscf.parseScf import alat_from_scf, latVecs_from_scf

# Switch to size eigenvalue intensity markers based on component values
# (instead of only using color to denote component value).
# Avoids problem of overlapping markers.
eval_dot_size = True

def plotBands(evalsDFT, Hr, alat, latVecs, minE, maxE, outpath, show=False, symList=None, fermi_energy=None, plot_evecs=False, plot_DFT_evals=True, comp_groups=None, fermi_shift=False):
    '''Create a plot of the eigenvalues given by evalsDFT (which has the form
    returned by extractQEBands()). Additionally, plot the eigenvalues of the
    system described by the Wannier Hamiltonian Hr (which has the form
    returned by extractHr()), assuming periodicity in all directions.

    alat is the lattice vector units used by QE.
    latVecs contains a list of lattice vectors.
    If Hr == None, alat and latVecs are not used.

    The range of energies plotted is restricted to fall in [minE, maxE].
    If minE and/or maxE are None, the range of energies covers all bands.

    k-points used to plot the eigenvalues of Hr are linearly interpolated
    between the k-points listed in evalsDFT.
    '''
    D = np.array(latVecs).T
    R = 2*np.pi*np.linalg.inv(D)
    # Get list of all k-points in evalsDFT.
    # While we're iterating through evalsDFT, also construct eigenvalue
    # sequences for plotting. Instead of a list of eigenvalues for each
    # k-point, we need to plot a sequence of lists ranging over all k-points,
    # where each list has one eigenvalue for every k-point.
    DFT_ks = []
    DFT_ys = []
    for k, evs in evalsDFT:
        DFT_ks.append(k)
        # Set up DFT_ys to be a length-len(evs) list of empty lists.
        if len(DFT_ys) == 0:
            for i in range(len(evs)):
                DFT_ys.append([])
        # Add each of this k-point's eigenvalues to the corresponding list.
        for i, ev in enumerate(evs):
            if fermi_shift and fermi_energy is not None:
                DFT_ys[i].append(ev - fermi_energy)
            else:
                DFT_ys[i].append(ev)
    # Construct list of k-points to evaluate Wannier Hamiltonian at by
    # interpolating between k-points in evalsDFT.
    Hr_ks, Hr_xs, Hr_ys, DFT_xs, Hr_evecs = None, None, None, None, None
    Hr_ks_per_DFT_k = 1
    if Hr is not None:
        Hr_ks_per_DFT_k = 10
        Hr_ks = _interpolateKs(DFT_ks, Hr_ks_per_DFT_k)
        if not plot_evecs:
            Hr_ys = _getHks(Hr, Hr_ks, alat, latVecs, plot_evecs=False,
                    fermi_shift=fermi_shift, fermi_energy=fermi_energy)[0]
        else:
            Hr_ys, Hr_evecs = _getHks(Hr, Hr_ks, alat, latVecs, plot_evecs,
                    fermi_shift=fermi_shift, fermi_energy=fermi_energy)

    Hr_ys_eval = None
    if plot_evecs and eval_dot_size:
        # Also need Hr_ys formatted for plot() in this case.
        Hr_ys_eval = _getHks(Hr, Hr_ks, alat, latVecs, plot_evecs=False,
                fermi_shift=fermi_shift, fermi_energy=fermi_energy)[0]

    # Scale k-point index values to size each panel according to the Cartesian
    # distance between high-symmetry points.
    if symList is not None:
        nk_per_sym = (len(DFT_ks) - 1) // (len(symList) - 1)
        DFT_sym_k_indices = [i*nk_per_sym for i in range(len(symList))]
        # ks in DFT_ks and Hr_ks are in Cartesian basis in units of 2pi/alat.
        # _recip_dist and _scaled_k_xs expect reciprocal lattice basis.
        Rinv = np.linalg.inv(R)
        DFT_ks_Recip = [np.dot(Rinv, (2*np.pi/alat)*np.array(k)) for k in DFT_ks]
        DFT_recip_dists = _recip_dist(DFT_sym_k_indices, DFT_ks_Recip, R)
        DFT_xs = _scaled_k_xs(DFT_sym_k_indices, DFT_ks_Recip, DFT_recip_dists)
        if Hr is not None:
            sym_k_indices = [i*nk_per_sym*Hr_ks_per_DFT_k for i in range(len(symList))]
            Hr_ks_Recip = [np.dot(Rinv, (2*np.pi/alat)*np.array(k)) for k in Hr_ks]
            recip_dists = _recip_dist(sym_k_indices, Hr_ks_Recip, R)
            Hr_xs = _scaled_k_xs(sym_k_indices, Hr_ks_Recip, recip_dists)
            sym_xs = [Hr_xs[sym_k_indices[i]] for i in range(len(symList))]
        else:
            sym_xs = [DFT_xs[DFT_sym_k_indices[i]] for i in range(len(symList))]
    else:
        sym_k_indices = None
        sym_xs = None
        if Hr is not None:
            Hr_xs = range(len(Hr_ks))
            DFT_xs = range(0, len(Hr_ks), Hr_ks_per_DFT_k)
        else:
            DFT_xs = range(0, len(DFT_ks))

    # Make plot.
    if not plot_evecs:
        if Hr is not None:
            for Hr_evs in Hr_ys:
                plt.plot(Hr_xs, Hr_evs, 'r')

        if plot_DFT_evals:
            for DFT_evs in DFT_ys:
                if show:
                    plt.plot(DFT_xs, DFT_evs, 'ko')
                else:
                    plt.plot(DFT_xs, DFT_evs, 'ko', markersize=2)

        _set_fermi_energy_line(fermi_shift, fermi_energy)
        _set_sympoints_ticks(symList, sym_xs)
        _set_plot_boundaries(DFT_xs, minE, maxE, fermi_shift, fermi_energy)
        _save_plot(show, outpath)

        if Hr is not None:
            _export_data(outpath, Hr_xs, sym_k_indices, symList, Hr_ys)
    else:
        # Eigenvectors are columns of each entry in Hr_evecs.
        # --> The number of eigenvector components = the number of rows.
        # Make plot with eigenvector weight = |eigenvector component|^2.
        evec_components = Hr_evecs[0].shape[0]
        if comp_groups is None:
            used_comp_groups = []
            for comp in range(evec_components):
                used_comp_groups.append([comp])
        else:
            used_comp_groups = comp_groups

        for comp_group_index, comp_group in enumerate(used_comp_groups):
            plt_xs, plt_ys, plt_cs = [], [], []
            for x in Hr_xs:
                ys = Hr_ys[x]
                for eval_index, y in enumerate(ys):
                    comp_group_total = 0.0
                    for comp in comp_group:
                        comp_val = abs(Hr_evecs[x][comp, eval_index])**2
                        comp_group_total += comp_val

                    plt_xs.append(x)
                    plt_ys.append(y)
                    plt_cs.append(comp_group_total)
            # Plotting eigenvalues colored by eigenvector weight.
            if not eval_dot_size:
                plt.scatter(plt_xs, plt_ys, c=plt_cs, cmap='gnuplot', s=5, edgecolors="none")
            # Or: plotting eigenvalues sized by eigenvector weight.
            else:
                for Hr_evs in Hr_ys_eval:
                    plt.plot(Hr_xs, Hr_evs, 'k', linewidth=0.5)
                s_weights = []
                scale = 10.0
                for val in plt_cs:
                    s_weights.append(scale*val)
                plt.scatter(plt_xs, plt_ys, c=plt_cs, cmap='gnuplot', s=s_weights,
                        edgecolors="none", facecolors="none")
            plt.colorbar()

            _set_fermi_energy_line(fermi_shift, fermi_energy)
            _set_sympoints_ticks(symList, sym_xs)
            _set_plot_boundaries(DFT_xs, minE, maxE, fermi_shift, fermi_energy)
            _save_plot(show, outpath + "_{}".format(str(comp_group_index)))

def _set_fermi_energy_line(fermi_shift, fermi_energy):
    # Line to show Fermi energy.
    if fermi_shift and fermi_energy is not None:
        plt.axhline(0.0, color='k')
    elif fermi_energy is not None:
        plt.axhline(fermi_energy, color='k')

def _set_sympoints_ticks(symList, sym_xs):
    # Lines and labels for symmetry points.
    if symList is not None:
        for x in sym_xs:
            plt.axvline(x, color='k')
        plt.xticks(sym_xs, symList)

def _set_plot_boundaries(DFT_xs, minE, maxE, fermi_shift, fermi_energy):
    plt.xlim(0, DFT_xs[-1])
    plt.ylabel("$E$ [eV]")
    if minE is not None and maxE is not None:
        if fermi_shift and fermi_energy is not None:
            plt.ylim(minE - fermi_energy, maxE - fermi_energy)
        else:
            plt.ylim(minE, maxE)

def _save_plot(show, outpath):
    if show:
        plt.show()
    else:
        plt.savefig(outpath + '.png', bbox_inches='tight', dpi=500)
    plt.clf()

def _remove_duplicate_k_pairs(evalsDFT):
    result = []
    eps = 1e-9
    for k_index in range(len(evalsDFT)):
        if k_index == 0:
            result.append(evalsDFT[k_index])
        else:
            k = evalsDFT[k_index][0]
            prev_k = evalsDFT[k_index-1][0]
            if not _vec_equal_upto(k, prev_k, eps):
                result.append(evalsDFT[k_index])
    return result

def _vec_equal_upto(u, v, eps):
    if len(u) != len(v):
        return False
    for i in range(len(u)):
        if abs(u[i] - v[i]) > eps:
            return False
    return True

def _recip_dist(sym_indices, ks_cut, R):
    '''Return a list of values which give the Cartesian distance between each
    pair of consecutive symmetry points.
    '''
    dists = []
    for point_index, k_index in enumerate(sym_indices):
        # Skip first point (need to make [k, previous k] pairs).
        if point_index == 0:
            continue
        # Get k and previous k (prev_k).
        k = ks_cut[k_index]
        prev_k_index = sym_indices[point_index-1]
        prev_k = ks_cut[prev_k_index]
        # Convert k and prev_k from reciprocal lattice coordinates to
        # Cartesian coordinates.
        # Note that k and prev_k here are row vectors (i.e. 1x3 'dual vectors',
        # the transpose of 3x1 column 'vectors').
        k_Cart = np.dot(k, R)
        prev_k_Cart = np.dot(prev_k, R)
        # Get vector from k to prev_k and its length.
        k_to_prev_k = np.subtract(prev_k_Cart, k_Cart)
        this_dist = np.linalg.norm(k_to_prev_k)
        dists.append(this_dist)

    return dists

def _scaled_k_xs(sym_indices, ks_cut, recip_dists):
    '''Return a list of x values at which each k value in ks_cut will be
    plotted. The sizes of the panels between symmetry points are scaled
    such that these sizes correspond to the Cartesian distance between the
    k-points at the ends of the panel (relative to the sum of these distances
    over all panels).
    '''
    total_dist = sum(recip_dists)
    xs = []

    current_x = 0.0
    base_x = 0.0
    panel_start_index = 0
    panel_number = 0
    for x_index in range(len(ks_cut)):
        # At the last value in a panel, x_index - panel_start_index = points_in_panel - 1.
        # Set step such that (points_in_panel - 1) * step = panel_x_length,
        # i.e. such that the last value in a panel has an x value the appropriate distance
        # away from the first value in that panel.
        points_in_panel = sym_indices[panel_number+1] - sym_indices[panel_number] + 1
        panel_x_length = recip_dists[panel_number] / total_dist
        step = panel_x_length / (points_in_panel - 1)
        current_x = base_x + (x_index - panel_start_index)*step
        xs.append(current_x)

        # When we reach the right side of a panel, we are at the left side of the next panel.
        # In this situation, current_x = the left side of the next panel: make this the new base_x.
        # Also set panel_start_index = x_index so that xs[panel_start_index] = the new base_x,
        # and advance the count of which panel we are on.
        if x_index in sym_indices and x_index != 0:
            base_x = current_x
            panel_start_index = x_index
            panel_number += 1

    return xs

def _export_data(outpath, Hr_xs, sym_k_indices, symList, Hr_ys):
    out = {}
    out["scaled_k_pos"] = list(Hr_xs)
    out["special_ks_values"] = [[i, label] for i, label in zip(sym_k_indices, symList)]
    out["Ekm"] = [list(band) for band in Hr_ys]
    export_path = "{}_bands_export.json".format(outpath)
    with open(export_path, 'w') as fp:
        json.dump(out, fp)

def plotDFTBands(dft_bands_filepath, outpath, minE=None, maxE=None, show=False, spin=None):
    nb, nks, qe_bands = extractQEBands(dft_bands_filepath)
    plotBands(qe_bands, None, None, None, minE, maxE, outpath, show)

def _interpolateKs(klist, fineness):
    '''Return a list of k-points which linearly interpolates between the
    values given in klist, with the number of points in the returned list
    equal to (fineness) * (the number of points in klist).
    '''
    interpolated = []
    for ik, k in enumerate(klist[:-1]):
        step = np.subtract(klist[ik+1], k) / float(fineness)
        for j in range(fineness):
            interpolated.append(k + j * step)
    interpolated.append(klist[-1])
    return interpolated

def _getHks(Hr, Hr_ks, alat, latVecs, plot_evecs=False, fermi_shift=False, fermi_energy=None):
    '''Iterate through Hr_ks and return a sequence of lists ranging
    over all k-points, where each list has one eigenvalue for every k-point.

    Assume QE format: Hr_ks is in Cartesian basis in units
    of 2pi/alat. latVecs is then required to contain a list of lattice vectors.
    '''
    Hk_ys, Hk_evecs = [], []
    for k in Hr_ks:
        this_Hk = None
        # k in QE's bands.dat is given in Cartesian basis inunits 2pi/alat;
        # convert to distance units.
        kDist = []
        for i in range(3):
            kDist.append(k[i] * 2.0 * np.pi / alat)
        # Get eigenvalues for this k.
        this_Hk = Hk(kDist, Hr, latVecs)

        if not plot_evecs:
            evals = sorted(linalg.eigvalsh(this_Hk))
            # Set up Hk_ys to be a length-len(evs) list of empty lists.
            if len(Hk_ys) == 0:
                for i in range(len(evals)):
                    Hk_ys.append([])
            # Add each of this k-point's eigenvalues to the corresponding list.
            for i, ev in enumerate(evals):
                if fermi_shift and fermi_energy is not None:
                    Hk_ys[i].append(ev - fermi_energy)
                else:
                    Hk_ys[i].append(ev)
        else:
            evals, evecs = linalg.eigh(this_Hk)
            if fermi_shift and fermi_energy is not None:
                evals_shifted = []
                for ev in evals:
                    evals_shifted.append(ev - fermi_energy)
                Hk_ys.append(evals_shifted)
            else:
                Hk_ys.append(evals)

            Hk_evecs.append(evecs)

    return Hk_ys, Hk_evecs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot QE and Wannier bands.")
    parser.add_argument('QE_path', type=str,
            help="Path to QE eigenvalue file.")
    parser.add_argument('outPath', type=str, help="Path for output file.")
    parser.add_argument('--Hr_path', type=str,
            help="Path to Wannier Hamiltonian file.", default=None)
    parser.add_argument('--scf_path', type=str,
            help="Path to QE scf output file.", default=None)
    parser.add_argument('--minE', type=float, help="Minimal energy to plot.", default=None)
    parser.add_argument('--maxE', type=float, help="Maximum energy to plot.", default=None)
    parser.add_argument('--show', help="Show plot before writing file.",
                        action='store_true')
    args = parser.parse_args()

    if args.Hr_path is not None:
        nbnd, nks, evalsQE = extractQEBands(args.QE_path)
        print("QE eigenvalues loaded.")
        Hr = extractHr(args.Hr_path)
        print("Wannier Hamiltonian loaded.")
        alat = alat_from_scf(args.scf_path)
        latVecs = latVecs_from_scf(args.scf_path)
        plotBands(evalsQE, Hr, alat, latVecs, args.minE, args.maxE, args.outPath, args.show)
    else:
        plotDFTBands(args.QE_path, args.outPath, args.minE, args.maxE, args.show)
