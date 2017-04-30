import os
import subprocess
from uuid import uuid4
from mkheusler.wannier.wannier_util import _base_dir
from mkheusler.wannier.bands import Hk_recip

def Dos(minE, maxE, num_dos, na, nb, nc, R, HrPath):
    '''Return two lists, dos_vals and E_vals. dos_vals contains the density
    of states D(E) at num_dos energies E between minE and maxE. E_vals
    contains the E values at which the corresponding element of dos_vals
    was evaluated.
    
    Hr gives the Wannier Hamiltonian in the form returned by extractHr.
    na, nb, nc give the number of k-points in each reciprocal lattice direction
        to use to obtain D(E).
    R is a numpy matrix with rows equal to the reciprocal lattice vectors.
    HrPath gives the path to the hr.dat file containing the Wannier Hamiltonian.
    '''
    cwannierPath = os.path.join(_base_dir(), "cwannier")
    dos_outpath = str(uuid4())
    run_dos_path = os.path.join(cwannierPath, "RunDosValues.out")
    run_dos_call = [run_dos_path, HrPath, dos_outpath, str(minE), str(maxE), str(num_dos), str(na), str(nb), str(nc), _rlist(R[0, :]), _rlist(R[1, :]), _rlist(R[2, :])]

    subprocess.call(run_dos_call)

    dos_vals, E_vals = _extract_cdos_vals(dos_outpath)
    subprocess.call(["rm", dos_outpath])

    return dos_vals, E_vals

def _rlist(row):
    r_str = "{} {} {}".format(row[0], row[1], row[2])
    return r_str

def _extract_cdos_vals(dosPath):
    fp = open(dosPath)
    lines = fp.readlines()
    fp.close()

    dos_vals, E_vals = [], []
    for i, line in enumerate(lines):
        # Skip header.
        if i == 0:
            continue
        # If we get here, on a data line.
        line_split = line.strip().split('\t')
        E_vals.append(float(line_split[0]))
        dos_vals.append(float(line_split[1]))

    return dos_vals, E_vals
