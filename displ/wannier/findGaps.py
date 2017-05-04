from displ.wannier.Dos import Dos

def HrFindGaps(minE, maxE, num_dos, na, nb, nc, R, HrPath):
    dos_vals, E_vals = Dos(minE, maxE, num_dos, na, nb, nc, R, HrPath)
    gaps = FindGaps(dos_vals, E_vals)
    return gaps, dos_vals, E_vals

def FindGaps(dos_values, Es):
    '''Search for gaps in the density of states given by the list dos_values.
    The elements of dos_values correspond to the energies given by Es.

    Return a list with elements (gap_start, gap_stop) for each gap detected.
    If no gaps are detected, return a 0-element list.
    '''
    gaps = []
    last_dos_nonzero = False
    in_gap = False
    eps = 1e-12
    gap_start = None
    for i, dos in enumerate(dos_values):
        E = Es[i]
        zero_dos = abs(dos) < eps

        # Detect the beginning of a gap.
        # Need to come from a nonzero DOS value to a zero value.
        if not in_gap and last_dos_nonzero and zero_dos:
            in_gap = True
            gap_start = E
        # Detect the end of a gap.
        # Need to come from a zero dos value to a nonzero value
        elif in_gap and not zero_dos:
            in_gap = False
            gap_stop = E
            gaps.append((gap_start, gap_stop))
            gap_start = None
        # Detect the beginning of the bands (i.e. lowest-energy nonzero dos value).
        elif not zero_dos:
            last_dos_nonzero = True

    return gaps
