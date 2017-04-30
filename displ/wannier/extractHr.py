import numpy as np
import numpy.matlib as matlib

def extractHr(filePath):
    '''Extract the Wannier Hamiltonian H_{ij}^{nm} from the hr.dat file
    located at filePath.

    Returns a dictionary mapping (ra, rb, rc) vectors to a tuple which
    contains numpy matrices storing values of type complex128 and the
    inverse weight value ndegen.

    The keys (ra, rb, rc) represent r_i - r_j in lattice coordinates
    (i.e. r_i - r_j = ra * a + rb * b + rc * c with a, b, and c the lattice
    vectors). The matrix values represent H_{ij} and the elements H[n, m]
    correspond to H_{ij}^{nm} in the Wannier basis.
    '''
    fp = open(filePath, 'r')
    lines = fp.readlines()
    fp.close()

    comment_line, nb, nr, degen, startH = _extractHrHeader(lines)
    numLinesH = nr * nb * nb

    result = {}
    # index for (ra, rb, rc), used to address degen list
    r_index = 0
    prev_r = None

    for i in range(startH, startH + numLinesH):
        lineStr = lines[i].strip().split()
        # Line format: ra  rb  rc  n  m  Re[H_{r}^{nm}]  Im[H_{r}^{nm}]
        ra, rb, rc, n, m = map(int, lineStr[0:5])
        reH, imH = map(float, lineStr[5:7])

        # If (ra, rb, rc) has changed, update r_index.
        if prev_r != None and (ra, rb, rc) != prev_r:
            r_index += 1
        prev_r = (ra, rb, rc)

        # Store H entry for this line.
        if (ra, rb, rc) in result:
            # Subtract 1 since numpy matrix is 0-indexed.
            result[(ra, rb, rc)][0][n - 1, m - 1] = complex(reH, imH)
        else:
            H = matlib.zeros((nb, nb), dtype=np.complex128)
            H[n - 1, m - 1] = complex(reH, imH)
            result[(ra, rb, rc)] = (H, degen[r_index])

    return result

def _extractHrHeader(lines):
    '''Extract the header lines from the hr.dat file.

    Returns the values of the comment line, nb, nr, a list of degeneracies,
    and the index of the first line after the header.
    '''
    # The first line is a comment - we will ignore this.
    comment_line = lines[0]
    # The second line is the number of bands in the Wannier basis.
    nb = int(lines[1].strip())
    # The third line is the number of (ra, rb, rc) vectors included in the file.
    nr = int(lines[2].strip())

    # The next lines give the degeneracies of each (ra, rb, rc) vector, which
    # will be used as inverse weights. There are 15 values are listed per line.
    numDegenLines = int(np.ceil(float(nr) / 15.0))
    degenStartLine = 3
    degenFinishedLine = degenStartLine + numDegenLines
    degens = []

    for i in range(degenStartLine, degenFinishedLine):
        degen_line = map(int, lines[i].strip().split())
        degens.extend(degen_line)

    return comment_line, nb, nr, degens, degenFinishedLine

def BandNumber(Hr):
    r0 = (0, 0, 0)
    Hr_0 = Hr[r0][0]
    nb = Hr_0.shape[0]
    return nb

def WithDistCutoff(Hr, cutoff, latVecs, exclude_cutoff_states=None, atom_pos=None, atom_offsets=None):
    '''Return copy of Hr with matrix elements connecting unit cells or atoms
    farther apart than the given cutoff distance removed.

    latVecs is a list of lattice vectors given in the same units as cutoff.

    If exclude_cutoff_states != None, this gives a list of the indices of
    states to which the cutoff is not applied.

    If atom_pos != None, this gives a list of the positions (in lattice
    coordinates) of the atoms in the unit cell. If this list is given,
    the cutoff is applied to individual atoms instead of unit cells.

    If atom_pos != None, then atom_offsets must give a list of the band
    indices at which the bands corresponding to each atom begin (e.g. for
    CoMn, with atom_pos = [(0, 0, 0), (0.5, 0.5, 0.5)] giving the positions
    of Co and Mn respectively, and Co bands being (0, 1, ..., 8) and Mn bands being
    (9, 10, ..., 17), atom_offsets is given by [0, 9].
    '''
    a, b, c = map(np.array, latVecs)
    Hr_cut = {}
    for r, val in Hr.items():
        num_bands = val[0].shape[0]
        # Do we have information here about multiple atoms in the cell?
        # If so, need to consider cutoff distance individually for each
        # pair of (atom in origin cell, atom in cell at r).
        if atom_pos != None:
            r_arr = r[0]*a + r[1]*b + r[2]*c
            val_with_cutoff = [np.zeros((num_bands, num_bands), dtype=np.complex128), val[1]]
            val_set = False
            for at0_index, pos_0 in enumerate(atom_pos):
                for atc_index, pos_c in enumerate(atom_pos):
                    # rdist_at = from at0 (atom in cell at 0) to atc (atom in cell at r)
                    r_at = r_arr + pos_c[0]*a + pos_c[1]*b + pos_c[2]*c - pos_0[0]*a - pos_0[1]*b - pos_0[2]*c
                    rdist_at = np.linalg.norm(r_at)
                    #print(r, pos_0, pos_c, r_at, rdist_at)
                    offset0 = atom_offsets[at0_index]
                    offsetc = atom_offsets[atc_index]
                    bands_row_start, bands_col_start = offset0, offsetc
                    bands_row_stop, bands_col_stop = None, None
                    if at0_index != len(atom_pos) - 1:
                        bands_row_stop = atom_offsets[at0_index+1]
                    else:
                        bands_row_stop = num_bands
                    if atc_index != len(atom_pos) - 1:
                        bands_col_stop = atom_offsets[atc_index+1]
                    else:
                        bands_col_stop = num_bands
                    bands_iter = (bands_row_start, bands_row_stop, bands_col_start, bands_col_stop)
                    #print(bands_iter)

                    this_val_set = _add_r_with_cutoff(r, val_with_cutoff, val, rdist_at, cutoff, bands_iter, exclude_cutoff_states)
                    val_set = val_set or this_val_set
            if val_set:
                #print(val_with_cutoff)
                Hr_cut[r] = val_with_cutoff
        # No information about atom positions -- apply cutoff to whole cell.
        else:
            rdist = np.linalg.norm(np.add(np.add(r[0]*a, r[1]*b), r[2]*c))
            bands_iter = [0, num_bands, 0, num_bands]
            val_with_cutoff = [np.zeros((num_bands, num_bands), dtype=np.complex128), val[1]]
            val_set = _add_r_with_cutoff(r, val_with_cutoff, val, rdist, cutoff, bands_iter, exclude_cutoff_states)
            if val_set:
                Hr_cut[r] = val_with_cutoff

    return Hr_cut

def _add_r_with_cutoff(r, val_with_cutoff, Hval, rdist, cutoff, bands_iter, exclude_cutoff_states):
    bands_row_start, bands_row_stop, bands_col_start, bands_col_stop = bands_iter
    val_set = False
    #print(rdist, cutoff)
    #print(rdist < cutoff)
    if rdist < cutoff:
        val_set = True
        for row in range(bands_row_start, bands_row_stop):
            for col in range(bands_col_start, bands_col_stop):
                val_with_cutoff[0][row, col] += Hval[0][row, col]
    elif exclude_cutoff_states != None:
        for row in range(bands_row_start, bands_row_stop):
            for col in range(bands_col_start, bands_col_stop):
                if row in exclude_cutoff_states and col in exclude_cutoff_states:
                    val_with_cutoff[0][row, col] += Hval[0][row, col]
                    val_set = True
    return val_set

def CopyWithBandsRemoved(Hr, bands_to_remove):
    '''Return a copy of Hr with the bands listed by their indices in
    bands_to_remove cut out; the new Hamiltonian has a number of bands
    equal to the original number minus len(bands_to_remove).
    '''
    Hr_band_num = Hr[(0, 0, 0)][0].shape[0]
    new_band_num = Hr_band_num - len(bands_to_remove)

    old_indices_for_new_bands = _cut_entries(range(Hr_band_num), bands_to_remove)

    new_Hr = {}
    for r, v in Hr.items():
        degen = v[1]
        new_Hr[r] = [np.zeros((new_band_num, new_band_num), dtype=np.complex128), degen]
        for row in range(new_band_num):
            for col in range(new_band_num):
                old_row, old_col = old_indices_for_new_bands[row], old_indices_for_new_bands[col]
                new_Hr[r][0][row, col] = v[0][old_row, old_col]
    return new_Hr

def _cut_entries(v, cut):
    u = []
    for i, elem in enumerate(v):
        if elem not in cut:
            u.append(elem)
    return u
