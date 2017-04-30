import numpy as np

def extractQEBands(filePath):
    '''Extract band structure from the Quantum Espresso bands.dat file present
    at filePath. Return the number of bands, the number of k-points, and
    a list of tuples of the form:
        [((ka1, kb1, kc1), (E11, E12, E13...)), ...]
    where each outer tuple contains a k-point (in the Cartesian basis in units
    of 2pi/alat) and the corresponding set of eigenvalues.
    '''
    fp = open(filePath, 'r')
    lines = fp.readlines()
    fp.close()

    # First line: " &plot nbnd= #, nks= # /"
    # where the #'s represent the number of bands and the number of
    # k-points, respectively.
    line0 = lines[0].strip().split()
    nbnd = int(line0[2][:-1]) # use [:-1] to remove comma
    nks = int(line0[4])

    # After header, .dat entry format:
    #
    #   ka, kb, kc
    # ev0  ev1  ev2  ...  ev3
    # ev10 ev11 ...
    #
    # Always have 10 eigenalues per line.
    # Each eigenvalue uses a maximum of 8 characters; eigenvalues may not be
    # separated by a space.
    evnumlines = int(np.ceil(float(nbnd) / 10.0))
    knumlines = 1 + evnumlines

    all_evs = []
    for ik in range(nks):
        # Extract k value.
        startlinenum = 1 + ik * knumlines
        kline = lines[startlinenum].strip().split()
        ka, kb, kc = map(float, kline)
        k_evs = []
        # Extract eigenvalues.
        for ievline in range(evnumlines):
            evlinenum = startlinenum + 1 + ievline
            evline = lines[evlinenum]
            for iev in range(10):
                # Don't overshoot on last line.
                evs_seen = 10*ievline + iev
                if evs_seen >= nbnd:
                    break
                # Still have more eigenvalues; get the next one.
                evstart = 8 * iev
                ev = float(evline[evstart:evstart+8].strip())
                k_evs.append(ev)
        # Got all eigenvalues for this k-point; add to list.
        all_evs.append(((ka, kb, kc), k_evs))

    return nbnd, nks, all_evs
