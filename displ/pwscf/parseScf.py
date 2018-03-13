import numpy as np

def fermi_from_scf(scf_path):
    fp = open(scf_path, 'r')
    lines = fp.readlines()
    fp.close()

    for line in lines:
        if 'Fermi' in line:
            fermi = float(line.strip().split()[-2])
            return fermi
    # If we get here, didn't find 'Fermi' line in scf_path.
    return None

def alat_from_scf(scf_path):
    lines = None
    with open(scf_path, 'r') as fp:
        lines = fp.readlines()

    # Lattice parameter line has format:
    #   lattice parameter (alat)  =      10.2423  a.u.

    for line in lines:
        lsp = line.strip().split('=')
        if lsp[0].strip() == 'lattice parameter (alat)':
            rhs = lsp[-1].strip().split()
            return float(rhs[0])

def D_from_scf(scf_path):
    '''Returns D in units of alat.
    '''
    fp = open(scf_path, 'r')
    lines = fp.readlines()
    fp.close()

    D = np.zeros((3, 3), dtype=np.float64)
    for line_index, line in enumerate(lines):
        if "crystal axes: (cart. coord. in units of alat)" in line:
            a1_line = lines[line_index+1]
            a2_line = lines[line_index+2]
            a3_line = lines[line_index+3]
            D[:, 0] = list(map(float, a1_line.strip().split()[3:6]))
            D[:, 1] = list(map(float, a2_line.strip().split()[3:6]))
            D[:, 2] = list(map(float, a3_line.strip().split()[3:6]))
            break

    return D

def latVecs_from_scf(scf_path):
    '''Returns latVecs (= D.T) in distance units which are the same as alat
    (i.e. bohr units, not alat units).
    '''
    alat = alat_from_scf(scf_path)
    D = alat * np.array(D_from_scf(scf_path))
    return D.T

def num_electrons_from_scf(scf_path):
    fp = open(scf_path, 'r')
    lines = fp.readlines()
    fp.close()

    for line in lines:
        if 'number of electrons' in line:
            num_electrons = float(line.strip().split()[-1])
            return num_electrons
    # If we get here, didn't find num_electrons in scf_path.
    # Can't continue without it.
    raise ValueError("Did not find num_electrons in scf_path")

def magnetization_from_scf(scf_path):
    total_mag, abs_mag, site_mags = None, None, []
    with open(scf_path, 'r') as fp:
        convergence_line = None
        lines = fp.readlines()
        for i, line in enumerate(lines):
            if 'convergence has been achieved' in line:
                convergence_line = i
                break
        total_mag_line = lines[convergence_line - 3].strip().split()
        abs_mag_line = lines[convergence_line - 2].strip().split()

        total_mag = float(total_mag_line[3])
        abs_mag = float(abs_mag_line[3])

        last_site_moms_line = None
        for i, line in enumerate(lines):
            if 'Magnetic moment per site' in line:
                last_site_moms_line = i

        mag_line = last_site_moms_line + 1
        if mag_line != None:
            spl = lines[mag_line].split()
            while len(spl) > 0 and spl[0] == 'atom:':
                mag = float(spl[5])
                site_mags.append(mag)
                mag_line += 1
                spl = lines[mag_line].split()

    return total_mag, abs_mag, site_mags

def total_energy_eV_from_scf(scf_path):
    eV_per_Ry = 13.605693

    with open(scf_path, 'r') as fp:
        lines = fp.readlines()

    total_energy_eV = None
    for line in lines:
        if line.startswith("!    total energy"):
            total_energy_Ry = float(line.strip().split()[-2])
            total_energy_eV = total_energy_Ry * eV_per_Ry
            break

    if total_energy_eV is None:
        raise ValueError("total energy not found")

    return total_energy_eV

def final_coordinates_from_scf(scf_path):
    with open(scf_path, 'r') as fp:
        lines = fp.readlines()

    final_block_start = None
    final_block_end = None
    for i, line in enumerate(lines):
        if 'Begin final coordinates' in line:
            final_block_start = i

        if 'End final coordinates' in line:
            final_block_end = i
            break

    if final_block_start is None or final_block_end is None:
        raise ValueError("final coordinates block not found")

    # Before atomic positions list, have a line of the form
    # ATOMIC_POSITIONS (crystal)
    atomic_positions_line = lines[final_block_start + 2]
    positions_type = atomic_positions_line.split("(")[1].strip()[:-1]

    atoms_start = final_block_start + 3
    atom_symbols, atom_positions = [], []
    for line in lines[atoms_start:final_block_end]:
        atom_line = list(filter(lambda x: len(x) > 0, line.split(' ')))
        atom_symbols.append(atom_line[0])
        atom_positions.append([])
        for coord in atom_line[1:]:
            if len(atom_positions[-1]) >= 3:
                break

            atom_positions[-1].append(float(coord))

    return positions_type, atom_symbols, atom_positions
