from displ.pwscf.parseScf import final_coordinates_from_scf

def with_coordinates(pw_in_path, positions_type, atom_symbols, atom_positions):
    """Return a string giving a new input file, which is the same as the one at
    `pw_in_path` except that the ATOMIC_POSITIONS block is replaced by the one
    specified by the other parameters of this function.

    `positions_type`, `atom_symbols`, and `atom_positions` have the same
    meaning as the return values from `final_coordinates_from_scf()`.

    Assumes that there are no whitespace lines in the ATOMIC_POSITIONS block
    (not sure whether this is allowed by QE).
    """
    with open(pw_in_path, 'r') as fp:
        in_lines = fp.readlines()

    out_lines = []
    in_atomic_block = False
    atom_count = 0
    for i, line in enumerate(in_lines):
        if 'ATOMIC_POSITIONS' in line:
            out_lines.append("ATOMIC_POSITIONS {}\n".format(positions_type))
            in_atomic_block = True
        elif in_atomic_block:
            sym, pos = atom_symbols[atom_count], atom_positions[atom_count]
            pos_line = " {} {} {} {}\n".format(sym, str(pos[0]), str(pos[1]),
                    str(pos[2]))
            out_lines.append(pos_line)
            atom_count += 1

            if atom_count == len(atom_symbols):
                in_atomic_block = False
        else:
            out_lines.append(line)

    return ''.join(out_lines)

def set_relaxed_coordinates(pw_in_paths, relax_path):
    positions_type, atom_symbols, atom_positions = final_coordinates_from_scf(relax_path)

    for path in pw_in_paths:
        relaxed_input = with_coordinates(path, positions_type, atom_symbols,
                atom_positions)

        with open(path, 'w') as fp:
            fp.write(relaxed_input)
