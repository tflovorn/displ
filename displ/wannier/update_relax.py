import argparse
from displ.pwscf.parseScf import final_coordinates_from_scf

def with_coordinates(win_path, positions_type, atom_symbols, atom_positions):
    if positions_type != "crystal":
        raise ValueError(".win file `with_coordinates` implemented only for crystal coordinates")

    # Sort by z coordinate.
    sorted_latpos = sorted(zip(atom_symbols, atom_positions), key=lambda sympos: sympos[1][2])

    with open(win_path, 'r') as fp:
        lines = fp.readlines()

    win_replaced = []
    in_pos_block = False
    pos_block_start = None
    for i, line in enumerate(lines):
        if "begin atoms_frac" in line:
            pos_block_start = i
            in_pos_block = True
            win_replaced.append(line)
        elif in_pos_block:
            atom_index = i - pos_block_start - 1
            sym, pos = sorted_latpos[atom_index]

            win_replaced.append(" {} {} {} {}\n".format(sym, str(pos[0]), str(pos[1]), str(pos[2])))

            if atom_index == len(atom_symbols) - 1:
                in_pos_block = False
        else:
            win_replaced.append(line)

    return ''.join(win_replaced)

def set_relaxed_coordinates(win_paths, relax_path):
    positions_type, atom_symbols, atom_positions = final_coordinates_from_scf(relax_path)

    for path in win_paths:
        relaxed_input = with_coordinates(path, positions_type, atom_symbols,
                atom_positions)

        with open(path, 'w') as fp:
            fp.write(relaxed_input)

def _main():
    parser = argparse.ArgumentParser("Set coordinates to relaxed value",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("relaxed_output", type=str,
            help="Path to pw.x relaxed output text file")
    parser.add_argument("input_win_to_change", type=str,
            help="Path to W90 input file to change to relaxed coordinates")
    args = parser.parse_args()

    set_relaxed_coordinates([args.input_win_to_change], args.relaxed_output)

if __name__ == "__main__":
    _main()
