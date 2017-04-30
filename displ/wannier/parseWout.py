def atom_order_from_wout(wout_path):
    with open(wout_path, 'r') as fp:
        lines = fp.readlines()

    site_table_index = None
    site_table_head = " |   Site       Fractional Coordinate          Cartesian Coordinate (Ang)     |"
    for line_index, line in enumerate(lines):
        if line.startswith(site_table_head):
            site_table_index = line_index
            break

    if site_table_index is None:
        raise ValueError("could not find site table in wout file {}".format(wout_path))

    site_table_foot = " *----------------------------------------------------------------------------*"
    atom_line_index = site_table_index + 2
    atom_symbols, atom_indices, cart_coords = [], [], []
    while not lines[atom_line_index].startswith(site_table_foot):
        atom_line = lines[atom_line_index].strip().split()
        symbol = atom_line[1]
        atom_index = int(atom_line[2])
        cart = [float(atom_line[7]), float(atom_line[8]), float(atom_line[9])]

        atom_symbols.append(symbol)
        atom_indices.append(atom_index)
        cart_coords.append(cart)
        atom_line_index += 1

    return atom_symbols, atom_indices, cart_coords
