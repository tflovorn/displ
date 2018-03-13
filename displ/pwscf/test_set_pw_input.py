import unittest
from displ.pwscf.parseScf import final_coordinates_from_scf
from displ.pwscf.set_pw_input import with_coordinates

class TestSetRelaxed(unittest.TestCase):
    def test_set_relaxed(self):
        expected_positions_type = "crystal"
        expected_atom_symbols = ["Mo", "S", "S", "W", "S", "S"]
        expected_atom_positions = [[0.000000000, 0.000000000, 0.395206707],
                [0.333333333, 0.666666667, 0.342695335],
                [0.333333333, 0.666666667, 0.447806971],
                [0.333333333, 0.666666667, 0.605110054],
                [0.000000000, 0.000000000, 0.552314942],
                [0.000000000, 0.000000000, 0.657832285]]

        relax_path = "test_data/relax.out"
        positions_type, atom_symbols, atom_positions = final_coordinates_from_scf(relax_path)
        self.assertEqual(positions_type, expected_positions_type)
        self.assertEqual(atom_symbols, expected_atom_symbols)

        eps = 1e-9
        for pos, expected_pos in zip(atom_positions, expected_atom_positions):
            for c, expected_c in zip(pos, expected_pos):
                self.assertTrue(abs(c - expected_c) < eps)

        scf_path = "test_data/MoS2_WS2_da_0.000_db_0.000.scf.in"
        scf_with_relaxed_lines = with_coordinates(scf_path, positions_type, atom_symbols, atom_positions).split('\n')

        with open(scf_path, 'r') as fp:
            scf_orig_lines = fp.readlines()

        scf_atom_head_line = 30
        scf_atom_lines = list(range(31, 37))

        for i, (relaxed_line, orig_line) in enumerate(zip(scf_with_relaxed_lines, scf_orig_lines)):
            if i == scf_atom_head_line:
                self.assertEqual(relaxed_line, "ATOMIC_POSITIONS {}".format(positions_type))
            elif i in scf_atom_lines:
                atom_number = i - scf_atom_lines[0]
                sym = atom_symbols[atom_number]
                pos = atom_positions[atom_number]
                pos_line = " {} {} {} {}".format(sym, str(pos[0]), str(pos[1]),
                        str(pos[2]))
                self.assertEqual(relaxed_line, pos_line)
            else:
                self.assertEqual(relaxed_line + "\n", orig_line)

if __name__ == "__main__":
    unittest.main()
