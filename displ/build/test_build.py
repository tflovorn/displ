import unittest
import os
import json
import numpy as np
from ase import Atoms
import ase.db
from displ.pwscf.build import build_qe
from displ.build.cell import make_cell, get_layer_system, a_from_2H, h_from_2H
from displ.build.build import (_extract_syms, get_c_sep, get_wann_valence,
        get_num_bands, make_qe_config, get_pseudo_dir)
from displ.build.util import _base_dir

def has_pos_seq(all_pos, all_expected_2d):
    eps = 1e-12
    for pos, expected in zip(all_pos, all_expected_2d):
        for i in range(2):
            if abs(pos[i] - expected[i]) > eps:
                return False

    return True

class TestShift(unittest.TestCase):
    def test_cell_unshifted(self):
        syms_bilayer = ["WSe2", "WSe2"]
        syms_trilayer = ["WSe2", "WSe2", "WSe2"]
        vacuum_dist = 20.0 # Angstrom
        AB_stacking = True

        db_path = os.path.join(_base_dir(), "c2dm.db")
        db = ase.db.connect(db_path)

        for syms in [syms_bilayer, syms_trilayer]:
            c_sep = get_c_sep(db, syms[0])
            layer_systems = [get_layer_system(db, sym, 'H') for sym in syms]
            a = a_from_2H(layer_systems[0])
            hs = [h_from_2H(layer_system) for layer_system in layer_systems]

            for AB_stacking in [False, True]:
                # layer_shifts = None should be the same as specifying
                # a shift of (0.0, 0.0) for all layers.
                latvecs_None, at_syms_None, cartpos_None = make_cell(db, syms, c_sep, vacuum_dist,
                        AB_stacking=AB_stacking, layer_shifts=None)

                shifts_zero = [(0.0, 0.0)] * len(syms)
                latvecs_zero, at_syms_zero, cartpos_zero = make_cell(db, syms, c_sep, vacuum_dist,
                        AB_stacking=AB_stacking, layer_shifts=None)

                assert((latvecs_None == latvecs_zero).all())
                assert(at_syms_None == at_syms_zero)

                for at_pos_None, at_pos_zero in zip(cartpos_None, cartpos_zero):
                    assert((at_pos_None == at_pos_zero).all())

                assert(at_syms_zero == ["Se", "W", "Se"] * len(syms))

                # Should have the correct lattice constant.
                eps = 1e-12
                assert(abs(np.linalg.norm(latvecs_None[0]) - a) < eps)
                assert(abs(np.linalg.norm(latvecs_None[1]) - a) < eps)

                system = Atoms(symbols=at_syms_zero, positions=cartpos_zero,
                        cell=latvecs_zero, pbc=True)
                system.center(axis=2)

                A = (0.0, 0.0)
                B = (1/3, 2/3)

                latpos = system.get_scaled_positions()
                for layer_index in range(len(syms)):
                    # Atoms should have the correct in-place positions.
                    layer_pos = latpos[3*layer_index:3*layer_index+3]
                    if AB_stacking:
                        if layer_index % 2 == 0:
                            assert(has_pos_seq(layer_pos, [A, B, A]))
                        else:
                            assert(has_pos_seq(layer_pos, [B, A, B]))
                    else:
                        assert(has_pos_seq(layer_pos, [A, B, A]))
                    
                    # Atoms should have the correct vertical positions.
                    h = hs[layer_index]
                    layer_cartpos = cartpos_zero[3*layer_index:3*layer_index+3]
                    zs = [layer_cartpos[i][2] for i in range(3)]
                    assert(abs(zs[2] - zs[0] - h) < eps)
                    assert(abs(zs[2] - zs[1] - h/2) < eps)

                    if layer_index != 0:
                        z_below = cartpos_zero[3*layer_index - 1][2]
                        assert(abs(zs[0] - z_below - c_sep) < eps)

def check_qe_config(testcase, qe_config, qe_config_expected, soc, xc, pp):
    testcase.assertEqual(sorted(qe_config.keys()), sorted(qe_config_expected.keys()))

    for k, v in qe_config.items():
        v_expected = qe_config_expected[k]

        # pseudo_dir value is system-dependent.
        if k == 'pseudo_dir':
            pseudo_dir_expected = get_pseudo_dir(soc, xc, pp)
            testcase.assertEqual(v, pseudo_dir_expected)
            continue

        # Remaining values are not.
        testcase.assertEqual(v, v_expected)

def check_qe_input(testcase, qe_input, qe_input_expected, soc, xc, pp):
    for line, line_expected in zip(qe_input.split('\n'), qe_input_expected.split('\n')):
        # pseudo_dir value is system-dependent.
        if line.split('=')[0].strip() == 'pseudo_dir':
            pseudo_dir_expected = "'{}',".format(get_pseudo_dir(soc, xc, pp))
            testcase.assertEqual(line.split('=')[1].strip(), pseudo_dir_expected)
            continue

        # Remaining values are not.
        testcase.assertEqual(line, line_expected)

class TestQE(unittest.TestCase):
    def test_qe_config(self):
        syms = ["WSe2", "WSe2", "WSe2"]
        soc = True
        vacuum_dist = 20.0 # Angstrom
        D = 0.5 # V/nm
        AB_stacking = True
        xc = 'lda'
        pp = 'nc'

        db_path = os.path.join(_base_dir(), "c2dm.db")
        db = ase.db.connect(db_path)
        c_sep = get_c_sep(db, syms[0])

        latvecs, at_syms, cartpos = make_cell(db, syms, c_sep, vacuum_dist, AB_stacking)
        system = Atoms(symbols=at_syms, positions=cartpos, cell=latvecs, pbc=True)
        system.center(axis=2)

        wann_valence, num_wann = get_wann_valence(system.get_chemical_symbols(), soc)
        num_bands = get_num_bands(num_wann)

        qe_config = make_qe_config(system, D, soc, num_bands, xc, pp)
        #with open('test_build_qe_config_new.json', 'w') as fp:
        #    json.dump(qe_config, fp)

        with open('test_build_qe_config.json', 'r') as fp:
            qe_config_expected = json.load(fp)

        check_qe_config(self, qe_config, qe_config_expected, soc, xc, pp)

        prefix = 'test'
        qe_input = build_qe(system, prefix, 'scf', qe_config)
        #with open('test_build_qe_input_new', 'w') as fp:
        #    fp.write(qe_input)

        with open('test_build_qe_input', 'r') as fp:
            qe_input_expected = fp.read()

        check_qe_input(self, qe_input, qe_input_expected, soc, xc, pp)

if __name__ == "__main__":
    unittest.main()
