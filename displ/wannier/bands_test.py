import unittest
import numpy as np
import numpy.matlib as matlib
from numpy.linalg import inv
from displ.wannier.extractHr import extractHr
from displ.wannier.bands import Hk, dHk_dk

class TestHk(unittest.TestCase):
    def test_HkFe(self):
        Hr = extractHr("test_data/Fe_hr_test.dat")
        k0 = (0.0, 0.0, 0.0)
        latVecs = _latVecs()
        Hk_0 = Hk(k0, Hr, latVecs)

        expected = complex(28.252214, -2.312399946e-18)
        eps = 1e-6
        assertEquivComplex(self, Hk_0[0, 0], expected, eps, eps)

class TestdHkdk(unittest.TestCase):
    def test_dHkdkFe(self):
        Hr = extractHr("test_data/Fe_hr_test.dat")
        k0 = (0.0, 0.0, 0.0)
        latVecs = _latVecs()

        deriv = dHk_dk(k0, Hr, latVecs)

        expected = complex(-2.8699914595e-06, -1.08160008727e-15)
        eps = 1e-6
        assertEquivComplex(self, deriv[0][0,0], expected, eps, eps)

def _latVecs():
    recipLatVecs = matlib.zeros((3, 3))
    # Reciprocal lattice vectors taken from scf.out, "reciprocal axes".
    # Units there are "2pi / alat"; alat = 5.4235 a.u. = 2.87 Angstrom.
    # Converted here to 1/Angstrom.
    latval = 2.0 * np.pi / (5.4235 * 0.529177)
    recipLatVecs[0, 0] = latval
    recipLatVecs[0, 2] = latval
    recipLatVecs[1, 0] = -latval
    recipLatVecs[1, 1] = latval
    recipLatVecs[2, 1] = -latval
    recipLatVecs[2, 2] = latval

    # latVecs = D^T = 2 pi (R^{-1})^T
    latVecs = 2.0 * np.pi * inv(recipLatVecs).T
    return latVecs

def assertEquivComplex(testcase, val, expected, epsRel, epsAbs):
    testcase.assertTrue((val.real == 0 and expected.real == 0)
                        or (abs(val.real - expected.real) < epsAbs)
                        or (abs((val.real - expected.real)/expected.real) < epsRel))
    testcase.assertTrue((val.imag == 0 and expected.imag == 0)
                        or (abs(val.imag - expected.imag) < epsAbs)
                        or (abs((val.imag - expected.imag)/expected.imag) < epsRel))

if __name__ == "__main__":
    unittest.main()
