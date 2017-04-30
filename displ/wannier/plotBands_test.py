import unittest
import numpy as np
from mkheusler.wannier.plotBands import _interpolateKs

class TestInterpolateKs(unittest.TestCase):
    def test_InterpolateKs(self):
        klist = [(0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (2.0, 2.0, 2.0)]
        fineness = 10
        interp = _interpolateKs(klist, fineness)
        self.assertEqual(len(interp), 1 + fineness * (len(klist) - 1))
        for i in range(len(klist[0])):
            self.assertEqual(interp[0][i], klist[0][i])
            self.assertEqual(interp[-1][i], klist[-1][i])
            
            step = np.subtract(klist[1], klist[0]) / float(fineness)
            self.assertEqual(interp[1][i], (klist[0] + step)[i])

if __name__ == "__main__":
    unittest.main()
