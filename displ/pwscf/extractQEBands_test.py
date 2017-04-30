import unittest
from mkheusler.pwscf.extractQEBands import extractQEBands

class TestExtractQEBands(unittest.TestCase):
    def test_ExtractQEBandsTa(self):
        nbnd, nks, evlist = extractQEBands("test_data/Ta110_bands_test.dat")
        self.assertEqual(nbnd, 112)
        self.assertEqual(nks, 181)
        self.assertEqual(evlist[0][0], (0.0, 0.0, 0.0))
        self.assertEqual(evlist[0][1][0], -215.189)
        self.assertEqual(evlist[0][1][nbnd-1], 43.570)

if __name__ == "__main__":
    unittest.main()
