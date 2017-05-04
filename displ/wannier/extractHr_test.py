import unittest
from displ.wannier.extractHr import extractHr

class TestExtractHr(unittest.TestCase):
    def test_extractHrFe(self):
        Hr = extractHr("test_data/Fe_hr_test.dat")
        elem = Hr[(-6, 2, -4)]
        self.assertEqual(elem[0][0, 0], complex(0.004554, 0.0))
        self.assertEqual(elem[1], 4)

if __name__ == "__main__":
    unittest.main()
