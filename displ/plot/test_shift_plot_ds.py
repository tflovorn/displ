import unittest
from displ.plot.shift_plot_ds import _find_atom_order

class TestOrder(unittest.TestCase):
    def test_find_atom_order(self):
        MoS2_WS2_coords = [[0.0, 0.0, 11.67943], # Mo
                [1.65987, 0.95833, 10.0104], # S
                [1.65987, 0.95833, 13.34846], # S
                [0.0, 0.0, 16.4695], # S
                [0.0, 0.0, 19.82835], # S
                [1.65987, 0.95833, 18.14893]] # W

        assert(_find_atom_order(MoS2_WS2_coords) == ["M", "X1", "X2", "X1p", "X2p", "Mp"])

        WSe2_WSe2_coords = [[0.0, 0.0, 10.0], # Se
                [0.0, 0.0, 13.35885], # Se
                [1.65952, 0.95812, 16.488], # Se
                [1.65952, 0.95812, 19.84685], # Se
                [1.65952, 0.95812, 11.67943], # W
                [0.0, 0.0, 18.16743]] # W

        assert(_find_atom_order(WSe2_WSe2_coords) == ["X1", "X2", "X1p", "X2p", "M", "Mp"])

if __name__ == "__main__":
    unittest.main()
