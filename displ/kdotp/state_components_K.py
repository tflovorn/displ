from __future__ import division
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from displ.build.build import _get_work
from displ.pwscf.parseScf import fermi_from_scf
from displ.wannier.extractHr import extractHr
from displ.wannier.bands import Hk_recip
from displ.kdotp.model_weights_K import top_valence_indices
from displ.kdotp.separability_K import get_layer_orbitals

def _weight(state, orb1, orb2, fac):
    '''Construct the weight corresponding to the state |orb1> + fac * |orb2>.
    '''
    assert(abs(fac) == 1.0)

    total = (1.0/np.sqrt(2.0)) * (state[[orb1, 0]] + fac.conjugate() * state[[orb2, 0]])
    return np.linalg.norm(total)**2

def get_state_weights(layer_orbitals, state):
    weights = {"(+, up)": [], "(-, up)": [], "(+, down)": [], "(-, down)": []}

    for z_orb in layer_orbitals:
        px_up_bot, px_down_bot, py_up_bot, py_down_bot = [z_orb[i] for i in range(2, 6)]
        px_up_top, px_down_top, py_up_top, py_down_top = [z_orb[i] for i in range(8, 12)]
        dx2y2_up, dx2y2_down, dxy_up, dxy_down = [z_orb[i] for i in range(18, 22)]

        weights["(+, up)"].append(_weight(state, px_up_bot, py_up_bot, 1j)
                + _weight(state, px_up_top, py_up_top, 1j)
                + _weight(state, dx2y2_up, dxy_up, 1j))

        weights["(-, up)"].append(_weight(state, px_up_bot, py_up_bot, -1j)
                + _weight(state, px_up_top, py_up_top, -1j)
                + _weight(state, dx2y2_up, dxy_up, -1j))

        weights["(+, down)"].append(_weight(state, px_down_bot, py_down_bot, 1j)
                + _weight(state, px_down_top, py_down_top, 1j)
                + _weight(state, dx2y2_down, dxy_down, 1j))

        weights["(-, down)"].append(_weight(state, px_down_bot, py_down_bot, -1j)
                + _weight(state, px_down_top, py_down_top, -1j)
                + _weight(state, dx2y2_down, dxy_down, -1j))

    return weights

def plot_orbital_weights_K(subdir, prefix):
    num_layers = 3
    assert(num_layers == 3) # != 3 unimplemented

    work = _get_work(subdir, prefix)
    wannier_dir = os.path.join(work, "wannier")
    scf_path = os.path.join(wannier_dir, "scf.out")
    wout_path = os.path.join(wannier_dir, "{}.wout".format(prefix))

    E_F = fermi_from_scf(scf_path)
    layer_orbitals = get_layer_orbitals(wout_path, num_layers)

    Hr_path = os.path.join(wannier_dir, "{}_hr.dat".format(prefix))
    Hr = extractHr(Hr_path)

    K_lat = np.array([1/3, 1/3, 0.0])
    H_TB_K = Hk_recip(K_lat, Hr)
    Es, U = np.linalg.eigh(H_TB_K)
    top = top_valence_indices(E_F, 2*num_layers, Es)

    eigenstate_weights = [get_state_weights(layer_orbitals, U[:, [t]]) for t in top]

    print(eigenstate_weights)

def _main():
    parser = argparse.ArgumentParser("Plot orbital contributions at K",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("prefix", type=str,
            help="Prefix for calculation")
    parser.add_argument("--subdir", type=str, default=None,
            help="Subdirectory under work_base where calculation was run")
    args = parser.parse_args()

    plot_orbital_weights_K(args.subdir, args.prefix)

if __name__ == "__main__":
    _main()
