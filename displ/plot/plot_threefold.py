from __future__ import division
import argparse
import numpy as np
import matplotlib.pyplot as plt

def _main():
    parser = argparse.ArgumentParser("Plot real function with threefold symmetry",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("psi_deg", type=float, help="phase parameter in degrees")
    args = parser.parse_args()

    V = 1.0
    psi = args.psi_deg * np.pi / 180.0

    d_bounds = [-3.0, 3.0]
    Nds = 1000

    D = np.array([[0.5, 0.5],
            [-np.sqrt(3.0)/2.0, np.sqrt(3.0)/2.0]])

    G1 = [1, 1]
    G2 = [-1, 0]
    G3 = [0, -1]

    X, Y, Z = np.zeros([Nds, Nds]), np.zeros([Nds, Nds]), np.zeros([Nds, Nds])

    ds = []
    for d0_index, d0 in enumerate(np.linspace(d_bounds[0], d_bounds[1], Nds)):
        for d1_index, d1 in enumerate(np.linspace(d_bounds[0], d_bounds[1], Nds)):
            this_z = 0.0
            for G in [G1, G2, G3]:
                G_dot_d = 2.0 * np.pi * (d0 * G[0] + d1 * G[1])
                this_z += np.cos(G_dot_d + psi)

            Z[d0_index, d1_index] = this_z

            d_Cart = np.dot(D, np.array([d0, d1]).T)
            X[d0_index, d1_index] = d_Cart[0]
            Y[d0_index, d1_index] = d_Cart[1]

    plt.contourf(X, Y, Z, cmap=plt.get_cmap('viridis'))
    plt.colorbar()
    plt.show()

if __name__ == "__main__":
    _main()
