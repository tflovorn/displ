from __future__ import division
import argparse
import json
import os.path
import numpy as np
from scipy.interpolate import bisplrep, bisplev
from scipy.integrate import nquad

def fourier(ds, fs, Glat):
    d0s, d1s = [d[0] for d in ds], [d[1] for d in ds]
    tck = bisplrep(d0s, d1s, fs)

    def integrand(d0, d1):
        Glat_dot_rlat = 2.0 * np.pi * (Glat[0] * d0 + Glat[1] * d1)
        coeff = np.exp(-1j * Glat_dot_rlat)
        f = bisplev(d0, d1, tck)
        return coeff * f

    def integrand_re(d0, d1):
        return integrand(d0, d1).real

    def integrand_im(d0, d1):
        return integrand(d0, d1).imag

    result_re, err_re = nquad(integrand_re, [[0.0, 1.0], [0.0, 1.0]])
    result_im, err_re = nquad(integrand_im, [[0.0, 1.0], [0.0, 1.0]])

    return np.complex128(complex(result_re, result_im))

def _main():
    parser = argparse.ArgumentParser("Calculation of gaps",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("gap_file", type=str, help="json file produced by shift_gap")
    args = parser.parse_args()

    with open(args.gap_file, 'r') as fp:
        gap_data = json.load(fp)

    ds = gap_data["_ds"]

    result = {}
    for k, v in gap_data.items():
        if k == "_ds":
            continue

        # G = [1, 0] = b1 = G5. V5 = V1.
        Gs, labels = [[0.0, 0.0], [1.0, 0.0]], ["f0", "f1"]

        result[k] = {}
        for G, label in zip(Gs, labels):
            f = fourier(ds, v, G)
            result[k]["{}_V".format(label)] = abs(f)
            result[k]["{}_psi_deg".format(label)] = np.angle(f, deg=True)

    outpath = "{}_fourier.json".format(os.path.splitext(args.gap_file)[0])
    with open(outpath, 'w') as fp:
        json.dump(result, fp)

if __name__ == "__main__":
    _main()
