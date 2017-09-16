from __future__ import division
import argparse
import numpy as np
from displ.kdotp.dft_efield import get_prefixes
from displ.kdotp.effective_valence_K import make_effective_Hamiltonian_K, get_layer_basis_from_dm_K
from displ.kdotp.effective_valence_Gamma import make_effective_Hamiltonian_Gamma

def effective_dielectric(layer_energies, d_nm, E_V_nm):
    num = E_V_nm * d_nm
    diff_mid_bot = layer_energies[1] - layer_energies[0]
    diff_top_mid = layer_energies[2] - layer_energies[1]
    diff_top_bot = layer_energies[2] - layer_energies[0]

    return [num / diff_mid_bot, num / diff_top_mid, 2 * num / diff_top_bot]

def _main():
    np.set_printoptions(threshold=np.inf)

    parser = argparse.ArgumentParser("Calculate effective dielectric constant",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--global_prefix", type=str, default="WSe2_WSe2_WSe2",
            help="Prefix used for all calculations")
    parser.add_argument("--subdir", type=str, default=None,
            help="Subdirectory under work_base where calculation was run")
    args = parser.parse_args()

    E_V_nms, prefixes = get_prefixes(args.global_prefix, args.subdir)

    d_A = 6.488 # Angstrom
    d_nm = d_A / 10.0

    eps_eff_K_high, eps_eff_K_low, eps_eff_Gamma = [], [], []

    for E_V_nm, prefix in zip(E_V_nms, prefixes):
        if E_V_nm == 0.0:
            continue

        print("E_V_nm = ", E_V_nm)

        K_lat = np.array([1/3, 1/3, 0.0])
        H0_tot_K, ps_tot_K, mstar_inv_tot_K = make_effective_Hamiltonian_K(K_lat, args.subdir,
                prefix, get_layer_basis_from_dm_K, verbose=False)

        print("on-site K:", [H0_tot_K[i, i] for i in range(6)])

        onsite_K_high = [H0_tot_K[1, 1].real, H0_tot_K[3, 3].real, H0_tot_K[5, 5].real]
        onsite_K_low = [H0_tot_K[0, 0].real, H0_tot_K[2, 2].real, H0_tot_K[4, 4].real]

        H0_tot_Gamma, ps_tot_Gamma, mstar_inv_tot_Gamma = make_effective_Hamiltonian_Gamma(args.subdir,
                prefix, top_two_only=False, verbose=False)

        print("on-site Gamma:", [H0_tot_Gamma[i, i] for i in range(6)])

        onsite_Gamma = [H0_tot_Gamma[0, 0].real, H0_tot_Gamma[2, 2].real, H0_tot_Gamma[4, 4].real]

        eps_eff_K_high.append((E_V_nm, effective_dielectric(onsite_K_high, d_nm, E_V_nm)))
        eps_eff_K_low.append((E_V_nm, effective_dielectric(onsite_K_low, d_nm, E_V_nm)))
        eps_eff_Gamma.append((E_V_nm, effective_dielectric(onsite_Gamma, d_nm, E_V_nm)))

    print("effective dielectric constant: K, high states; [middle->bottom, top->middle, top->bottom]")
    print(eps_eff_K_high)

    print("effective dielectric constant: K, low states")
    print(eps_eff_K_low)

    print("effective dielectric constant: Gamma")
    print(eps_eff_Gamma)

    print("avg dielectric constant, E >= 0.5 V/nm: top->bottom, K high, K low, Gamma")
    def filtered_avg(eps_vals, E_threshold, inner_index):
        filtered = list(filter(lambda Ev: Ev[0] >= E_threshold, eps_vals))
        return sum([eps[1][inner_index] for eps in filtered]) / len(filtered)

    all_eps_vals = [eps_eff_K_high, eps_eff_K_low, eps_eff_Gamma]
    print([filtered_avg(eps_vals, 0.5, 2) for eps_vals in all_eps_vals])

    print("avg dielectric constant, E >= 0.5 V/nm: top->mid, K high, K low, Gamma")
    print([filtered_avg(eps_vals, 0.5, 1) for eps_vals in all_eps_vals])

    print("avg dielectric constant, E >= 0.5 V/nm: mid->bottom, K high, K low, Gamma")
    print([filtered_avg(eps_vals, 0.5, 0) for eps_vals in all_eps_vals])

if __name__ == "__main__":
    _main()
