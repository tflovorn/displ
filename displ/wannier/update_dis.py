from argparse import ArgumentParser
import os
from displ.pwscf.parseScf import fermi_from_scf
from displ.wannier.wannier_util import global_config
from displ.wannier.build import Update_Disentanglement

def _main():
    parser = ArgumentParser(description="Update disentanglement window in W90 input")
    parser.add_argument('--subdir', type=str, default=None,
            help="Subdirectory under work_base for all job dirs")
    parser.add_argument('prefix', type=str,
            help="Prefix of system to update")
    parser.add_argument('outer_min', type=float,
            help="Distance below E_F to start outer window")
    parser.add_argument('outer_max', type=float,
            help="Distance above E_F to stop outer window")
    parser.add_argument('inner_min', type=float,
            help="Distance below E_F to start inner window")
    parser.add_argument('inner_max', type=float,
            help="Distance above E_F to stop inner window")
    args = parser.parse_args()
    
    gconf = global_config()
    base_path = os.path.expandvars(gconf["work_base"])
    if args.subdir is not None:
        base_path = os.path.join(base_path, args.subdir)

    wandir = os.path.join(base_path, args.prefix, "wannier")
    scf_path = os.path.join(wandir, "scf.out")
    E_Fermi = fermi_from_scf(scf_path)

    outer = [args.outer_min, args.outer_max]
    inner = [args.inner_min, args.inner_max]

    win_path = os.path.join(wandir, "{}.win".format(args.prefix))
    Update_Disentanglement(win_path, E_Fermi, outer, inner)

if __name__ == "__main__":
    _main()
