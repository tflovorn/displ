import os

def build_pw2wan(prefix, soc):
    pw2wan = [" &inputpp"]
    pw2wan.append("   outdir='./',")
    pw2wan.append("   prefix='{}',".format(prefix))
    pw2wan.append("   write_mmn=.true.,")
    pw2wan.append("   write_amn=.true.,")

    if soc:
        pw2wan.append("   write_spn=.true.,")
    else:
        pw2wan.append("   write_spn=.false.,")

    # TODO - support collinear spin-polarized case

    pw2wan.append("   write_unk=.false.,")
    pw2wan.append("   seedname='{}'".format(prefix))
    pw2wan.append(" /\n")

    return "\n".join(pw2wan)

def build_bands(prefix):
    '''Construct a string which gives the QE input file for bands postprocessing
    for the given material.
    '''
    bands = [" &bands"]
    bands.append("   prefix='{}',".format(prefix))
    bands.append("   outdir='./',")
    bands.append("   filband='{}_bands.dat'".format(prefix))
    bands.append(" /\n")

    return "\n".join(bands)

def get_cell(ase_system, latconst):
    axes_np = ase_system.cell / latconst
    axes = [axes_np[0, :], axes_np[1, :], axes_np[2, :]]

    syms = ase_system.get_chemical_symbols()
    pos = ase_system.get_scaled_positions() # positions in lattice coords
    
    return axes, list(zip(syms, pos))

def build_qe(ase_system, prefix, calc_type, config):
    '''Construct a string which gives the QE input file for the specified
    calculation.

    calc_type: one of 'scf', 'nscf', 'bands'.
    '''
    calc_vals = ["scf", "nscf", "bands"]
    if calc_type not in calc_vals:
        raise ValueError("Unsupported calc_type " + calc_type)

    pseudo_dir = os.path.expandvars(config["pseudo_dir"])
    conv_thr = config["conv_thr"][calc_type]
    axes, latpos = get_cell(ase_system, config["latconst"])

    # Make strings representing each section of the PW input.
    # These may be None.
    blocks = [("control", _control(calc_type, pseudo_dir, prefix)),
            ("system", _system(ase_system, calc_type, config)),
            ("electrons", _electrons(conv_thr)),
            ("atomic_species", _atomic_species(config["pseudo"], config["weight"])),
            ("cell_parameters", _cell_parameters(axes)),
            ("atomic_positions",  _atomic_positions(latpos)),
            ("k_points", _k_points(calc_type, config))]

    # Join the sections with newlines, ignoring any None values.
    pw_input = _join(blocks)

    return pw_input

def _join(xs):
    ret = None
    for label, x in xs:
        if ret == None and x != None:
            ret = x
        elif x != None:
            ret = "\n".join([ret, x])
    return ret + "\n"

def _control(calc_type, pseudo_dir, prefix):
    nl = [" &control"]
    nl.append("    calculation='{}',".format(calc_type))
    nl.append("    restart_mode='from_scratch',")
    nl.append("    disk_io='low',")
    nl.append("    wf_collect=.true.,")
    nl.append("    pseudo_dir='{}',".format(pseudo_dir))
    nl.append("    outdir='./',")
    nl.append("    prefix='{}'".format(prefix))
    nl.append(" /")
    return "\n".join(nl)

def _system(ase_system, calc_type, config):
    bohr_in_A = 1.889726164
    latconst_bohr = bohr_in_A * config["latconst"]

    num_atoms, num_atom_types = _atom_types(ase_system)

    nl = [" &system"]
    nl.append("    ibrav=0,celldm(1)={},nat={},ntyp={},".format(str(latconst_bohr),
        str(num_atoms), str(num_atom_types)))
    nl.append("    ecutwfc={},".format(str(config["ecutwfc"])))
    nl.append("    ecutrho={},".format(str(config["ecutrho"])))

    if config["soc"]:
        nl.append("    noncolin=.true.,")
        nl.append("    lspinorb=.true.,")

    if calc_type == 'scf':
        nl.append("    occupations='tetrahedra'")
    else:
        nl.append("    nosym=.true.,")
        nl.append("    nbnd={},".format(config["num_bands"]))
        nl.append("    occupations='smearing',smearing='cold',degauss={}".format(str(config["degauss"])))

    nl.append(" /")    
    return "\n".join(nl)

def _atom_types(ase_system):
    syms = ase_system.get_chemical_symbols()
    num_atoms = len(syms)
    atom_types = set()
    for sym in syms:
        atom_types.add(sym)

    num_atom_types = len(atom_types)
    return num_atoms, num_atom_types

def _electrons(conv_thr):
    nl = [" &electrons"]
    nl.append("    startingwfc='atomic+random',")
    nl.append("    diagonalization='david',")
    nl.append("    conv_thr={}".format(str(conv_thr)))
    nl.append(" /")
    return "\n".join(nl)

def _atomic_species(pseudo, weight):
    card = ["ATOMIC_SPECIES"]
    for k, v in pseudo.items():
        card.append(" {} {} {}".format(k, weight[k], v))
    return "\n".join(card)

def _cell_parameters(axes):
    card = ["CELL_PARAMETERS alat"]
    for ax in range(3):
        ax, ay, az = str(axes[ax][0]), str(axes[ax][1]), str(axes[ax][2])
        card.append(" {}    {}    {}".format(ax, ay, az))
    return "\n".join(card)

def _atomic_positions(pos):
    card = ["ATOMIC_POSITIONS crystal"]
    for atom, p in pos:
        pa, pb, pc = str(p[0]), str(p[1]), str(p[2])
        card.append(" {} {} {} {}".format(atom, pa, pb, pc))
    return "\n".join(card)

def _k_points(calc_type, config):
    if calc_type == 'scf':
        Nk1, Nk2, Nk3 = config["Nk"]["scf"]

        card = ["K_POINTS automatic"]
        card.append("{} {} {} 0 0 0".format(Nk1, Nk2, Nk3))
        return "\n".join(card)
    elif calc_type == 'nscf':
        Nk1, Nk2, Nk3 = config["Nk"]["nscf"]
        Nks = Nk1*Nk2*Nk3
        weight = 1.0/Nks
        nscf_klist = nscf_ks(Nk1, Nk2, Nk3)

        card = ["K_POINTS crystal"]
        card.append("{}".format(Nks))
        for k in nscf_klist:
            card.append("    {} {} {} {}".format(str(k[0]), str(k[1]), str(k[2]), str(weight)))

        return "\n".join(card)
    else:
        # calc_type == 'bands'
        num_points = len(config["band_path"])
        Nkband = config["Nk"]["bands"]

        card = ["K_POINTS crystal_b"]
        card.append("{}".format(num_points))
        for k in config["band_path"]:
            card.append(" {} {} {} {}".format(str(k[0]), str(k[1]), str(k[2]), Nkband))

        return "\n".join(card)

def nscf_ks(Nk1, Nk2, Nk3):
    '''Returns a list of [k1, k2, k3] values giving the ks to be included
    in a nscf/Wannier run.
    '''
    ks = []
    for i in range(Nk1):
        for j in range(Nk2):
            for k in range(Nk3):
                ks.append([float(i)/float(Nk1), float(j)/float(Nk2), float(k)/float(Nk3)])

    return ks
