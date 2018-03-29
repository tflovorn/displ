# Overview

`displ/build/build.py` constructs input for Quantum Espresso and Wannier90 for DFT calculations and associated tight-binding model extraction for a TMD multilayer in the presence of a transverse electric field. See --help for options.

`global_config.json` field `work_base` specifies the directory under which QE/W90 inputs will be created or are assumed to be present. The --subdir option specifies a directory under this associated with a particular calculation.

`displ/plot/bands.py` plots band structure and associated eigenvector weights.

`displ/kdotp/` contains programs for extracting k dot p models from the tight-binding models and associated calculations.
* `effective_valence_K.py` and `effective_valence_Gamma.py` generate the k dot p models for K and Gamma points.
* `effective_dielectric.py` calculates the effective dielectric constant.
* `efield.py`, `dft_efield.py`, and `kdotp_efield.py` calculate the response of the system to transverse electric field in the presence of holes.

# Dependencies and installation: Lonestar5

Install python3:

    cd ~
    wget https://www.python.org/ftp/python/3.5.1/Python-3.5.1.tar.xz
    unxz Python-3.5.1.tar.xz
    tar -xvf Python-3.5.1.tar
    cd Python-3.5.1
    ./configure --prefix=$HOME/python3.5
    make
    make install

Add to ~/.bashrc:

    export PATH=$HOME/python3.5/bin:$PATH
    export PYTHONPATH=$HOME/python3.5/lib/python3.5/site-packages:$PYTHONPATH
    export CPATH=$HOME/local:$CPATH
    export LIBRARY_PATH=$HOME/local:$LIBRARY_PATH

Quit ssh session and come back to have these settings take effect.

Install displ:

    python3 setup.py develop

# Dependencies and installation: local

Uses [ASE](https://wiki.fysik.dtu.dk/ase/index.html) and the [2D materials repository](https://cmr.fysik.dtu.dk/c2dm/c2dm.html).

Install displ:

    python3 setup.py develop --user
