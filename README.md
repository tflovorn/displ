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

Add to .bashrc:

    export PATH=$HOME/python3.5/bin
    export PYTHONPATH=$HOME/python3.5/lib/python3.5/site-packages:$PYTHONPATH
    export CPATH=$HOME/local:$CPATH
    export LIBRARY_PATH=$HOME/local:$LIBRARY_PATH

Quit ssh session and come back to have these settings take effect.

Install tmd:

    python3 setup.py develop

# Dependencies and installation: local

Uses [ASE](https://wiki.fysik.dtu.dk/ase/index.html) and the [2D materials repository](https://cmr.fysik.dtu.dk/c2dm/c2dm.html).

Install:

    python3 setup.py develop --user
