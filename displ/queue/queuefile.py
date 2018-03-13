import os
import stat
from displ.queue.queue_util import global_config, _base_dir

def mpi_procs_per_node(machine):
    if machine == "ls5":
        return 24
    elif machine == "stampede2":
        return 68
    else:
        raise ValueError("unrecognized machine value")

def write_queuefile(config):
    machine = config["machine"]

    gconf = global_config()

    if machine in ["ls5", "stampede2"]:
        _write_queuefile_tacc(config)
    else:
        raise ValueError("Unrecognized config['machine'] value")

def write_launcherfiles(config):
    machine = config["machine"]
    gconf = global_config()

    if machine in ["ls5", "stampede2"]:
        _write_launcherfiles_tacc(config)
    else:
        raise ValueError("Unrecognized config['machine'] value")

def write_job_group_files(config, prefix_groups):
    machine = config["machine"]
    gconf = global_config()

    if machine in ["ls5", "stampede2"]:
        for group_id, group in enumerate(prefix_groups):
            _write_group_queuefile(config, group, group_id)
    else:
        raise ValueError("Unrecognized config['machine'] value")

def _write_group_queuefile(config, group, group_id):
    duration = _ls_format_duration(config["hours"], config["minutes"])
    prefix = config["prefix"]

    qf = ["#!/bin/bash"]
    qf.append("#SBATCH -p {}".format(config["queue"]))
    qf.append("#SBATCH -J {}".format(prefix))
    #qf.append("#SBATCH -o {}.out".format(prefix))
    qf.append("#SBATCH -e {}.err".format(prefix))
    qf.append("#SBATCH -t {}".format(duration))
    qf.append("#SBATCH -N {}".format(str(config["nodes"])))
    qf.append("#SBATCH -n {}".format(str(config["cores"])))
    qf.append("#SBATCH -A {}".format(config["project"]))
    qf.append("")
    qf.append("export OMP_NUM_THREADS=1")

    for prefix in group:
        wan_path = os.path.join(config["base_path"], prefix, "wannier")
        qf.append("cd {}".format(wan_path))
        qf.append("./run_{}".format(config["calc"]))

    all_groups_path = os.path.join(config["base_path"], config["global_prefix"])
    qf_path = "{}_{}_{}".format(all_groups_path, config["calc"], str(group_id))

    with open(qf_path, 'w') as fp:
        qf_str = "\n".join(qf)
        fp.write(qf_str)

    os.chmod(qf_path, stat.S_IRWXU)

def get_qf_path(config):
    qf_name = "run_{}".format(config["calc"])
    qf_path = os.path.join(config["base_path"], config["prefix"], "wannier", qf_name)

    return qf_path

def _ls_format_duration(hours, minutes):
    hstr = str(hours)
    if minutes < 10:
        mstr = "0{}".format(str(minutes))
    else:
        mstr = str(minutes)

    return "{}:{}:00".format(hstr, mstr)

def _write_queuefile_tacc(config):
    prefix = config["prefix"]

    if config["machine"] == "stampede2":
        nk = str(4*config["nodes"])
    else:
        nk = str(config["nodes"])

    wan_dir = os.path.join(config["base_path"], prefix, "wannier")
    bands_dir = os.path.join(config["base_path"], prefix, "bands")

    qf = ["#!/bin/bash"]
    if config["calc"] == "wan_setup":
        if config["relax"]:
            # Do relax run.
            qf.append("ibrun tacc_affinity pw.x -nk {} -input {}.relax.in > relax.out".format(nk, prefix))
            # Update coordinates to relaxed values.
            set_pw_input = os.path.join(_base_dir(), "displ", "pwscf", "set_pw_input.py")
            relax_out_path = os.path.join(wan_dir, "relax.out")
            scf_in_path = os.path.join(wan_dir, "{}.scf.in".format(prefix))
            nscf_in_path = os.path.join(wan_dir, "{}.nscf.in".format(prefix))
            bands_in_path = os.path.join(bands_dir, "{}.bands.in".format(prefix))

            for infile_path in [scf_in_path, nscf_in_path, bands_in_path]:
                qf.append("python3 '{}' '{}' '{}'".format(set_pw_input, relax_out_path, infile_path))

        # Do scf run and copy charge density result to bands directory.
        qf.append("ibrun tacc_affinity pw.x -nk {} -input {}.scf.in > scf.out".format(nk, prefix))
        qf.append("cd ..")
        qf.append("cp -r wannier/* bands")
        qf.append("cd bands")
        # Do bands run.
        qf.append("ibrun tacc_affinity pw.x -nk {} -input {}.bands.in > bands.out".format(nk, prefix))

        if config["wannier"]:
            # Do nscf run for Wannier90 input.
            qf.append("cd ..")
            qf.append("mkdir -p scf/{}.save".format(prefix))
            qf.append("cp wannier/{}.save/data-file.xml scf/{}.save/data-file.xml".format(prefix, prefix))
            qf.append("cd wannier")
            qf.append("ibrun tacc_affinity pw.x -nk {} -input {}.nscf.in > nscf.out".format(nk, prefix))
    elif config["calc"] == "pw_post":
        # Do bands post-processing.
        qf.append("cd ../bands")
        qf.append("ibrun tacc_affinity {} -input {}.bands_post.in > bands_post.out".format(config["qe_bands"], prefix))
        qf.append("rm {}.wfc*".format(prefix))
        if config["wannier"]:
            if config["relax"]:
                # Update coordinates to relaxed values.
                set_relaxed_win = os.path.join(_base_dir(), "displ", "wannier", "update_relax.py")
                relax_out_path = os.path.join(wan_dir, "relax.out")
                win_path = os.path.join(wan_dir, "{}.win".format(prefix))
                qf.append("python3 '{}' '{}' '{}'".format(set_relaxed_win, relax_out_path, win_path))
            # Do pw2annier90.
            qf.append("cd ../wannier")
            qf.append("wannier90.x -pp {}".format(prefix))
            qf.append("ibrun tacc_affinity pw2wannier90.x -input {}.pw2wan.in > pw2wan.out".format(prefix))
            # Clean up redundant wfc extracted from .save directory
            qf.append("rm {}.wfc*".format(prefix))
    elif config["calc"] == "wan_run":
        # Set inner and outer windows in Wannier90 input appropriately:
        # they are given in input file in absolute energies; we specify them relative to the
        # Fermi energy.

        update_dis = os.path.join(_base_dir(), "displ", "wannier", "update_dis.py")
        outer_min, outer_max = str(config["outer_min"]), str(config["outer_max"])
        inner_min, inner_max = str(config["inner_min"]), str(config["inner_max"])
        py_str = "python3 '{}' '{}' {} {} {} {}".format(update_dis, prefix, outer_min, outer_max, inner_min, inner_max)
        if config["subdir"] is not None:
            py_str = py_str + " --subdir {}".format(config["subdir"])

        # Run Wannier90.
        qf.append(py_str)
        qf.append("cd {}".format(wan_dir))
        qf.append("wannier90.x {}".format(prefix))
    elif config["calc"] == "bands_only":
        qf.append("ibrun tacc_affinity pw.x -nk {} -input {}.scf.in > scf.out".format(nk, prefix))
        qf.append("cd ..")
        qf.append("cp -r wannier/* bands")
        qf.append("cd bands")
        qf.append("ibrun tacc_affinity pw.x -nk {} -input {}.bands.in > bands.out".format(nk, prefix))
        qf.append("ibrun tacc_affinity {} -input {}.bands_post.in > bands_post.out".format(config["qe_bands"], prefix))
        qf.append("rm {}.wfc*".format(prefix))
        qf.append("rm -r {}.save".format(prefix))
    else:
        raise ValueError("unrecognized config['calc'] ('wan_setup' and 'wan_run' supported)")

    qf_path = get_qf_path(config)

    with open(qf_path, 'w') as fp:
        qf_str = "\n".join(qf) + "\n"
        fp.write(qf_str)

    os.chmod(qf_path, stat.S_IRWXU)

def _write_launcherfiles_tacc(config):
    _write_launcher_qf_tacc(config)
    _write_launcher_job_tacc(config)

def _write_launcher_qf_tacc(config):
    duration = _ls_format_duration(config["hours"], config["minutes"])
    prefix = config["global_prefix"]

    qf = ["#!/bin/bash"]
    qf.append("#SBATCH -p {}".format(config["queue"]))
    qf.append("#SBATCH -J {}".format(prefix))
    qf.append("#SBATCH -o {}.out".format(prefix))
    qf.append("#SBATCH -e {}.err".format(prefix))
    qf.append("#SBATCH -t {}".format(duration))
    qf.append("#SBATCH -N {}".format(str(config["nodes"])))
    qf.append("#SBATCH -n {}".format(str(config["cores"])))
    qf.append("#SBATCH -A {}".format(config["project"]))
    qf.append("")
    qf.append("export OMP_NUM_THREADS=1")
    qf.append("export LAUNCHER_PLUGIN_DIR=$LAUNCHER_DIR/plugins")
    qf.append("export LAUNCHER_RMI=SLURM")
    qf.append("export LAUNCHER_JOB_FILE={}_jobfile".format(prefix))
    qf.append("")
    qf.append("$LAUNCHER_DIR/paramrun")

    qf_prefix = os.path.join(config["base_path"], config["global_prefix"])
    qf_path = "{}_launcher".format(qf_prefix)

    with open(qf_path, 'w') as fp:
        qf_str = "\n".join(qf) + "\n"
        fp.write(qf_str)

    os.chmod(qf_path, stat.S_IRWXU)

def _write_launcher_job_tacc(config):
    prefix_list = config["prefix_list"]
    global_prefix = config["global_prefix"]

    jf = []
    if config["calc"] == "wan_run":
        for prefix in prefix_list:
            jf.append("{}/wannier/run_wan_run".format(prefix))
    else:
        raise ValueError("unrecognized config['calc']")

    jf_path = os.path.join(config["base_path"], "{}_jobfile".format(global_prefix))

    with open(jf_path, 'w') as fp:
        jf_str = "\n".join(jf) + "\n"
        fp.write(jf_str)

    os.chmod(jf_path, stat.S_IRWXU)
