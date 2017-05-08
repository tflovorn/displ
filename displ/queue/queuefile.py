import os
import stat
from displ.queue.queue_util import global_config, _base_dir

def write_queuefile(config):
    machine = config["machine"]

    gconf = global_config()

    if machine == "ls5":
        _write_queuefile_ls5(config)
    else:
        raise ValueError("Unrecognized config['machine'] value")

def write_launcherfiles(config):
    machine = config["machine"]
    gconf = global_config()

    if machine == "ls5":
        _write_launcherfiles_ls5(config)
    else:
        raise ValueError("Unrecognized config['machine'] value")

def write_job_group_files(config, prefix_groups):
    machine = config["machine"]
    gconf = global_config()

    if machine == "ls5":
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

def _write_queuefile_ls5(config):
    prefix = config["prefix"]

    qf = ["#!/bin/bash"]
    if config["calc"] == "wan_setup":
        nk = str(config["nodes"])
        qf.append("ibrun tacc_affinity pw.x -nk {} -input {}.scf.in > scf.out".format(nk, prefix))
        qf.append("cd ..")
        qf.append("cp -r wannier/* bands")
        qf.append("cd bands")
        qf.append("ibrun tacc_affinity pw.x -nk {} -input {}.bands.in > bands.out".format(nk, prefix))
        if config["wannier"]:
            qf.append("cd ../wannier")
            qf.append("ibrun tacc_affinity pw.x -nk {} -input {}.nscf.in > nscf.out".format(nk, prefix))
    elif config["calc"] == "pw_post":
        qf.append("cd ../bands")
        qf.append("ibrun tacc_affinity {} -input {}.bands_post.in > bands_post.out".format(config["qe_bands"], prefix))
        qf.append("rm {}.wfc*".format(prefix))
        if config["wannier"]:
            qf.append("cd ../wannier")
            qf.append("wannier90.x -pp {}".format(prefix))
            qf.append("ibrun tacc_affinity pw2wannier90.x -input {}.pw2wan.in > pw2wan.out".format(prefix))
            # Clean up redundant wfc extracted from .save directory
            qf.append("rm {}.wfc*".format(prefix))
    elif config["calc"] == "wan_run":
        wan_dir = os.path.join(config["base_path"], config["prefix"], "wannier")
        update_dis = os.path.join(_base_dir(), "displ", "wannier", "update_dis.py")
        outer_min, outer_max = str(config["outer_min"]), str(config["outer_max"])
        inner_min, inner_max = str(config["inner_min"]), str(config["inner_max"])
        py_str = "python3 '{}' '{}' {} {} {} {}".format(update_dis, prefix, outer_min, outer_max, inner_min, inner_max)
        if config["subdir"] is not None:
            py_str = py_str + " --subdir {}".format(config["subdir"])

        qf.append(py_str)
        qf.append("cd {}".format(wan_dir))
        qf.append("wannier90.x {}".format(prefix))
    elif config["calc"] == "bands_only":
        nk = str(config["nodes"])
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

def _write_launcherfiles_ls5(config):
    _write_launcher_qf_ls5(config)
    _write_launcher_job_ls5(config)

def _write_launcher_qf_ls5(config):
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

def _write_launcher_job_ls5(config):
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
