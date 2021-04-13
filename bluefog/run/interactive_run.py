# Copyright 2020 Bluefog Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import argparse
import json
import os
import multiprocessing
import signal
import subprocess
import time
from typing import Dict, List

import ipyparallel as ipp
import bluefog
from bluefog.run import env_util, network_util, horovod_driver


BLUEFOG_TIMELINE = 'BLUEFOG_TIMELINE'
BLUEFOG_LOG_LEVEL = 'BLUEFOG_LOG_LEVEL'


def parse_args():

    parser = argparse.ArgumentParser(
        description='Bluefog Interactive Python Runner')

    parser.add_argument('-v', '--version', action="store_true", dest="version",
                        help="Shows bluefog version.")

    subparsers = parser.add_subparsers(dest="action",
                                       help="Start or stop interactive Bluefog cluster. "
                                       "You usually do not need to stop the cluster explicitly.")
    parser_start = subparsers.add_parser(
        'start', help="Start the interactive Bluefog cluster")
    parser_stop = subparsers.add_parser(
        'stop', help="Stop the interactive Bluefog cluster")

    parser_start.add_argument('-np', '--num-proc', action="store", dest="np", required=True,
                              type=int, help="Total number of training processes.")

    parser_start.add_argument('-p', '--ssh-port', action="store", dest="ssh_port",
                              type=int, help="SSH port on all the hosts.")

    parser_start.add_argument('--network-interface', action='store', dest='nic',
                              help='Specify the network interface used for communication.')

    parser_start.add_argument('--use-infiniband', action="store_true", dest="use_infiniband",
                              help='If set, use inifiniband to communication instead of TCP.')

    parser_start.add_argument('--ipython-profile', action="store", dest="profile",
                              type=str, default="bluefog",
                              help="The profile name for ipython environment.")

    parser_start.add_argument('--enable-heartbeat', action="store_true", dest="enable_heartbeat",
                              help='Enable the heartbeat checking service between '
                              'ipcontroller and ipengines.')

    group_hosts_parent = parser_start.add_argument_group('host arguments')
    group_hosts = group_hosts_parent.add_mutually_exclusive_group()
    group_hosts.add_argument('-H', '--hosts', action='store', dest='hosts',
                             help='List of host names and the number of available slots '
                                  'for running processes on each, of the form: <hostname>:<slots> '
                                  '(e.g.: host1:2,host2:4,host3:1 indicating 2 processes can run '
                                  'on host1, 4 on host2, and 1 on host3). If not specified, '
                                  'defaults to using localhost:<np>')
    group_hosts.add_argument('-hostfile', '--hostfile', action='store', dest='hostfile',
                             help='Path to a host file containing the list of host names and '
                                  'the number of available slots. Each line of the file must be '
                                  'of the form: <hostname> slots=<slots>')

    parser_start.add_argument('--verbose', action="store_true", dest="verbose",
                              help="If this flag is set, extra messages will printed.")

    parser_start.add_argument('--extra-mpi-flags', action="store", dest="extra_flags",
                              help='Extra mpi flages you want to pass for mpirun.')

    parser_stop.add_argument('--ipython-profile', action="store", dest="profile",
                              type=str, default="bluefog",
                              help="The profile name for ipython environment.")

    parsed_args = parser.parse_args()
    return parsed_args


def _get_ip_file_dir(profile):
    ip_file_dir = "~/.ipython/profile_{profile}/security".format(
        profile=profile)
    return os.path.expanduser(ip_file_dir)

def _disable_heart_beatcheck(profile):
    config_file = os.path.join(_get_ip_file_dir(
        profile), "..",  "ipengine_config.py")
    try:
        with open(config_file, 'w') as f:
            f.write("c.EngineFactory.max_heartbeat_misses = 0")
        return True
    except:
        return False

def _delete_ipengine_config(profile):
    config_file = os.path.join(_get_ip_file_dir(
        profile), "..",  "ipengine_config.py")
    if os.path.exists(config_file):
        os.remove(config_file)
        print("removed ipengine_config file")

def _wait_engine_file_ready(profile, trial=10):
    engine_file = os.path.join(_get_ip_file_dir(
        profile), "ipcontroller-engine.json")
    file_ready = False
    for _ in range(trial):
        if not os.path.exists(engine_file):
            time.sleep(0.5)
        else:
            file_ready = True
            break
    if not file_ready:
        raise RuntimeError("Cannot find the ipcontroller-engine.json file.")
    return engine_file


def _wait_client_file_ready(profile, trial=10):
    engine_file = os.path.join(_get_ip_file_dir(
        profile), "ipcontroller-client.json")
    file_ready = False
    for _ in range(trial):
        if not os.path.exists(engine_file):
            time.sleep(0.5)
        else:
            file_ready = True
            break
    if not file_ready:
        raise RuntimeError("Cannot find the ipcontroller-client.json file.")
    return engine_file


def _get_ipcontroller_pid(profile):
    pid_file = os.path.join(_get_ip_file_dir(
        profile), "..", "pid", "ipcontroller.pid")
    if not os.path.exists(pid_file):
        return None
    with open(pid_file, 'r') as f:
        try:
            s = f.read().strip()
            pid = int(s)
        except:
            return None
    return pid


# This function must be called after the ipengine is up.
# Note if the ipengine is on multiple machines, the pid
# is the id at the remote machines.
def _get_ipengine_pid(profile):
    rc = ipp.Client(profile=profile)
    engine_pids = rc[:].apply(os.getpid).get_dict()
    return engine_pids


def _write_ipengine_pid(profile):
    engine_pids = _get_ipengine_pid(profile)
    path = _get_ip_file_dir(profile)
    with open(os.path.join(path, "engine_pids.json"), 'w+') as f:
        json.dump(engine_pids, f)


def _get_ipengine_pid_from_file(profile):
    path = _get_ip_file_dir(profile)
    engine_pid_file = os.path.join(path, "engine_pids.json")
    if not os.path.exists(engine_pid_file):
        return None
    with open(engine_pid_file, 'r') as f:
        engine_pids = json.load(f)
    return engine_pids


def _delete_ipengine_pid(profile):
    path = _get_ip_file_dir(profile)
    engine_pid_file = os.path.join(path, "engine_pids.json")
    if os.path.exists(engine_pid_file):
        os.remove(engine_pid_file)


def _maybe_kill_ipcontroller_process(profile):
    "Try to kill the ipcontroller process through read the pid file."
    "Return True if it found process and killed it successfully."
    pid = _get_ipcontroller_pid(profile)
    if pid is None:
        print("Try to kill ipcontroller process but cannot retrieve its pid. "
              "Maybe it is already been stopped.")
        return False
    try:
        for _ in range(2):  # Kill two times
            os.kill(pid, signal.SIGINT)
        return True
    except:
        return False


def _maybe_kill_ipengine_processes(profile):
    "Try to kill the ipengine processes through read the pid file."
    "It only works for local machine case."
    engine_pids = _get_ipengine_pid_from_file(profile)
    if engine_pids is None:
        return
    for _ in range(2):  # Kill two times
        for _, pid in engine_pids.items():
            try:
                os.kill(pid, signal.SIGINT)
            except:
                pass

    _delete_ipengine_config(profile)
    _delete_ipengine_pid(profile)


def interrupt_hanged_processes(profile="bluefog"):
    """ Send the interrupt signal to all hanged workers.

    Args:
        profile (str): The profile name for ipython environment, i.e.
            the --ipython-profile you specified in `ibfrun`. By default,
            this value should be 'bluefog'.

    Note: This function is supported under localhost mode.
    """
    engine_pids = _get_ipengine_pid_from_file(profile)
    if engine_pids is None:
        raise FileNotFoundError("Cannot find pids to interrupt the engines. Note this"
                                "function is supported under localhost mode only")
    timeout = 0.2

    def send_request_to_rc(i):
        rc = ipp.Client(profile=profile)
        rc[i].apply_sync(lambda: 0)

    # Send an empty function to the workers. If it cannot be finished within the
    # {timeout} second, we assume the worker is hanged then send the interrupt
    # signal to it. If finished, do nothing.
    p_list = []
    for i in range(len(engine_pids)):
        p = multiprocessing.Process(target=send_request_to_rc, args=(i,))
        p.start()
        p_list.append(p)
    for i, p in enumerate(p_list):
        p.join(timeout)
        if p.exitcode is None:
            try:
                os.kill(engine_pids[str(i)], signal.SIGINT)
                print(f"send signal to {engine_pids[i]}")
            except:
                pass


def local_machine_launch(args, env: Dict[str, str]):
    ipcontroller_command = "ipcontroller --profile {profile}".format(
        profile=args.profile)
    ipengine_command = (
        "bfrun -np {np} ipengine start --profile {profile}".format(
            np=args.np,
            profile=args.profile,
        )
    )
    # Maybe kill the last time unfinished process.
    if _maybe_kill_ipcontroller_process(args.profile):
        print("Found and killed the unfinished ipcontroller process.")
    subprocess.run('ipcluster nbextension enable --user',
                   shell=True, env=env)
    print("Starting the controller.")
    stdout = None if args.verbose else subprocess.PIPE
    p_controller = subprocess.Popen(ipcontroller_command, shell=True, env=env, stdout=stdout,
                                    stderr=subprocess.STDOUT)
    _wait_engine_file_ready(args.profile)
    print("Starting the engines.")
    if not args.enable_heartbeat:
        disabled = _disable_heart_beatcheck(args.profile)
        print(f"Heartbeat Service Disabled: {disabled}")

    p_engine = subprocess.Popen(ipengine_command, shell=True, env=env)
    engine_pid_done = False
    while not engine_pid_done:
        try:
            time.sleep(2)
            _write_ipengine_pid(args.profile)
            engine_pid_done = True
        except ipp.NoEnginesRegistered as e:
            pass
    while not p_controller.poll() and not p_engine.poll():
        time.sleep(600)

def multiple_machines_launch(args, env: Dict[str, str],
                             hosts_arg: str,
                             all_host_names: List[str],
                             remote_host_names: List[str]):
    common_intfs = set()   # common network interface
    # 1. Check if we can ssh into all remote hosts successfully.
    assert network_util.check_all_hosts_ssh_successful(
        remote_host_names, args.ssh_port)
    if not args.nic:
        # 2. Find the set of common, routed interfaces on all the hosts (remote
        # and local) and specify it in the args. It is expected that the following
        # function will find at least one interface.
        # otherwise, it will raise an exception.
        # So far, we just use horovodrun to do this job since the task are the same.
        local_host_names = set(all_host_names) - set(remote_host_names)
        common_intfs = horovod_driver.driver_fn(all_host_names, local_host_names,
                                                args.ssh_port, args.verbose)
    else:
        common_intfs = [args.nic]

    tcp_intf_arg = '-mca btl_tcp_if_include {common_intfs}'.format(
        common_intfs=','.join(common_intfs)) if common_intfs else ''
    nccl_socket_intf_arg = '-x NCCL_SOCKET_IFNAME={common_intfs}'.format(
        common_intfs=','.join(common_intfs)) if common_intfs else ''

    if args.use_infiniband:
        ib_arg = "-mca btl openib,self"
    else:
        ib_arg = "-mca btl ^openib"

    if args.ssh_port:
        ssh_port_arg = "-mca plm_rsh_args \"-p {ssh_port}\"".format(
            ssh_port=args.ssh_port)
    else:
        ssh_port_arg = ""

    extra_flags = args.extra_flags if args.extra_flags else ''

    ipcontroller_command = "ipcontroller --profile {profile} --ip='*'".format(
        profile=args.profile)

    # Maybe kill the last time unfinished process.
    if _maybe_kill_ipcontroller_process(args.profile):
        print("Found and killed the unfinished ipcontroller process.")
    subprocess.run('ipcluster nbextension enable --user',
                   shell=True, env=env)
    print("Starting the controller.")
    if args.disable_heartbeat:
        _disable_heart_beatcheck(args.profile)
    stdout = None if args.verbose else subprocess.PIPE
    p_controller = subprocess.Popen(ipcontroller_command, shell=True, env=env, stdout=stdout,
                                    stderr=subprocess.STDOUT)
    engine_file = _wait_engine_file_ready(args.profile)
    client_file = _wait_client_file_ready(args.profile)
    # Copy the engine file to all remote hosts
    assert network_util.scp_transmit_file(
        engine_file, remote_host_names, args.ssh_port)
    assert network_util.scp_transmit_file(
        client_file, remote_host_names, args.ssh_port)

    print("Starting the engines.")
    ipengine_command = "ipengine start --profile {profile}".format(
        profile=args.profile,
    )

    # TODO(ybc) Cannot carry the env variable. May encounter:
    # ORCE-TERMINATE AT Data unpack would read past end of buffer:-26 - error grpcomm_direct.c(359)?

    # Use mpirun to start ipengines
    mpi_ipengine_command = (
        'mpirun --allow-run-as-root '
        '-np {num_proc} {hosts_arg} '
        '-bind-to none -map-by slot '
        '-mca pml ob1 {ib_arg} '
        '{ssh_port_arg} {tcp_intf_arg} '
        '{extra_flags} {nccl_socket_intf_arg} '
        '{command}'
        .format(num_proc=args.np,
                hosts_arg=hosts_arg,
                ssh_port_arg=ssh_port_arg,
                tcp_intf_arg=tcp_intf_arg,
                ib_arg=ib_arg,
                nccl_socket_intf_arg=nccl_socket_intf_arg,
                extra_flags=extra_flags,
                env=' '.join('-x %s' % key for key in env.keys()
                             if env_util.is_exportable(key)),
                command=ipengine_command)
    )
    p_engine = subprocess.Popen(mpi_ipengine_command, shell=True, env=env)
    while not p_controller.poll() and not p_engine.poll():
        time.sleep(600)


def main():
    args = parse_args()

    if args.version:
        print(bluefog.__version__)
        exit(0)

    def handler(signum, frame):
        _maybe_kill_ipcontroller_process(args.profile)
        raise KeyboardInterrupt
    signal.signal(signal.SIGINT, handler)

    env = os.environ.copy()
    # No longer needed after using condition variable.
    # env['BLUEFOG_CYCLE_TIME'] = str(20)  # Increase the cycle time

    # action of stop
    if args.action == "stop":
        # TODO(ybc) How to stop it both controller and engines properly?
        # In multiple machine env, we also need to remove engine.json and client.json file.
        ipcluster_stop_command = "ipcluster stop --profile {profile}".format(
            profile=args.profile)
        subprocess.run(ipcluster_stop_command, shell=True,
                       env=env, capture_output=True)
        _maybe_kill_ipcontroller_process(args.profile)
        _maybe_kill_ipengine_processes(args.profile)
        exit(0)

    # action of start
    hosts_arg, all_host_names = network_util.get_hosts_arg_and_hostnames(args)
    remote_host_names = network_util.filter_local_addresses(all_host_names)

    if not env_util.is_open_mpi_installed():
        raise Exception(
            'ibfrun convenience script currently only supports Open MPI.\n'
            'Please Install Open MPI 4.0.0+ and re-install Bluefog.')

    if not env_util.is_ipyparallel_installed():
        raise Exception(
            'ibfrun is based on the ipyparallel package. Please install it in your\n'
            'system like `pip install ipyparallel` first, then run ibfrun again.'
        )

    try:
        if not remote_host_names:
            local_machine_launch(args, env)
        else:
            multiple_machines_launch(args, env, all_host_names=all_host_names,
                                     hosts_arg=hosts_arg,
                                     remote_host_names=remote_host_names)
    except Exception as e:
        print("Fail to launch ibfrun. Error: ", e)
    finally:
        _maybe_kill_ipcontroller_process(args.profile)
        if not remote_host_names:
            time.sleep(1.0)
            _maybe_kill_ipengine_processes(args.profile)


if __name__ == "__main__":
    main()
