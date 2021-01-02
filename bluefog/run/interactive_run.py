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
import os
import re
import signal
import shlex
import socket
import subprocess
import sys
import time
import traceback
from typing import Dict, List

import psutil
import bluefog
from bluefog.run import env_util, network_util, horovod_driver


BLUEFOG_TIMELINE = 'BLUEFOG_TIMELINE'
BLUEFOG_LOG_LEVEL = 'BLUEFOG_LOG_LEVEL'


def parse_args():

    parser = argparse.ArgumentParser(
        description='Bluefog Interactive Python Runner')

    parser.add_argument('-v', '--version', action="store_true", dest="version",
                        help="Shows bluefog version.")

    parser.add_argument('-np', '--num-proc', action="store", dest="np",
                        type=int, help="Total number of training processes.")

    parser.add_argument('-p', '--ssh-port', action="store", dest="ssh_port",
                        type=int, help="SSH port on all the hosts.")

    parser.add_argument('--network-interface', action='store', dest='nic',
                        help='Specify the network interface used for communication.')

    parser.add_argument('--use-infiniband', action="store_true", dest="use_infiniband",
                        help='If set, use inifiniband to communication instead of TCP.')

    parser.add_argument('--ipython-profile', action="store", dest="profile",
                        type=str, default="bluefog",
                        help="The profile name for ipython environment.")

    group_hosts_parent = parser.add_argument_group('host arguments')
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

    parser.add_argument('--verbose', action="store_true", dest="verbose",
                        help="If this flag is set, extra messages will printed.")

    parser.add_argument('--extra-mpi-flags', action="store", dest="extra_flags",
                        help='Extra mpi flages you want to pass for mpirun.')

    parser.add_argument('command', nargs=argparse.REMAINDER,
                        help="Command to be executed.")

    parsed_args = parser.parse_args()

    if not parsed_args.version and not parsed_args.np:
        parser.error('argument -np/--num-proc is required')

    return parsed_args


def _get_ip_file_dir(profile):
    ip_file_dir = "~/.ipython/profile_{profile}/security".format(
        profile=profile)
    return os.path.expanduser(ip_file_dir)


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


def _maybe_kill_ipcontroller_process(profile):
    "Try to kill the ipcontroller process through read the pid file."
    "Return True if it found process and killed it successfully."
    pid = _get_ipcontroller_pid(profile)
    if pid is None:
        print("Try to kill ipcontroller process but cannot retrieve its pid. "
              "Maybe it is already been stopped.")
        return False
    try:
        os.kill(pid, signal.SIGINT)
        return True
    except:
        return False


def local_machine_launch(args, env: Dict[str, str], command: str, ipcluster_stop_command: str):
    ipcontroller_command = "ipcontroller --profile {profile}".format(
        profile=args.profile)
    ipengine_command = (
        "bfrun -np {np} ipengine {command} --profile {profile}".format(
            np=args.np,
            profile=args.profile,
            command=command
        )
    )
    if command == 'start':
        # Maybe kill the last time unfinished process.
        if _maybe_kill_ipcontroller_process(args.profile):
            print("Found and killed the unfinished ipcontroller process.")
        subprocess.run('ipcluster nbextension enable --user',
                       shell=True, env=env)
        print(ipcontroller_command)
        subprocess.Popen(ipcontroller_command, shell=True, env=env)
        _wait_engine_file_ready(args.profile)
        print(ipengine_command)
        subprocess.run(ipengine_command, shell=True,
                       env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    else:  # stop
        subprocess.run(ipcluster_stop_command, shell=True,
                       env=env, capture_output=True)
        _maybe_kill_ipcontroller_process(args.profile)


def multiple_machines_launch(args, env: Dict[str, str],
                             hosts_arg: str,
                             all_host_names: List[str],
                             remote_host_names: List[str], command: str,
                             ipcluster_stop_command: str):
    common_intfs = set()   # common network interface
    # 1. Check if we can ssh into all remote hosts successfully.
    assert network_util.check_all_hosts_ssh_successful(remote_host_names, args.ssh_port)
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

    if command == 'start':
        # Maybe kill the last time unfinished process.
        if _maybe_kill_ipcontroller_process(args.profile):
            print("Found and killed the unfinished ipcontroller process.")
        subprocess.run('ipcluster nbextension enable --user', shell=True, env=env)
        print(ipcontroller_command)
        subprocess.Popen(ipcontroller_command, shell=True, env=env)
        engine_file = _wait_engine_file_ready(args.profile)
        client_file = _wait_client_file_ready(args.profile)
        # Copy the engine file to all remote hosts
        assert network_util.scp_transmit_file(engine_file, remote_host_names, args.ssh_port)
        assert network_util.scp_transmit_file(client_file, remote_host_names, args.ssh_port)

        ipengine_command = "ipengine {command} --profile {profile}".format(
            profile=args.profile,
            command=command
        )

        # TODO(ybc) Cannot carry the env variable. May encounter:
        # ORCE-TERMINATE AT Data unpack would read past end of buffer:-26 - error grpcomm_direct.c(359)?
        
        # Use mpirun to start ipengines
        mpi_ipengine_command = (
            'mpirun --allow-run-as-root '
            '-np {num_proc} {hosts_arg} '
            '-bind-to none -map-by slot '
            '-mca pml ob1 '
            '{ssh_port_arg} {tcp_intf_arg} '
            '{extra_flags} {nccl_socket_intf_arg} '
            '{command}'
            .format(num_proc=args.np,
                    hosts_arg=hosts_arg,
                    ssh_port_arg=ssh_port_arg,
                    tcp_intf_arg=tcp_intf_arg,
                    nccl_socket_intf_arg=nccl_socket_intf_arg,
                    extra_flags=extra_flags,
                    env=' '.join('-x %s' % key for key in env.keys()
                                    if env_util.is_exportable(key)),
                    command=ipengine_command)
        )
        subprocess.run(ipengine_command, shell=True,
                        env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    else:
        subprocess.run(ipcluster_stop_command, shell=True,
                       env=env, capture_output=True)
        _maybe_kill_ipcontroller_process(args.profile)


def main():
    args = parse_args()

    if args.version:
        print(bluefog.__version__)
        exit(0)

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

    env = os.environ.copy()
    env['BLUEFOG_CYCLE_TIME'] = str(20)  # Increase the cycle time
    if len(args.command) != 1 or args.command[0] not in ("start", "stop"):
        raise ValueError("The last command has to be either 'start' or 'stop', but it is "
                         "{} now.".format(args.command))
    command = args.command[0]

    # TODO(ybc) How to stop it properly?
    # In multiple machine env, we alose need to remove engine.json and client.json file.
    ipcluster_stop_command = "ipcluster stop --profile {profile}".format(
        profile=args.profile)

    try:
        if not remote_host_names:
            local_machine_launch(args, env, command, ipcluster_stop_command)
        else:
            multiple_machines_launch(args, env, all_host_names=all_host_names,
                                     hosts_arg=hosts_arg,
                                     remote_host_names=remote_host_names,
                                     command=command,
                                     ipcluster_stop_command=ipcluster_stop_command)
    except Exception as e:
        print("Fail to launch ibfrun. Error: ", e)
        _maybe_kill_ipcontroller_process(args.profile)


if __name__ == "__main__":
    main()
