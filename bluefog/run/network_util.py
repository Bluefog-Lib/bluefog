# Modifications copyright (C) 2020 Bluefog Team. All Rights Reserved.
# Copyright 2019 Uber Technologies, Inc. All Rights Reserved.
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

import concurrent.futures
import io
import os
import re
import socket

import psutil
from bluefog.run.horovodrun.common.util import safe_shell_exec

# Number of retries for sshing into the hosts
SSH_RETRIES = 3


def execute_function_multithreaded(exec_command, args_list, max_workers=100):
    """ Execute functions in multithread manner.

    Args:
        exec_command: Callable function to run.
        args_list: The arguments list for exec_command, each element will be passed
            into exec_command and run in multithread paralleled.
        max_workers (int, optional): Defaults to 100.

    Returns:
        A dictionary that maps index to result of exec_command, i.e. index to the
        result of exec_command taking "index"-th args in the args_list.
    """
    results = {}
    _max_workers = min(max_workers, len(args_list))
    with concurrent.futures.ThreadPoolExecutor(max_workers=_max_workers) as executor:
        futures_to_index = {executor.submit(exec_command, arg): index
                            for index, arg in enumerate(args_list)}
        for future in concurrent.futures.as_completed(futures_to_index):
            index = futures_to_index[future]
            results[index] = future.result()
    return results


def _parse_host_files(filename):
    """Transform the hostfile into a format of <IP address> or <host name>:<Number of GPUs>

    Args:
        filename: Should contains only <IP address> or <host name> slots=<number of GPUs>
    Returns:
        Comma separated string of <IP address> or <host name>:<Number of GPUs>
    """
    hosts = []
    for line in open(filename):
        line = line.rstrip()
        hostname = line.split()[0]
        slots = line.split('=')[1]
        hosts.append('{name}:{slots}'.format(name=hostname, slots=slots))

    return ','.join(hosts)


def get_hosts_arg_and_hostnames(args):
    # if hosts are not specified, either parse from hostfile, or default as
    # localhost
    if not args.hosts:
        if args.hostfile:
            args.hosts = _parse_host_files(args.hostfile)
        else:
            # Set hosts to localhost if not specified
            args.hosts = 'localhost:{np}'.format(np=args.np)

    all_host_names = []
    host_list = args.hosts.split(',')
    all_host_names = []
    pattern = re.compile(r'^[\w.-]+:\d+$')
    for host in host_list:
        if not pattern.match(host.strip()):
            raise ValueError('Invalid host input, please make sure it has '
                             'format as : worker-0:2,worker-1:2.')
        all_host_names.append(host.strip().split(':')[0])
    hosts_arg = '-H {hosts}'.format(hosts=args.hosts)
    return hosts_arg, all_host_names


def _get_local_host_addresses():
    local_addresses = []
    for intf_info_list in psutil.net_if_addrs().values():
        for intf_info in intf_info_list:
            if intf_info.family == socket.AF_INET:
                local_addresses.append(intf_info.address)
    return local_addresses


def get_local_host_intfs():
    return set(psutil.net_if_addrs().keys())


def filter_local_addresses(all_host_names):
    local_addresses = _get_local_host_addresses()

    def resolve_host_name(host_name):
        try:
            return socket.gethostbyname(host_name)
        except socket.gaierror:
            return None

    args_list = [host for host in all_host_names]
    host_addresses = execute_function_multithreaded(
        resolve_host_name, args_list).values()

    remote_host_names = []
    for host_address, host_name in zip(host_addresses, all_host_names):
        if not host_address or host_address not in local_addresses:
            remote_host_names.append(host_name)

    return remote_host_names

def exec_command(command):
    exit_code = 1
    output_msg = ""

    # Try ssh 3 times
    for _ in range(SSH_RETRIES):
        output = io.StringIO()
        try:
            exit_code = safe_shell_exec.execute(command,
                                                stdout=output,
                                                stderr=output)
            if exit_code == 0:
                break
            else:
                output_msg = output.getvalue()
        finally:
            output.close()
    return exit_code, output_msg


def check_all_hosts_ssh_successful(host_addresses, ssh_port=None):
    """Checks if ssh can successfully be performed to all the hosts.

    Args:
      host_addresses: list of addresses to ssh into. for example,
        ['worker-0','worker-1']
        ['10.11.11.11', '10.11.11.12']
      ssh_port: A int for ssh port number.

    Returns:
      True if all ssh was successful into all the addresses.
    """
    if ssh_port:
        ssh_port_arg = "-p {ssh_port}".format(ssh_port=ssh_port)
    else:
        ssh_port_arg = ""

    ssh_command_format = 'ssh -o StrictHostKeyChecking=no {host} {ssh_port_arg} date'

    args_list = [ssh_command_format.format(host=host_address,
                                           ssh_port_arg=ssh_port_arg)
                 for host_address in host_addresses]
    ssh_exit_codes = execute_function_multithreaded(exec_command,
                                                    args_list)

    ssh_successful_to_all_hosts = True
    for index, ssh_status in ssh_exit_codes.items():
        exit_code, output_msg = ssh_status[0], ssh_status[1]
        if exit_code != 0:
            print("ssh not successful for host {host}:\n{msg_output}".format(
                host=host_addresses[index],
                msg_output=output_msg
            ))

            ssh_successful_to_all_hosts = False
    if not ssh_successful_to_all_hosts:
        exit(1)
    return True


def scp_transmit_file(file, remote_host_names, ssh_port=None):
    if ssh_port:
        mkdir_command_format = "ssh -p {port} {host} 'mkdir -p {file_dir}'"
        scp_command_format = "scp -P {port} {file} {host}:{file}"
    else:
        mkdir_command_format = "ssh {host} 'mkdir -p {file_dir}'"
        scp_command_format = "scp {file} {host}:{file}"

    commands = []
    for host in remote_host_names:
        mkdir_command = mkdir_command_format.format(host=host, port=ssh_port,
                                                    file_dir=os.path.dirname(file))
        scp_command = scp_command_format.format(host=host, file=file,
                                                port=ssh_port)
        commands.append(mkdir_command + " && " + scp_command)

    scp_exit_codes = execute_function_multithreaded(exec_command,
                                                    commands)

    scp_successful_to_all_hosts = True
    for index, ssh_status in scp_exit_codes.items():
        exit_code, output_msg = ssh_status[0], ssh_status[1]
        if exit_code != 0:
            print("scp not successful for host {host}:\n{msg_output}".format(
                host=remote_host_names[index],
                msg_output=output_msg
            ))

            scp_successful_to_all_hosts = False
    if not scp_successful_to_all_hosts:
        exit(1)
    return True
