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

import argparse
import os
import re
import shlex
import socket
import subprocess
import sys
import traceback

import psutil
import bluefog
from bluefog.run import env_util, network_util, horovod_driver


def make_override_action(override_args):
    class StoreOverrideAction(argparse.Action):
        def __init__(self,
                     option_strings,
                     dest,
                     default=None,
                     type=None,
                     choices=None,
                     required=False,
                     help=None):
            super(StoreOverrideAction, self).__init__(
                option_strings=option_strings,
                dest=dest,
                nargs=1,
                default=default,
                type=type,
                choices=choices,
                required=required,
                help=help)

        def __call__(self, parser, args, values, option_string=None):
            override_args.add(self.dest)
            setattr(args, self.dest, values[0])

    return StoreOverrideAction


def parse_args():

    override_args = set()

    parser = argparse.ArgumentParser(description='Bluefog Runner')

    parser.add_argument('-v', '--version', action="store_true", dest="version",
                        help="Shows bluefog version.")

    parser.add_argument('-np', '--num-proc', action="store", dest="np",
                        type=int, help="Total number of training processes.")

    parser.add_argument('-p', '--ssh-port', action="store", dest="ssh_port",
                        type=int, help="SSH port on all the hosts.")

    parser.add_argument('--network-interface', action='store', dest='nic',
                        help='Specify the network interface used for communication.')

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
    parser.add_argument('--use-infiniband', action="store_true", dest="use_infiniband",
                        help='If set, use inifiniband to communication instead of TCP.')

    parser.add_argument('--extra-mpi-flags', action="store", dest="extra_flags",
                        help='Extra mpi flages you want to pass for mpirun.')

    parser.add_argument('--prefix', action='store', dest="prefix", default='',
                        type=str, help='A prefix path to add before mpirun. '
                        'Set if you need to specify the path for mpirun')

    parser.add_argument('--verbose', action="store_true", dest="verbose",
                        help="If this flag is set, extra messages will "
                             "printed.")

    parser.add_argument('command', nargs=argparse.REMAINDER,
                        help="Command to be executed.")

    group_timeline = parser.add_argument_group('timeline arguments')
    group_timeline.add_argument('--timeline-filename', action=make_override_action(override_args),
                                help='JSON file containing timeline of '
                                     'Bluefog events used for debugging '
                                     'performance. If this is provided, '
                                     'timeline events will be recorded, '
                                     'which can have a negative impact on training performance.')

    parsed_args = parser.parse_args()

    if not parsed_args.version and not parsed_args.np:
        parser.error('argument -np/--num-proc is required')

    return parsed_args


def main():
    args = parse_args()

    if args.version:
        print(bluefog.__version__)
        exit(0)

    hosts_arg, all_host_names = network_util.get_hosts_arg_and_hostnames(args)
    remote_host_names = network_util.filter_local_addresses(all_host_names)

    common_intfs = set()
    if remote_host_names:
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

    if args.ssh_port:
        ssh_port_arg = "-mca plm_rsh_args \"-p {ssh_port}\"".format(
            ssh_port=args.ssh_port)
    else:
        ssh_port_arg = ""

    if args.use_infiniband:
        ib_arg = "-mca btl openib,self"
    else:
        ib_arg = "-mca btl ^openib"

    if args.prefix:
        mpi_prefix = args.prefix
    else:
        mpi_prefix = ""

    if not env_util.is_open_mpi_installed():
        raise Exception(
            'bfrun convenience script currently only supports Open MPI.\n\n'
            'Choose one of:\n'
            '1. Install Open MPI 4.0.0+ and re-install Bluefog.\n'
            '2. Run distributed '
            'training script using the standard way provided by your'
            ' MPI distribution (usually mpirun, srun, or jsrun).')

    extra_flags = args.extra_flags if args.extra_flags else ''
    # Pass all the env variables to the mpirun command.
    env = os.environ.copy()
    env = env_util.set_env_from_args(env, args)
    mpirun_command = (
        '{prefix}mpirun --allow-run-as-root '
        '-np {num_proc} {hosts_arg} '
        '-bind-to none -map-by slot '
        '-mca pml ob1 {ib_arg} '
        '{ssh_port_arg} {tcp_intf_arg} '
        '{nccl_socket_intf_arg} '
        '{extra_flags} {env} {command}'
        .format(prefix=mpi_prefix,
                num_proc=args.np,
                hosts_arg=hosts_arg,
                ib_arg=ib_arg,
                ssh_port_arg=ssh_port_arg,
                tcp_intf_arg=tcp_intf_arg,
                nccl_socket_intf_arg=nccl_socket_intf_arg,
                extra_flags=extra_flags,
                env=' '.join('-x %s' % key for key in env.keys() if env_util.is_exportable(key)),
                command=' '.join(shlex.quote(par) for par in args.command))
    )

    if args.verbose:
        print(mpirun_command)
    # Execute the mpirun command.
    os.execve('/bin/sh', ['/bin/sh', '-c', mpirun_command], env)


if __name__ == "__main__":
    main()
