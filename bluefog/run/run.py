import argparse
import os
import shlex
import subprocess
import sys
import traceback


import bluefog


def _is_open_mpi_installed():
    command = 'mpirun --version'
    try:
        output_msg = str(subprocess.check_output(
            shlex.split(command), universal_newlines=True))
    except Exception:
        print("Was not able to run %s:\n%s" % (command, output_msg),
              file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        return False

    if 'Open MPI' not in output_msg:
        print('Open MPI not found in output of mpirun --version.',
              file=sys.stderr)
        return False
    return True


def parse_args():
    parser = argparse.ArgumentParser(description='Bluefog Runner')

    parser.add_argument('-v', '--version', action="store_true", dest="version",
                        help="Shows bluefog version.")

    parser.add_argument('-np', '--num-proc', action="store", dest="np",
                        type=int, help="Total number of training processes.")

    parser.add_argument('-p', '--ssh-port', action="store", dest="ssh_port",
                        type=int, help="SSH port on all the hosts.")

    parser.add_argument('-H', '--host', action="store", dest="host",
                        help="To specify the list of host names as well as the "
                             "number of available slots on each host for "
                             "training processes using the following format: "
                             "<hostname>:<number of slots>,... . "
                             "E.g., host1:2,host2:4,host3:1 "
                             "indicates that 2 processes can run on "
                             "host1, 4 processes on host2, and 1 process "
                             "on host3.")

    parser.add_argument('--verbose', action="store_true",
                        dest="verbose",
                        help="If this flag is set, extra messages will "
                             "printed.")

    parser.add_argument('command', nargs=argparse.REMAINDER,
                        help="Command to be executed.")

    parsed_args = parser.parse_args()

    parsed_args.verbose = 2 if parsed_args.verbose else 1
    if not parsed_args.version and not parsed_args.np:
        parser.error('argument -np/--num-proc is required')

    return parsed_args


def main():
    args = parse_args()
    if args.version:
        print(bluefog.__version__)
        exit(0)

    if args.host:
        hosts_arg = args.host
    else:
        hosts_arg = ""

    if args.ssh_port:
        ssh_port_arg = "-mca plm_rsh_args \"-p {ssh_port}\"".format(
            ssh_port=args.ssh_port)
    else:
        ssh_port_arg = ""

    if not _is_open_mpi_installed():
        raise Exception(
            'bfrun convenience script currently only supports Open MPI.\n\n'
            'Choose one of:\n'
            '1. Install Open MPI 4.0.0+ and re-install Bluefog.\n'
            '2. Run distributed '
            'training script using the standard way provided by your'
            ' MPI distribution (usually mpirun, srun, or jsrun).')

    # Pass all the env variables to the mpirun command.
    env = os.environ.copy()

    mpirun_command = (
        'mpirun --allow-run-as-root --tag-output '
        '-np {num_proc} {hosts_arg} '
        '-bind-to none -map-by slot '
        '-mca pml ob1 -mca btl,mtl ^openib '
        '{ssh_port_arg} '
        '{env} {command}'  # expect a lot of environment variables
        .format(num_proc=args.np,
                hosts_arg=hosts_arg,
                ssh_port_arg=ssh_port_arg,
                env='',
                command=' '.join(shlex.quote(par) for par in args.command))
    )

    if args.verbose >= 2:
        print(mpirun_command)
    # Execute the mpirun command.
    os.execve('/bin/sh', ['/bin/sh', '-c', mpirun_command], env)


if __name__ == "__main__":
    main()
