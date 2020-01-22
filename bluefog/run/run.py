import argparse
import bluefog


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

    if parsed_args.verbose is None:
        # This happens if the user does "bfrun --verbose" without
        # specifying any value to verbose. For the sake of consistency, we set
        # the verbosity here to the default value of 1.
        parsed_args.verbose = 1

    if not parsed_args.version and not parsed_args.np:
        parser.error('argument -np/--num-proc is required')

    return parsed_args


def main():
    args = parse_args()
    if args.version:
        print(bluefog.__version__)
        exit(0)


if __name__ == "__main__":
    main()
