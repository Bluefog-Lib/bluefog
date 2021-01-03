# Copyright 2020 BlueFog Team. All Rights Reserved.
#
# This code is modified from Uber Horovod
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

import re
import shlex
import subprocess
import sys
import traceback

# List of regular expressions to ignore environment variables by.
IGNORE_REGEXES = {'BASH_FUNC_.*\(\)', 'OLDPWD'}

BLUEFOG_TIMELINE = 'BLUEFOG_TIMELINE'
BLUEFOG_LOG_LEVEL = 'BLUEFOG_LOG_LEVEL'


def is_exportable(v):
    return not any(re.match(r, v) for r in IGNORE_REGEXES)


def is_open_mpi_installed():
    command = 'mpirun --version'
    try:
        output_msg = str(subprocess.check_output(
            shlex.split(command), universal_newlines=True))
    except Exception:  # pylint disable=broad-except
        print("Was not able to run %s:\n%s" % (command, output_msg),
              file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        return False

    if 'Open MPI' not in output_msg:
        print('Open MPI not found in output of mpirun --version.',
              file=sys.stderr)
        return False
    return True


def is_ipyparallel_installed():
    try:
        import ipyparallel  # pylint: disable=unused-import
        return True
    except ImportError:
        return False


def _add_arg_to_env(env, env_key, arg_value, transform_fn=None):
    if arg_value is not None:
        value = arg_value
        if transform_fn:
            value = transform_fn(value)
        env[env_key] = str(value)

def set_env_from_args(env, args):
    # Timeline
    if args.timeline_filename:
        _add_arg_to_env(env, BLUEFOG_TIMELINE, args.timeline_filename)

    if args.verbose:
        _add_arg_to_env(env, BLUEFOG_LOG_LEVEL, "debug")

    return env
