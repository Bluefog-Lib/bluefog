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

from contextlib import contextmanager
from multiprocessing import Process, Queue
import os
import sysconfig

EXTENSIONS = ['tensorflow', 'torch']

def is_running_from_ipython():
    from IPython import get_ipython
    return get_ipython() is not None

def get_ext_suffix():
    """Determine library extension for various versions of Python."""
    ext_suffix = sysconfig.get_config_var('EXT_SUFFIX')
    if ext_suffix:
        return ext_suffix

    ext_suffix = sysconfig.get_config_var('SO')
    if ext_suffix:
        return ext_suffix

    return '.so'


def get_extension_full_path(pkg_path, *args):
    assert len(args) >= 1
    dir_path = os.path.join(os.path.dirname(pkg_path), *args[:-1])
    full_path = os.path.join(dir_path, args[-1] + get_ext_suffix())
    return full_path


def check_extension(ext_name, pkg_path, *args):
    full_path = get_extension_full_path(pkg_path, *args)
    if not os.path.exists(full_path):
        raise ImportError(
            'Extension {} has not been built. '.format(ext_name))


def _check_extension_lambda(ext_base_name, fn, fn_desc, verbose):
    """
    Tries to load the extension in a new process.  If successful, puts fn(ext)
    to the queue or False otherwise.  Mutes all stdout/stderr.
    """
    def _target_fn(ext_base_name, fn, fn_desc, queue, verbose):
        import importlib
        import sys
        import traceback

        if verbose:
            print('Checking whether extension {ext_base_name} was {fn_desc}.'.format(
                ext_base_name=ext_base_name, fn_desc=fn_desc))
        else:
            # Suppress output
            sys.stdout = open(os.devnull, 'w')
            sys.stderr = open(os.devnull, 'w')

        try:
            ext = importlib.import_module('.' + ext_base_name, 'bluefog')
            result = fn(ext)
        except:  # pylint: disable=bare-except
            traceback.print_exc()
            result = None

        if verbose:
            print('Extension {ext_base_name} {flag} {fn_desc}.'.format(
                ext_base_name=ext_base_name, flag=('was' if result else 'was NOT'),
                fn_desc=fn_desc))

        queue.put(result)

    queue = Queue()
    p = Process(target=_target_fn,
                args=(ext_base_name, fn, fn_desc, queue, verbose))
    p.daemon = True
    p.start()
    p.join()
    return queue.get_nowait()


def extension_available(ext_base_name, verbose=False):
    available_fn = lambda ext: ext is not None
    return _check_extension_lambda(
        ext_base_name, available_fn, 'built', verbose) or False


def mpi_built(verbose=False):
    for ext_base_name in EXTENSIONS:
        built_fn = lambda ext: ext.mpi_built()
        result = _check_extension_lambda(
            ext_base_name, built_fn, 'built with MPI', verbose)
        if result is not None:
            return result
    return False


@contextmanager
def env(**kwargs):
    # ignore args with None values
    for k in list(kwargs.keys()):
        if kwargs[k] is None:
            del kwargs[k]

    # backup environment
    backup = {}
    for k in kwargs:
        backup[k] = os.environ.get(k)

    # set new values & yield
    for k, v in kwargs.items():
        os.environ[k] = v

    try:
        yield
    finally:
        # restore environment
        for k in kwargs:
            if backup[k] is not None:
                os.environ[k] = backup[k]
            else:
                del os.environ[k]
