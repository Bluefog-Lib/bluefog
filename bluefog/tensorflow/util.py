from distutils.version import LooseVersion

import tensorflow

# Eager Mode has been introduced in TF 1.7.0
if LooseVersion(tensorflow.__version__) >= LooseVersion('1.7.0'):
    from tensorflow.python.eager import context
    _has_eager = True
else:
    _has_eager = False


def _executing_eagerly():
    """Returns true if eager execution is supported and enabled."""
    return _has_eager and context.executing_eagerly()


def _cache(f):
    cache = dict()

    def wrapper(*args):
        key = (args, _executing_eagerly())

        if key in cache:
            return cache[key]
        else:
            retval = f(*args)
            cache[key] = retval
            return retval

    return wrapper
