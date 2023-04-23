#!/usr/bin/env bash

# Usage:  (under the root directory)
#   tools/mypy.sh [--version]

mypy_files=$(find bluefog examples test -name "*.py")
mypy --config-file=./scripts/.mypy.ini "$@" ${mypy_files}
