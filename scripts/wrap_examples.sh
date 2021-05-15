#!/bin/bash
# Assume running under the root directory of Bluefog
set -e
set -x

RAND_ID=$(git log -1 --format='%H' | fold -w 16 | head -n 1)
TMP_DIR=/tmp/bluefog_${RAND_ID}
TMP_EXAMPLE_DIR=${TMP_DIR}/examples
TMP_TEST_DIR=${TMP_DIR}/test

mkdir ${TMP_DIR}
mkdir ${TMP_EXAMPLE_DIR}
mkdir ${TMP_TEST_DIR}

cp scripts/run_unittest.sh ${TMP_DIR}

cp examples/*.py ${TMP_EXAMPLE_DIR}
cp examples/*.ipynb ${TMP_EXAMPLE_DIR}

cp test/pytest.ini ${TMP_TEST_DIR}
cp test/common.py ${TMP_TEST_DIR}
cp test/torch_basics_test.py ${TMP_TEST_DIR}
cp test/torch_ops_test.py ${TMP_TEST_DIR}
cp test/torch_win_ops_test.py ${TMP_TEST_DIR}

cd ${TMP_DIR}
tar -czf examples.tar.gz *
cd -
cp ${TMP_DIR}/examples.tar.gz .

rm -r ${TMP_DIR}
