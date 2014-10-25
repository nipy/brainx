#!/usr/bin/env bash

set -e

export PYTHONWARNINGS="all"

for nb in brainx/notebooks/*.ipynb; do
    echo "Running: $nb"
    runipy -q $nb
done

if [[ $TRAVIS_PYTHON_VERSION == 3.* ]]; then
    export TEST_ARGS="--with-cov --cover-package brainx"
else
    export TEST_ARGS="brainx"
fi

# Add `--with-doctest` below, once doctests have been fixed
nosetests --exe -v $TEST_ARGS

