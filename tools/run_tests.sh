#!/usr/bin/env bash

set -ex

nosetests brainx

for nb in brainx/notebooks/*.ipynb; do
    echo "Running: $nb"
    runipy -q $nb
done

