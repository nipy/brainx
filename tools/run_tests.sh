#!/usr/bin/env bash

set -e

nosetests brainx

for nb in brainx/notebooks/*.ipynb; do
    echo "Running: $nb"
    runipy -q $nb
done

