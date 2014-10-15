#!/usr/bin/env bash
set -ex

WHEELHOUSE="--no-index --find-links=http://wheels.scikit-image.org/"

pip install wheel nose
pip install numpy networkx scipy matplotlib ipython runipy $WHEELHOUSE

