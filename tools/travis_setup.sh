#!/usr/bin/env bash
set -ex

WHEELHOUSE="--no-index --find-links=http://wheels.scikit-image.org/"

pip install wheel nose
pip install -r requirements.txt $WHEELHOUSE
apt-get install libzmq-dev
pip install ipython runipy

