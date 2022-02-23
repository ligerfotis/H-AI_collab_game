#!/usr/bin/bash
python3 -m venv env
source env/bin/activate
sudo apt-get install libjpeg-dev zlib1g-dev
pip install -U setuptools
pip install -r install_dependencies/requirements.txt
#export PYTHONPATH=$PWD:$PYTHONPATH
