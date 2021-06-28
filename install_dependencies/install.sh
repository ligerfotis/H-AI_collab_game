#!/usr/bin/bash
python3 -m venv env
source env/bin/activate
pip install -r install_dependencies/requirements.txt
#export PYTHONPATH=$PWD:$PYTHONPATH