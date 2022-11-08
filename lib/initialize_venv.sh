#!/bin/bash
# install all dependencies into venv and activate it
python3 -m venv TOX_GPU_VENV
TOX_GPU_VENV/bin/pip install --no-cache-dir --upgrade pip setuptools
TOX_GPU_VENV/bin/pip install --no-cache-dir tensorflow==2.8.0 tensorflow-addons==0.16.1 tensorflow-probability==0.16.0 tensorflow-hub==0.12.0 scipy numpy sklearn pandas tabulate matplotlib rdkit talos mordred matplotlib molvs
source ./TOX_GPU_VENV/bin/activate
