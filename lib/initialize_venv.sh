#!/bin/bash
# install all dependencies into venv and activate it
cd lib
python3 -m venv TOX_GPU_VENV
TOX_GPU_VENV/bin/pip install --no-cache-dir --upgrade pip setuptools
TOX_GPU_VENV/bin/pip install --no-cache-dir tensorflow==2.8.0 tensorflow-addons==0.16.1 tensorflow-probability==0.16.0 tensorflow-hub==0.12.0 scipy numpy scikit-learn pandas tabulate matplotlib rdkit mordred matplotlib molvs
cd ..
