#!/bin/bash
#PBS -N Keras_tuner_job
#PBS -l select=1:ncpus=1:mem=16gb:scratch_local=32gb
#PBS -l walltime=4:00:00

# define a DATADIR variable: directory where the input files are taken from and where output will be copied to
DATADIR=/auto/brno2/home/nierja/Tox/src/DL # substitute username and path to to your real username and path

# append a line to a file "jobs_info.txt" containing the ID of the job, the hostname of node it is run on and the path to a scratch directory
# this information helps to find a scratch directory in case the job fails and you need to remove the scratch directory manually 
echo "$PBS_JOBID is running on node `hostname -f` in a scratch directory $SCRATCHDIR" >> $DATADIR/jobs_info.txt

# '--use-pep517'

cd $DATADIR
export TMPDIR=$SCRATCHDIR
module add python/3.8.0-gcc-rab6t cuda/cuda-11.2.0-intel-19.0.4-tn4edsz cudnn/cudnn-8.1.0.77-11.2-linux-x64-intel-19.0.4-wx22b5t
python3 -m venv HP_SEARCH2
HP_SEARCH2/bin/pip install --no-cache-dir --upgrade pip setuptools
HP_SEARCH2/bin/pip install --no-cache-dir tensorflow==2.8.0 tensorflow-addons==0.16.1 tensorflow-probability==0.16.0 tensorflow-hub==0.12.0 scipy numpy pandas tabulate matplotlib rdkit talos mordred matplotlib molvs keras_tuner scikit-learn==1.1.2
source ./HP_SEARCH2/bin/activate
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
alias ls="ls --color"
export PS1="{\[\e[32m\]\u\[\e[m\]@\[\e[36m\]\h\[\e[m\]:\W_\$?}$ "


# test if scratch directory is set
# if scratch directory is not set, issue error message and exit
test -n "$SCRATCHDIR" || { echo >&2 "Variable SCRATCHDIR is not set!"; exit 1; }

python Tox21_tuner.py --target=$target --fp=$fp --pca=$pca --weighted=$weight --ensamble=$ensamble

# clean the SCRATCH directory
clean_scratch
