#!/bin/bash
module purge

module load genoa
module load foss/2025b python/3.14.2 OpenMPI/5.0.8 cmake/4.0.3

source /home/tut10/hron/venv-firedrake/bin/activate

export FIREDRAKE_CACHE_DIR=${HOME}/.firedrake_jit
export FIREDRAKE_CACHE_DIR=${HOME}/.firedrake_jit
export FIREDRAKE_TSFC_KERNEL_CACHE_DIR=${HOME}/.firedrake_jit/tsfc
export PYOP2_CACHE_DIR=${HOME}/.firedrake_jit/pyop2

export SLURM_MPI_TYPE=pmix
export OMP_NUM_THREADS=1
