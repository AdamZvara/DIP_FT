source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

set -a # exports all variables to parent scripts

HF_HOME="${HF_HOME:?Set HF_HOME in .env}"
TRANSFORMERS_CACHE=$HF_HOME
DATADIR="${DATADIR:?Set DATADIR in .env}"
CONDA_ENV="${CONDA_ENV:-easyedit}"

# append a line to a file "jobs_info.txt" containing the ID of the job, the hostname of the node it is run on, and the path to a scratch directory
# this information helps to find a scratch directory in case the job fails, and you need to remove the scratch directory manually
echo "$(date) $PBS_JOBID is running on node `hostname -f` in a scratch directory $SCRATCHDIR" >> "$DATADIR/jobs/jobs_info.txt"

# test if the scratch directory is set
# if scratch directory is not set, issue error message and exit
test -n "$SCRATCHDIR" || { echo >&2 "Variable SCRATCHDIR is not set!"; exit 1; }

# setup conda environment
if [ ! -d $HOME/.conda/ ]; then
    ln -s "$DATADIR/.conda" $HOME/.conda
fi
conda activate "$CONDA_ENV"
