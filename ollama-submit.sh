#!/bin/bash
#SBATCH -c 24 --mem 100G 
#SBATCH --gres gpu:A30:1 -p gpuq
#SBATCH --output ollama-server.log

set -eu

module purge
module load apptainer/1.2.3

# Update the conf file
sed -i "/host/c\host = \"`hostname`\"" conf.toml

ollama_models=/vast/scratch/users/$USER/ollama-models
ollama_tmp=/vast/scratch/users/$USER/tmp

mkdir -p $ollama_models $ollama_tmp

# Relevant Ollama environemnt variables to set:

#      OLLAMA_DEBUG               Show additional debug information (e.g. OLLAMA_DEBUG=1)
#      OLLAMA_HOST                IP Address for the ollama server (default 127.0.0.1:11434)
#      OLLAMA_KEEP_ALIVE          The duration that models stay loaded in memory (default "5m")
#      OLLAMA_MAX_LOADED_MODELS   Maximum number of loaded models (default 1)
#      OLLAMA_MAX_QUEUE           Maximum number of queued requests
#      OLLAMA_MODELS              The path to the models directory
#      OLLAMA_NUM_PARALLEL        Maximum number of parallel requests (default 1)
#      OLLAMA_NOPRUNE             Do not prune model blobs on startup
#      OLLAMA_ORIGINS             A comma separated list of allowed origins
#      OLLAMA_TMPDIR              Location for temporary files
#      OLLAMA_FLASH_ATTENTION     Enabled flash attention
#      OLLAMA_LLM_LIBRARY         Set LLM library to bypass autodetection
#      OLLAMA_MAX_VRAM            Maximum VRAM

export OLLAMA_HOST=$SLURM_NODELIST OLLAMA_MODELS=$ollama_models

apptainer run --nv \
     -B $TMPDIR:/tmp \
     -B $ollama_models:$HOME/.ollama/models \
     -B /vast,/stornext \
     docker://ollama/ollama
