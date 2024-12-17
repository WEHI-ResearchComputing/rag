#!/bin/bash
#SBATCH -c 12 --mem 30G 
#SBATCH --gres gpu:A30:1 -p gpuq
#SBATCH --output ollama-server.log

set -eu

module purge
module load apptainer/1.3.5

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

export OLLAMA_HOST=$SLURM_NODELIST OLLAMA_MODELS=$ollama_models OLLAMA_MAX_LOADED_MODELS=2

apptainer exec --nv \
     -B $TMPDIR:/tmp \
     -B /vast,/stornext \
     --env OLLAMA_HOST=$SLURM_NODELIST \
     --env OLLAMA_MODELS=$ollama_models \
     --env OLLAMA_MAX_LOADED_MODELS=2 \
     oras://ghcr.io/wehi-researchcomputing/rag:0.1.1 \
     	ollama serve
