#!/bin/bash
# wrapper script to pull models because Ollama pulls models into HOME

set -eu

host=$1
model=$2

module purge
module load apptainer/1.3.3

ollama_models=/vast/scratch/users/$USER/ollama-models
ollama_tmp=/vast/scratch/users/$USER/tmp

apptainer exec \
     -B $ollama_tmp:/tmp \
     -B /vast,/stornext \
     --env OLLAMA_HOST=$host \
     oras://ghcr.io/wehi-researchcomputing/rag:0.1.0 ollama pull $model
