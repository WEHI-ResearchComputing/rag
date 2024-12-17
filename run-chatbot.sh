#!/bin/bash
# wrapper script to pull models because Ollama pulls models into HOME

set -eu

ollama_host=$1

module purge
module load apptainer/1.3.3

ollama_models=/vast/scratch/users/$USER/ollama-models
ollama_tmp=/vast/scratch/users/$USER/tmp

apptainer run \
     -B $ollama_tmp:/tmp \
     -B /vast,/stornext \
     oras://ghcr.io/wehi-researchcomputing/rag:0.1.1 --ollama-host $ollama_host
