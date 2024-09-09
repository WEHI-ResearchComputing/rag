#!/bin/bash
# wrapper script to pull models because Ollama pulls models into HOME

set -eu

ollama_host=$1

module purge
module load apptainer/1.3.3

ollama_models=/vast/scratch/users/$USER/ollama-models
ollama_tmp=/vast/scratch/users/$USER/tmp

apptainer run \
     -B $TMPDIR:/tmp \
     -B /vast,/stornext \
     oras://ghcr.io/wehi-researchcomputing/rag:0.1.0 --ollama-host $ollama_host
