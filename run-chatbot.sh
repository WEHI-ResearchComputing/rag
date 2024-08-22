#!/bin/bash
# wrapper script to pull models because Ollama pulls models into HOME

set -eu

ollama_host=$1

# set port that chatbot is listening to
if [ -z "${2}" ]
then
	chatbot_port_flag=""
else
	chatbot_port_flag="--port $2"
fi

module purge
module load apptainer/1.2.5

ollama_models=/vast/scratch/users/$USER/ollama-models
ollama_tmp=/vast/scratch/users/$USER/tmp

export TMPDIR=/vast/scratch/users/$USER/tmp
mkdir -p $TMPDIR

apptainer run \
     -B $TMPDIR:/tmp \
     -B /vast,/stornext \
     oras://ghcr.io/wehi-researchcomputing/rag:0.1.0 --ollama-host $ollama_host $chatbot_port_flag
