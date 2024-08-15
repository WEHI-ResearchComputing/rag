bootstrap: docker
from: python:3.12.4-slim-bookworm
stage: devel

%post
  apt-get update
  apt-get install -y curl git
  pip install git+https://github.com/wehi-researchcomputing/rag.git@gradio --no-cache
  curl -L https://ollama.com/download/ollama-linux-amd64 -o /usr/local/bin/ollama
  chmod +x /usr/local/bin/ollama


# Production stage
bootstrap: docker
from: python:3.12.4-slim-bookworm
stage: prod

%files from devel
  /usr/local /usr

%labels
  AUTHOR Edward Yang
  VERSION 0.1.0

%runscript
  exec wehiragchat "$@"
