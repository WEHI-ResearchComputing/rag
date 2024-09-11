bootstrap: docker
from: python:3.12.4-slim-bookworm
stage: devel

%post
  apt-get update
  apt-get install -y curl git
  pip install git+https://github.com/wehi-researchcomputing/rag.git --no-cache
  curl -L https://github.com/ollama/ollama/releases/download/v0.3.10/ollama-linux-amd64.tgz -o /tmp/ollama-linux-amd64.tgz
  tar -C /usr -xzf /tmp/ollama-linux-amd64.tgz


# Production stage
bootstrap: docker
from: python:3.12.4-slim-bookworm
stage: prod

%files from devel
  /usr/local /usr

%labels
  AUTHOR Edward Yang
  VERSION 0.1.1
  OLLAMA_VERSION 0.3.10
  ARCH amd64
  GPU_ARCH cuda

%runscript
  exec wehiragchat "$@"
