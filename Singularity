bootstrap: docker
from: python:3.12.4-slim-bookworm

%files
  requirements.txt /opt/requirements.txt
  chatbot.py /usr/local/bin/chatbot.py

%post
  pip install -r /opt/requirements.txt --no-cache
  apt-get update
  apt-get install -y curl
  apt-get clean
  rm -rf /var/lib/apt/lists/*
  curl -L https://ollama.com/download/ollama-linux-amd64 -o /usr/local/bin/ollama
  chmod +x /usr/local/bin/ollama

%labels
  AUTHOR Edward Yang
  VERSION 0.0.3

%runscript
  exec chatbot.py "$@"
