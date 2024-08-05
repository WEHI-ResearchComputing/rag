bootstrap: docker
from: python:3.12.4-slim-bookworm

%files
  requirements.txt /opt/requirements.txt
  chatbot.py /usr/local/bin/chatbot.py

%post
  pip install -r /opt/requirements.txt --no-cache

%labels
  AUTHOR Edward Yang
  VERSION 0.0.1

%runscript
  exec chatbot.py "$@"
