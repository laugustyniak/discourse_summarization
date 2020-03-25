#!/bin/bash
docker run -d -p 8999:8888 -p 8998:8080 \
  -e GEN_CERT=yes -e JUPYTER_ENABLE_LAB=yes \
  -v /datasets/misinformation:/home/jovyan/work \
  prodigy_jupyter_lab start-notebook.sh --NotebookApp.token='MisInfPAss2017'
