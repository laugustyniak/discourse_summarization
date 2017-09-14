#!/bin/bash

docker build -t feng-2014-rstparser .
docker run -v /Path/to/text/files:/samples -i -t feng-2014-rstparser
cd gCRF_dist/src
python parse.py /samples/F0001.txt