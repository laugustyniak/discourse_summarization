#!/bin/bash
docker run -p 8877:8888 -p 6006:6006 -it -u user -w /home/user -v /datasets/sentiment/aspects:/home/user graph_tool bash

# then run in container -> jupyter notebook --ip 0.0.0.0