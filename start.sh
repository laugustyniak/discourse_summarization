#!/bin/bash
docker run -p 8877:8888 -it -u user -w /home/user -v $(pwd):/home/user discourse_summarization bash
