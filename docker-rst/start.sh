#!/bin/bash
docker run -d -p 27:22 -p 8778:8888 -e "PASSWORD=SAa2015" -v /home/lukasz:/notebooks rst

