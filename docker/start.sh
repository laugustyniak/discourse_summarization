#!/bin/bash
# old version of inline docker run, after building dockerfile as name sa
# docker run -d -p 27:22 -p 8778:8888 -e "PASSWORD=SAa2015" -v /datasets:/datasets -v /home/lukasz:/notebooks -v /nfs:/nfs  sa
docker run -d -p 27:22 -p 8778:8888 -e "PASSWORD=SAa2015" -v /home/lukasz:/notebooks sa

