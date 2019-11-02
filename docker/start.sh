#!/bin/bash
docker run -dt -p 27:22 -p 8778:8888 -e "PASSWORD=SAa2015" -v /home/laugustyniak:/notebooks -v /datasets:/datasets -v /nfs:/nfs docker_sa

