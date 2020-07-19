FROM tiagopeixoto/graph-tool

RUN pacman --noconfirm -S python-pip
COPY requirements.txt .
RUN pip install -r requirements.txt
RUN python -m spacy download en_core_web_sm

# then run in container
# remember to change volume that has been mounted to docker container in start.sh script
#
# pip install -e .
# it will install discourse summarization in python env
# jupyter notebook --ip 0.0.0.0
# it will run jupyter notebook server with all dependecies