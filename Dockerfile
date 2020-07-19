FROM tiagopeixoto/graph-tool

RUN pacman --noconfirm -S python-pip
COPY requirements.txt .
RUN pip install -r requirements.txt
RUN python -m spacy download en_core_web_sm