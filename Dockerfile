FROM tiagopeixoto/graph-tool

RUN pacman --noconfirm -S python-pip
COPY requirements.txt .
RUN pip install -r requirements.txt