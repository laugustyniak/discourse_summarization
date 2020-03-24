# docker stacks tensorflow image
ARG BASE_CONTAINER=jupyter/scipy-notebook
FROM $BASE_CONTAINER

LABEL maintainer="Jupyter Project <jupyter@googlegroups.com>"

# Install Tensorflow
RUN pip install --quiet \
    'tensorflow==2.1.0' && \
    fix-permissions $CONDA_DIR && \
    fix-permissions /home/$NB_USER

# Install Graphviz.
RUN conda install -y graphviz=2.40.1 python-graphviz=0.8.4 \
 && conda clean -ya

# isntall pytorch geometrics
ENV CUDA "cpu"
RUN pip install torch==1.4.0+${CUDA} torchvision==0.5.0+${CUDA} -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install torch-scatter==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.4.0.html \
    && pip install torch-sparse==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.4.0.html \
    && pip install torch-cluster==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.4.0.html \
    && pip install torch-spline-conv==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.4.0.html \
    && pip install torch-geometric


RUN pip install tqdm toolz==0.10.0 spacy==2.2.0 textblob==0.15.3 nltk==3.4.5 more-itertools==8.0.2 flair==0.4.5

RUN python -m spacy download en_core_web_sm
#RUN python -m spacy download en_core_web_lg
#RUN python -m spacy download en_trf_bertbaseuncased_lg
#RUN python -m spacy download de_trf_bertbasecased_lg
#RUN python -m spacy download en_trf_distilbertbaseuncased_lg
#RUN python -m spacy download en_trf_robertabase_lg
#RUN python -m spacy download en_trf_xlnetbasecased_lg

# add prodigy
USER root
RUN mkdir /prodigy
COPY prodigy-1.9.9-cp36.cp37.cp38-cp36m.cp37m.cp38-linux_x86_64.whl /prodigy
RUN pip install /prodigy/prodigy-1.9.9-cp36.cp37.cp38-cp36m.cp37m.cp38-linux_x86_64.whl

# prodigy work with jupyter lab <2.0
#RUN pip install jupyterlab==1.0.0rc0
#RUN jupyter labextension install jupyterlab-prodigy

# setup volume to this location in docker run command
ENV PRODIGY_HOME "/prodigy"

