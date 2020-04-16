# docker stacks tensorflow image
ARG BASE_CONTAINER=jupyter/scipy-notebook
FROM $BASE_CONTAINER

LABEL maintainer="Jupyter Project <jupyter@googlegroups.com>"

# Install Tensorflow
#RUN pip install --quiet \
#    'tensorflow==2.1.0' && \
#    fix-permissions $CONDA_DIR && \
#    fix-permissions /home/$NB_USER

# Install Graphviz.
#RUN conda install -y graphviz=2.40.1 python-graphviz=0.8.4 \
# && conda clean -ya

# isntall pytorch geometrics
#ENV CUDA "cpu"
#RUN pip install torch==1.4.0+${CUDA} torchvision==0.5.0+${CUDA} -f https://download.pytorch.org/whl/torch_stable.html
#RUN pip install torch-scatter==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.4.0.html \
#    && pip install torch-sparse==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.4.0.html \
#    && pip install torch-cluster==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.4.0.html \
#    && pip install torch-spline-conv==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.4.0.html \
#    && pip install torch-geometric


RUN pip install tqdm more-itertools==8.0.2 nltk==3.4.5 hdbscan==0.8.26

#RUN pip install https://storage.googleapis.com/spacy-pl-public-models/pl_model-1.0.0.tar.gz
#RUN python -m spacy download en_core_web_sm
#RUN python -m spacy download en

# add prodigy
ENV WORK_DIR ${HOME}/work
USER jovyan
COPY prodigy-1.9.9-cp36.cp37.cp38-cp36m.cp37m.cp38-linux_x86_64.whl ${WORK_DIR}
RUN pip install ${WORK_DIR}/prodigy-1.9.9-cp36.cp37.cp38-cp36m.cp37m.cp38-linux_x86_64.whl

# setup volume to this location in docker run command
ENV PRODIGY_HOME ${WORK_DIR}
ENV PRODIGY_LOGGING basic
#COPY prodigy.json ${WORK_DIR} copy this file to mounted volume
