FROM tiagopeixoto/graph-tool

RUN pacman --noconfirm -S python-pip
RUN pip install tqdm==4.45.0 networkx==2.4 more_itertools==8.2.0 joblib==0.14.1 mlflow seaborn toolz
RUN pip install spacy==2.3.5 scikit-learn scipy matplotlib nltk python-rake==1.4.5
RUN python -m spacy download en_vectors_web_lg
RUN python -m spacy download en_core_web_sm
RUN pip install textblob

EXPOSE 5005
EXPOSE 8888
