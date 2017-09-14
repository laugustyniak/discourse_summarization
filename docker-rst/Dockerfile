FROM java:7

MAINTAINER Łukasz Augustyniak "luk.augustyniak@gmail.com"

# Install Python.
RUN \
  apt-get update && \
  apt-get install -y python python-dev python-pip python-virtualenv && \
  rm -rf /var/lib/apt/lists/*

RUN wget http://pyyaml.org/download/pyyaml/PyYAML-3.09.tar.gz \
	&& tar -zxf PyYAML-3.09.tar.gz \
	&& cd PyYAML-3.09 \
	&& python setup.py install

RUN wget https://pypi.python.org/packages/source/n/nltk/nltk-2.0b9.tar.gz \
	&& tar -zxf nltk-2.0b9.tar.gz \
	&& cd nltk-2.0b9 \
	&& python setup.py install

RUN wget http://www.cs.toronto.edu/~weifeng/software/discourse_parse-2.01.tar.gz \
	&& tar -zxf discourse_parse-2.01.tar.gz \
        && cd gCRF_dist/tools/crfsuite

RUN wget https://github.com/downloads/chokkan/liblbfgs/liblbfgs-1.10.tar.gz \
	&& tar -zxf liblbfgs-1.10.tar.gz \
        && cd liblbfgs-1.10 \
        && ./configure --prefix=$HOME/local \
        && make \
        && make install

RUN cd gCRF_dist/tools/crfsuite/crfsuite-0.12 \
	&& chmod +x configure \
        && ./configure --prefix=$HOME/local --with-liblbfgs=$HOME/local \
	&& make \
	&& make install

RUN cd gCRF_dist/tools/crfsuite && \
	cp $HOME/local/bin/crfsuite crfsuite-stdin \
	&& chmod +x crfsuite-stdin

# test CRF
RUN cd gCRF_dist/tools/crfsuite \
	&& ./crfsuite-stdin tag -pi -m ../../model/tree_build_set_CRF/label/intra.crfsuite test.txt

# Patch crfsuite wrapper
ADD subprocess.patch .
RUN patch /gCRF_dist/src/classifiers/crf_classifier.py < /subprocess.patch

# Test discourse parser
RUN cd gCRF_dist/src \
	&& python sanity_check.py

USER root

# SSH
RUN apt-get update && apt-get install -y openssh-server
RUN mkdir /var/run/sshd
RUN echo 'root:pythonSA2016' | chpasswd
RUN sed -i 's/PermitRootLogin without-password/PermitRootLogin yes/' /etc/ssh/sshd_config

# SSH login fix. Otherwise user is kicked off after login
RUN sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd
ENV NOTVISIBLE "in users profile"
RUN echo "export VISIBLE=now" >> /etc/profile
EXPOSE 22
CMD ["/usr/sbin/sshd", "-D"]

#numpy needs python build tools
RUN apt-get install -y build-essential \
 python-dev \
 gfortran \
 swig \
 libatlas-dev \
 liblapack-dev \
 libpng-dev \
 libfreetype6 \
 libfreetype6-dev \
 libxft-dev \
 libxml2-dev \
 libxslt-dev \
 zlib1g-dev
RUN pip install numpy
RUN pip install scipy

# Python libs
RUN pip install networkx \
 setuptools \
 IPython==4.0 \
 pandas \
 statsmodels \
 BeautifulSoup4 \
 unidecode \
 justext \
 plotly \
 line_profiler \
 memory_profiler \
 tqdm \
 joblib \
 seaborn \
 matplotlib \
 scikit-learn
RUN pip install spacy

RUN pip install -U gunicorn falcon
EXPOSE 5000 8000 80
COPY ./app /app
WORKDIR /app

# run the API
# IMPOIRTANT look out for complex models timeout of 120 seconds could not be enough for load model
CMD ["gunicorn", "-b", "0.0.0.0:8000", "--workers", "1", "--timeout", "120", "sentimentAPI:app"]


CMD ["/bin/sh"]