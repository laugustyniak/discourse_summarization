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

RUN cd gCRF_dist/tools/crfsuite \
	&& ./crfsuite-stdin tag -pi -m ../../model/tree_build_set_CRF/label/intra.crfsuite test.txt

USER root

# Graph-tool install
#RUN echo "deb http://downloads.skewed.de/apt/trusty trusty universe" >> /etc/apt/sources.list
#RUN echo "deb-src http://downloads.skewed.de/apt/trusty trusty universe" >> /etc/apt/sources.list
#RUN apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys 92F371361A7ECE03
#RUN apt-get update -y
#RUN apt-get install software-properties-common python-rsvg wget -y && apt-get clean -y && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

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
RUN pip install numpy==1.13.3
RUN pip install scipy==1.2.1

# Python libs
RUN pip install networkx==2.2 \
 setuptools==41.0.1 \
 IPython==4.0 \
 pandas==0.24.2 \
 plotly==1.12.2 \
 tqdm==4.11.2 \
 joblib==0.13.2 \
 seaborn==0.7.1 \
 matplotlib==2.2.4 \
 scikit-learn==0.20.3 \
 simplejson==3.16.0 \
 pathlib==1.0.1 \
 more_itertools==5.0.0 \
 repoze.lru==0.7 \
 python-rake==1.4.5 \
 jellyfish==0.6.0

RUN pip install spacy==1.9
#RUN python -m spacy download en

#ADD id_rsa_pub /root/.ssh/authorized_keys
#RUN chmod 600  /root/.ssh/authorized_keys
#
# install Morfeusz - Polish Lang Lemmatizer
#RUN cd /tmp &&  wget -O - http://sgjp.pl/apt/sgjp.gpg.key|sudo apt-key add - && sudo add-apt-repository 'deb http://sgjp.pl/apt/ubuntu trusty main'
#RUN sudo apt-get update
#RUN sudo apt-get install python-morfeusz2 -y

#COPY /home/laugustyniak/github/phd/sentiment-backend/docker/start-notebook.sh /usr/local/bin/
#RUN chmod 600  /usr/local/bin/start_notebooks.sh
#
#CMD "/usr/local/bin/start_notebooks.sh"