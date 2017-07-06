FROM buildpack-deps:jessie

# ensure local python is preferred over distribution python
ENV PATH /usr/local/bin:$PATH

# http://bugs.python.org/issue19846
# > At the moment, setting "LANG=C" on a Linux system *fundamentally breaks Python 3*, and that's not OK.
ENV LANG C.UTF-8

# runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
		tcl \
		tk \
	&& rm -rf /var/lib/apt/lists/*

ENV GPG_KEY C01E1CAD5EA2C4F0B8E3571504C367C218ADD4FF
ENV PYTHON_VERSION 2.7.6

RUN set -ex \
	&& buildDeps=' \
		dpkg-dev \
		tcl-dev \
		tk-dev \
	' \
	&& apt-get update && apt-get install -y $buildDeps --no-install-recommends && rm -rf /var/lib/apt/lists/* \
	\
	&& wget -O python.tar.xz "https://www.python.org/ftp/python/${PYTHON_VERSION%%[a-z]*}/Python-$PYTHON_VERSION.tar.xz" \
	&& wget -O python.tar.xz.asc "https://www.python.org/ftp/python/${PYTHON_VERSION%%[a-z]*}/Python-$PYTHON_VERSION.tar.xz.asc" \
	&& export GNUPGHOME="$(mktemp -d)" \
	&& gpg --keyserver ha.pool.sks-keyservers.net --recv-keys "$GPG_KEY" \
	&& gpg --batch --verify python.tar.xz.asc python.tar.xz \
	&& rm -rf "$GNUPGHOME" python.tar.xz.asc \
	&& mkdir -p /usr/src/python \
	&& tar -xJC /usr/src/python --strip-components=1 -f python.tar.xz \
	&& rm python.tar.xz \
	\
	&& cd /usr/src/python \
	&& gnuArch="$(dpkg-architecture --query DEB_BUILD_GNU_TYPE)" \
	&& ./configure \
		--build="$gnuArch" \
		--enable-shared \
		--enable-unicode=ucs4 \
	&& make -j "$(nproc)" \
	&& make install \
	&& ldconfig \
	\
	&& apt-get purge -y --auto-remove $buildDeps \
	\
	&& find /usr/local -depth \
		\( \
			\( -type d -a -name test -o -name tests \) \
			-o \
			\( -type f -a -name '*.pyc' -o -name '*.pyo' \) \
		\) -exec rm -rf '{}' + \
	&& rm -rf /usr/src/python

# if this is called "PIP_VERSION", pip explodes with "ValueError: invalid truth value '<VERSION>'"
ENV PYTHON_PIP_VERSION 9.0.1

RUN set -ex; \
	\
	wget -O get-pip.py 'https://bootstrap.pypa.io/get-pip.py'; \
	\
	python get-pip.py \
		--disable-pip-version-check \
		--no-cache-dir \
		"pip==$PYTHON_PIP_VERSION" \
	; \
	pip --version; \
	\
	find /usr/local -depth \
		\( \
			\( -type d -a -name test -o -name tests \) \
			-o \
			\( -type f -a -name '*.pyc' -o -name '*.pyo' \) \
		\) -exec rm -rf '{}' +; \
	rm -f get-pip.py

# install "virtualenv", since the vast majority of users of this image will want it
RUN pip install --no-cache-dir virtualenv
RUN readlink -f $(which python) | xargs -I % sh -c 'echo -n "%: "; % -V'

#CMD ["python2"]

#todo: get ubuntu with python 2.7.6 and ipython/jupyter

USER root

# Graph-tool install
RUN echo "deb http://downloads.skewed.de/apt/trusty trusty universe" >> /etc/apt/sources.list
RUN echo "deb-src http://downloads.skewed.de/apt/trusty trusty universe" >> /etc/apt/sources.list
RUN apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys 92F371361A7ECE03
RUN apt-get update -y
RUN apt-get install software-properties-common python-rsvg wget -y && apt-get clean -y && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

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

#RUN sudo add-apt-repository ppa:webupd8team/java -y &&  apt-get update -y
#RUN echo debconf shared/accepted-oracle-license-v1-1 select true | debconf-set-selections && echo debconf shared/accepted-oracle-license-v1-1 seen true |  sudo debconf-set-selections
#RUN apt-get install oracle-java8-installer -y && apt-get clean -y && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
RUN apt-get install libxml2

#ENV JVM_OPT '-Xmx135G'

# Python libs
RUN pip install networkx pandas statsmodels BeautifulSoup4 unidecode justext plotly line_profiler memory_profiler tqdm joblib seaborn matplotlib scikit-learn
RUN pip install --force-reinstall spacy
RUN readlink -f $(which python) | xargs -I % sh -c 'echo -n "%: "; % -V'
RUN python -m spacy download en
#RUN python -m spacy download en_core_web_sm-2.0.0-alpha --direct

# install Morfeusz - Polish Lang Lemmatizer
RUN cd /tmp &&  wget -O - http://sgjp.pl/apt/sgjp.gpg.key|sudo apt-key add - && sudo add-apt-repository 'deb http://sgjp.pl/apt/ubuntu trusty main'
RUN sudo apt-get update
RUN sudo apt-get install python-morfeusz2 -y

# Python's nltk and other libs need this
RUN apt-get install libxml2-dev libxslt1-dev -y && pip install lxml

# Python libs for Aspect-based sentiment analysis
RUN apt-get update -y
RUN apt-get install python-yaml -y
RUN pip install http://pyyaml.org/download/pyyaml/PyYAML-3.09.tar.gz

RUN lsb_release -irc

# JAVA install
#RUN echo "deb http://http.debian.net/debian jessie-backports main" | \
#  sudo tee --append /etc/apt/sources.list.d/jessie-backports.list > /dev/null && \
#  apt-get update && \
#  apt-get install -t jessie-backports openjdk-8-jdk -y && \
#  update-java-alternatives -s java-1.8.0-openjdk-amd64 && \
##  apt-get install -y oracle-java8-installer && \
#  rm -rf /var/lib/apt/lists/* && \
#  rm -rf /var/cache/oracle-jdk8-installer

# Define commonly used JAVA_HOME variable
#ENV JAVA_HOME /usr/lib/jvm/java-8-oracle

# Install additional python libs necessary for ntlk installation from source
RUN pip install setuptools..

# Install NLTK
# WARNING script uses python2 !!
COPY /resources $HOME/resources
RUN sh $HOME/resources/install_nltk.sh

# Aspect-based SA lib
COPY /krzysztof-rajda-magisterka $HOME/sa
RUN sh $HOME/sa/edu_dependency_parser/tools/build_crf.sh
