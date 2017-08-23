# Aspect-based Sentiment Analysis using Rhetorical Structure Analysis (RST) 

`usage: run.py [-h] [-input INPUT_FILE_PATH] [-output OUTPUT_FILE_PATH]
              [-sent_model SENT_MODEL_PATH] [-batch BATCH_SIZE]
              [-p MAX_PROCESSES]`

## Installation with conda virtualenv [WIP]

### Install anaconda for python 2

`conda create -n aspects python=2.7.8 anaconda`

### Problem with setuptool lib in anaconda and python 2.7.8, update it manually

`conda update setuptools`

 # Update old packages for python 2.7.8
`conda install scikit-learn==0.18.2`

### Install other requirements

#### Must be install externally - see rhetorical-installation folder and rhetorical-parser-installation.sh for:

PyYAML <= 3.09
nltk<=2.0b9  special version to install and use by RST parser

#### Must be installed for ubuntu machine - description on web

morfeusz2 == 0.3.0

#### Requirements.txt installation

`pip install -r requirements.txt`

``

## Install crf

## Process documents

`optional arguments:
  -h, --help            show this help message and exit
  -input INPUT_FILE_PATH
                        Path to the file with documents (json, csv, pickle)
  -output OUTPUT_FILE_PATH
                        Number of processes
  -sent_model SENT_MODEL_PATH
                        path to sentiment model
  -batch BATCH_SIZE     batch size for each process
  -p MAX_PROCESSES      Number of processes
`

## Exemplary execution
`python run.py -input data/reviews/ipod.json -output results -batch 1000 -p 23`
