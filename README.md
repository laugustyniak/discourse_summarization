# Aspect-based Sentiment Analysis using Rhetorical Structure Analysis (RST) 

## Installation with conda virtualenv

### Install anaconda for python 2

`conda create -n aspects python=2.7.8 anaconda`

### Problem with setuptool lib in anaconda and python 2.7.8, update it manually

`conda update setuptools`

### Update old packages for python 2.7.8
Sentiment model are serialized with scikit-learn 0.18

`conda install scikit-learn==0.18.2`

### Install other requirements

#### Must be install externally:

PyYAML <= 3.09
nltk<=2.0b9  special version to install and use by RST parser

The best way to install RST parser is execute this script:

`rhetorical-installation folder and rhetorical-parser-installation.sh`

#### Must be installed for ubuntu machine - description on web

morfeusz2 == 0.3.0

#### Requirements.txt installation

`pip install -r requirements.txt`

## Process documents
## Install crf

```
optional arguments:
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
```

## Run
`python run.py -input data/reviews/ipod.json -output results -batch 1000 -p 23`
