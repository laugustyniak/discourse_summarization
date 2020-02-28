# Aspect-based Sentiment Analysis using Rhetorical Structure Analysis (RST) 

## Installation with conda virtualenv

```
conda create -n aspects python=3.7 anaconda
```

## Activate conda env

```
source activate aspects
```

## Requirements.txt installation

```
pip install -r requirements.txt
```

## Run dockers with API for sentiment, RST parsing and aspect extraction

```
cd docker
./restart.sh
```

## Prapare conceptnet.io pickle and amazon reviews
```
python aspects/io.parser.py
```

## Run
```
python run.py -input data/reviews/ipod.json -output results -batch 1000 -p 23
```
