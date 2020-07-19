# Discourse Summarization

## Installation with conda virtualenv

```
conda create -n aspects python=3.7 anaconda
```

## Activate conda env

```
source activate aspects
```

## Package installation

```
pip install -r requirements.txt
```

You can also add this repo to conda env using pip install trick 

```
pip install -e .
```

Afterwards you can import packages using aspect naming such as 

```python
import aspects
```

## Jupyter Notebook usage

```
jupyter notebook
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
