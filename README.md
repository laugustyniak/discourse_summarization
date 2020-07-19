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
...
```

## Download all necessary resources via DVC 

```
dvc pull
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

## Build and run discourse summarization docker

```
./build.sh
./start.sh
```

If you are using VS Code or PyCharm you can setup docker `discourse-summarization` image as your python interpreter  

## Prepare conceptnet.io pickle and amazon reviews
```
python aspects/io.parser.py
```

## Run
```
python run.py -input data/reviews/ipod.json -output results -batch 1000 -p 23
```
