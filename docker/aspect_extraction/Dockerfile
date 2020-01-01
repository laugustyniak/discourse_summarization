FROM tiangolo/uvicorn-gunicorn-fastapi:python3.6

RUN pip install fastapi==0.45.0 \
    git+https://www.github.com/keras-team/keras-contrib.git#egg=keras-contrib \
    -e git+https://github.com/NervanaSystems/ngraph-python.git#egg=ngraph \
    keras==2.1.5 \
    numpy \
    requests \
    scipy \
    spacy>=2.2 \
    tensorflow==1.8.0 \
    tqdm

RUN python -m spacy download en_core_web_sm

COPY aspect_extraction /app
WORKDIR /app

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]