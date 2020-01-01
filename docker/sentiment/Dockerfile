FROM tiangolo/uvicorn-gunicorn-fastapi:python3.7

RUN pip install tiny-tokenizer==3.1.0 \
    fastapi==0.45.0 \
    flair==0.4.4

COPY sentiment /app
WORKDIR /app

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]