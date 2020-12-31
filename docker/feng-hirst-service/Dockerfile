FROM nlpbox/feng-hirst-rst-parser:2019-01-04

RUN apk add py3-pip && \
    pip3 install sh \
    gunicorn \
    uvicorn \
    fastapi

COPY . /opt/feng-hirst-service/

WORKDIR /opt/feng-hirst-service

CMD ["uvicorn", "app:parser_app", "--host", "0.0.0.0", "--port", "80"]