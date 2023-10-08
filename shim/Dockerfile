FROM python:3.11-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends tini libmimalloc2.0 && rm -rf /var/lib/apt/lists/*
WORKDIR /usr/src/app

COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY *.py ./
COPY start_shim.sh ./
ENTRYPOINT ["tini", "--"]
CMD ["./start_shim.sh"]