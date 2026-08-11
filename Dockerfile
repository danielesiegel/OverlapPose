FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir ".[ros]"

# Index and manifest data are expected on mounted volumes, e.g.
#   docker run -v /data:/data -v /index:/index ghcr.io/world-archive/overlap \
#     index /data --index /index/corpus.ovl
EXPOSE 8377
ENTRYPOINT ["overlap"]
CMD ["--help"]
