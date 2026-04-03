FROM python:3.12-slim

# Build tippecanoe from source
RUN apt-get update && apt-get install -y \
    git build-essential libsqlite3-dev zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 https://github.com/felt/tippecanoe.git /tmp/tippecanoe \
    && cd /tmp/tippecanoe \
    && make -j$(nproc) \
    && make install \
    && rm -rf /tmp/tippecanoe

# Install Python deps
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy surveyor package + metamorphic server
COPY ../surveyor /app/surveyor
COPY ../species_db /app/species_db
COPY ../knowledge /app/knowledge
COPY serve.py /app/serve.py

WORKDIR /app

ENV DATA_DIR=/data
ENV SURVEYOR_OUTPUT=/data/output
ENV SPECIES_DB=/app/species_db/species.json

EXPOSE 8000

CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8000"]
