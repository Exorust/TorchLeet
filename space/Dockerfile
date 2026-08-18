# Single-port Space: Gradio catalogue at /, JupyterLab proxied at /lab.
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/*

# CPU wheel: Spaces have no GPU, and the CUDA build is ~5x the image size.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY space/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

WORKDIR /torchleet
COPY . /torchleet
RUN pip install --no-cache-dir ./python

# HF Spaces run as uid 1000 and the lab needs a writable home + notebook tree.
RUN useradd -m -u 1000 user && chown -R user:user /torchleet
USER user
ENV HOME=/home/user \
    TORCHLEET_ROOT=/torchleet \
    TORCHLEET_LAB_BASE=/lab \
    PORT=7860 \
    LAB_PORT=8888

WORKDIR /torchleet/space
EXPOSE 7860
CMD ["./start.sh"]
