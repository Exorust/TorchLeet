# Zero-install TorchLeet: notebooks plus the grader, no local Python setup.
#   docker run --rm -p 8888:8888 ghcr.io/exorust/torchleet
FROM python:3.12-slim

# torch AND torchvision both come from the CPU wheel index. Taking torchvision
# from the default index instead risks a build compiled against a different
# torch, which fails at import with an obscure symbol error.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
      torch torchvision \
 && pip install --no-cache-dir jupyterlab matplotlib numpy

WORKDIR /torchleet
COPY . /torchleet
RUN pip install --no-cache-dir ./python

EXPOSE 8888
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", \
     "--allow-root", "--ServerApp.token=", "--ServerApp.password="]
