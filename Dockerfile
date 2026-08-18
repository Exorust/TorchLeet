# Zero-install TorchLeet: notebooks + the grader, no local Python setup.
#   docker run --rm -p 8888:8888 ghcr.io/exorust/torchleet
FROM python:3.12-slim

# CPU-only torch keeps the image small; a GPU box can pip install a CUDA build.
RUN pip install --no-cache-dir \
      torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir jupyterlab torchvision matplotlib numpy

WORKDIR /torchleet
COPY . /torchleet
RUN pip install --no-cache-dir ./python

EXPOSE 8888
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", \
     "--allow-root", "--ServerApp.token=", "--ServerApp.password="]
