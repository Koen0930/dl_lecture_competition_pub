FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /root

RUN apt update && \
    apt install -y \
    wget \
    bzip2 \
    build-essential \
    git \
    git-lfs \
    curl \
    ca-certificates \
    libsndfile1-dev \
    libgl1 \
    python3.10 \
    python3-pip

# req.txtからパッケージをインストール
COPY requirements.txt /root/requirements.txt
RUN pip install -r /root/requirements.txt && \
    pip uninstall -y numpy && \
    pip install numpy==1.26.4
