FROM python:3.10-slim-bullseye

WORKDIR /root

RUN apt update -qq && \
    apt install -y build-essential \
    git \
    curl \
    wget \
    vim

# req.txtからパッケージをインストール
COPY requirements.txt /root/requirements.txt
RUN pip install --upgrade pip
RUN pip install -r /root/requirements.txt