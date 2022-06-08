FROM nvidia/cuda:11.5.1-cudnn8-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update -y

RUN apt-get install zip -y && \
    apt-get install unzip -y && \
    apt-get install nano -y && \
    apt-get install htop -y && \
    apt-get install git -y && \
    apt-get install curl -y

RUN apt-get install software-properties-common -y && \
    add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get install python3.8 -y && \
    apt-get install python3.8-venv python3-venv -y && \
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3.8 get-pip.py

RUN git clone https://github.com/sisinflab/Graph-Demo-CIKM2022.git

WORKDIR /Graph-Demp-CIKM2022

RUN python3.8 -m venv venv && \
    source venv/bin/activate && \
    pip install --upgrade pip && \
    pip install -r requirements.txt

RUN gdown 1HktsX2gBlox1QwOeDX39fKbNaMTKLg6j && \
    unzip data.zip && \
    rm data.zip
