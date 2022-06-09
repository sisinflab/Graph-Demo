FROM nvidia/cuda:11.5.1-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update -y && \
    apt-get install -y unzip git curl software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get install -y python3.8 python3.8-distutils && \
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3.8 get-pip.py && \
    git clone https://github.com/sisinflab/Graph-Demo-CIKM2022.git && \
    pip install --upgrade pip && \
    pip install -r Graph-Demo-CIKM2022/requirements.txt && \
    pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.11.0+cu115.html

WORKDIR Graph-Demo-CIKM2022

RUN python3.8 start_experiments.py