FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update -y && \
    apt-get install -y unzip git curl software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get install -y python3.8 python3.8-distutils && \
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3.8 get-pip.py && \
    git clone https://github.com/sisinflab/Graph-Demo.git && \
    apt-get install -y python3.8-dev && \
    pip install --upgrade pip && \
    pip install -r Graph-Demo/requirements.txt && \
    pip install -r Graph-Demo/requirements_torch_geometric.txt

WORKDIR Graph-Demo
