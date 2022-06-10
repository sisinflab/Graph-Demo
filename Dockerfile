FROM nvidia/cuda:10.2-cudnn8-devel-ubuntu18.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-key del 7fa2af80 && \
    apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub && \
    apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub && \
    apt-get update -y && \
    apt-get install -y unzip git curl software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get install -y python3.8 python3.8-distutils && \
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3.8 get-pip.py && \
    git clone https://github.com/sisinflab/Graph-Demo-CIKM2022.git && \
    pip install --upgrade pip && \
    pip install -r Graph-Demo-CIKM2022/requirements.txt && \
    pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.10.0+cu102.html

WORKDIR Graph-Demo-CIKM2022