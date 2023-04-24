# An Out-of-the-Box Application for Reproducible Graph Collaborative Filtering extending the Elliot Framework

This is the official implementation of the paper _An Out-of-the-Box Application for Reproducible Graph Collaborative Filtering extending the Elliot Framework_, under review as demo paper at UMAP 2023.

**Authors**: Vito Walter Anelli, Tommaso Di Noia, Antonio Ferrara, Daniele Malitesta, Claudio Pomo

## Table of Contents
- [Abstract](#abstract)
- [Pre-requisites](#pre-requisites)
- [Installation](#installation)
- [Implemented Models](#implemented-models)
- [Demo Datasets](#demo-datasets)
- [Running Examples](#running-examples)
- [Video Tutorial](#video-tutorial)
- [Contacts](#contacts)

## Abstract
Graph convolutional networks (GCNs) are taking over collaborative filtering-based recommendation. Their message-passing schema effectively distills the collaborative signal throughout the user-item graph by propagating informative content from neighbor to ego nodes. In this demonstration, we show how to run complete experimental pipelines with six state-of-the-art graph recommendation models in Elliot (i.e., our framework for recommender system evaluation). We seek to highlight four main features, namely: (i) we support reproducibility in PyTorch Geometric (i.e., the library we use to implement the baselines); (ii) reproduced graph models span across various GCN families; (iii) we allow to easily run ablation studies on the number of explored hops; (iv) we prepare a Docker image to provide a self-consistent ecosystem for the running of experiments. Codes, datasets, and a video tutorial to install and launch the application are accessible at https://github.com/sisinflab/Graph-Demo.

## Pre-requisites
Before installing and running our application, please make sure you have the proper NVIDIA drivers installed on your local machine. One of the possible ways to check it (on Ubuntu 18.04) is to run the command:

```
nvidia-smi
```

If everything works smoothly, you should have a similar output:

```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 470.129.06   Driver Version: 470.129.06   CUDA Version: 11.4     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  Tesla T4            Off  | 00000001:00:00.0 Off |                    0 |
| N/A   40C    P0    28W /  70W |      0MiB / 15109MiB |      4%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+
```

## Installation
These are the following steps to install our application (tested on Ubuntu 18.04):

### 1. Install Docker Engine
Reference link: https://docs.docker.com/engine/install/ubuntu/

Commands:

```
sudo apt-get update
```

```
sudo apt-get install \
    ca-certificates \
    curl \
    gnupg \
    lsb-release
```

```
sudo mkdir -p /etc/apt/keyrings
```

```
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
```

```
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
```

```
sudo apt-get update
```

```
sudo apt-get install docker-ce docker-ce-cli containerd.io docker-compose-plugin
```

### 2. Install NVIDIA Container Toolkit
Reference link: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#setting-up-nvidia-container-toolkit

Commands:
```
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
      && curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
      && curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
            sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
            sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
```

```
sudo apt-get update
```

```
sudo apt-get install -y nvidia-docker2
```

```
sudo systemctl restart docker
```

```
sudo docker run --rm --gpus all nvidia/cuda:11.0.3-base-ubuntu20.04 nvidia-smi
```

Expected output:
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 450.51.06    Driver Version: 450.51.06    CUDA Version: 11.0     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  Tesla T4            On   | 00000000:00:1E.0 Off |                    0 |
| N/A   34C    P8     9W /  70W |      0MiB / 15109MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+
```

### 3. Install Docker Compose
Reference link: https://www.digitalocean.com/community/tutorials/how-to-install-and-use-docker-compose-on-ubuntu-22-04

Commands:
```
mkdir -p ~/.docker/cli-plugins/
```

```
curl -SL https://github.com/docker/compose/releases/download/v2.6.0/docker-compose-linux-x86_64 -o ~/.docker/cli-plugins/docker-compose
```

```
chmod +x ~/.docker/cli-plugins/docker-compose
```

```
docker compose version
```

Expected output:
```
Docker Compose version v2.6.0
```

## Implemented Models

| **Models** | **Title**                                                                                              | **Authors** | **Year** | **Conference** | **Link DOI**                                           |
|------------|--------------------------------------------------------------------------------------------------------|-------------|----------|----------------|--------------------------------------------------------|
| NGCF       | Neural Graph Collaborative Filtering                                                                   | Wang et al. | 2019     | SIGIR          | https://dl.acm.org/doi/10.1145/3331184.3331267         |
| DGCF       | Disentangled Graph Collaborative Filtering                                                             | Wang et al. | 2020     | SIGIR          | https://dl.acm.org/doi/abs/10.1145/3397271.3401137     |
| LightGCN   | LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation                        | He et al.   | 2020     | SIGIR          | https://dl.acm.org/doi/10.1145/3397271.3401063         |
| SGL    | Self-supervised graph learning for recommendation | Wu et al. | 2021     | SIGIR           | https://doi.org/10.1145/3404835.3462862 |
| UltraGCN   | UltraGCN: Ultra Simplification of Graph Convolutional Networks for Recommendation                      | Mao et al.  | 2021     | CIKM           | https://dl.acm.org/doi/10.1145/3459637.3482291         |
| GFCF       | How Powerful is Graph Convolution for Recommendation?                                                  | Shen et al. | 2021     | CIKM           | https://dl.acm.org/doi/abs/10.1145/3459637.3482264     |

## Demo Datasets
| **Datasets**         | **Users** | **Items** | **Interactions** | **Official Link**                                 | **Download Link**                                                                 |
|----------------------|-----------|-----------|------------------|---------------------------------------------------|-----------------------------------------------------------------------------------|
| Gowalla         | 29,858     | 40,981     | 1,027,370          |  https://snap.stanford.edu/data/loc-gowalla.html    | [link](https://drive.google.com/file/d/1H635ObV5V1U1m2Xj5Xjgdr_pzmB7hEWM/view?usp=share_link) |
| Yelp 2018 | 31,668     | 38,048     | 1,561,406           |     https://www.yelp.com/dataset        | [link](https://drive.google.com/file/d/1i191hLKNRxdRAYmJFIy4l9VapLLn83m5/view?usp=share_link) |
| Amazon Book             | 52,643    | 91,599    | 2,984,108          | https://jmcauley.ucsd.edu/data/amazon/ | [link](https://drive.google.com/file/d/1H75-B1gbfjYNaTCGYX3NX4V85pobT8yh/view?usp=share_link) |

## Running Examples
Here, we provide two possible examples to run and test our application.

First, let us clone this GitHub repository so that we can access to the YAML docker compose file.

```
git clone https://github.com/sisinflab/Graph-Demo-WSDM2023.git
```

```
cd Graph-Demo-WSDM2023/
```

### 1. Train and evaluate NGCF (explicit message-passing)
```
sudo docker compose run demo-graph
```
Then, select:

- Model: ```ngcf```
- Dataset: ```gowalla```


### 2. Train and evaluate GFCF (no message-passing)
```
sudo docker compose run demo-graph
```
Then, select:

- Model: ```gfcf```
- Dataset: ```gowalla```

## Video Tutorial
If you need a practical guide to install and launch our application, click on the image below to go to the video tutorial on YouTube.

Video link: https://youtu.be/Zeet08LNVBg

<a href="https://youtu.be/Zeet08LNVBg"><img src="video_wsdm.png" align="left"></a>

## Contacts
- Daniele Malitesta (daniele.malitesta@poliba.it)
- Claudio Pomo (claudio.pomo@poliba.it)
- Vito Walter Anelli (vitowalter.anelli@poliba.it)
- Tommaso Di Noia (tommaso.dinoia@poliba.it)
- Antonio Ferrara (antonio.ferrara@poliba.it)
