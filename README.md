# Graph Collaborative Filtering in Elliot: from Reproducibility to Hyperparameter Exploration

This is the official implementation of the paper _Graph Collaborative Filtering in Elliot: from Reproducibility to Hyperparameter Exploration_, under review as demo paper at CIKM 2022.

**Authors**: Vito Walter Anelli, Tommaso Di Noia, Antonio Ferrara, Daniele Malitesta, Claudio Pomo

## Table of Contents
- [Abstract](#abstract)
- [Pre-requisites](#pre-requisites)
- [Installation](#installation)
- [Implemented Models](#implemented-models)
- [Demo Datasets](#demo-datasets)
- [Running Examples](#running-examples)
- [Results](#results)
- [Video Tutorial](#video-tutorial)
- [Contacts](#contacts)

## Abstract
Graph convolutional networks (GCNs) are taking over collaborative filtering-based recommendation by effectively distilling the collaborative signal throughout the user-item graph with the propagation of informative content from neighbor to ego nodes (i.e., the message-passing schema). In this demonstration, we show how to run complete experimental pipelines with six state-of-the-art graph recommendation models in Elliot (i.e., our framework for recommender system evaluation). We seek to highlight four main features, namely: (i) we support reproducibility in PyTorch Geometric (i.e., the library we use to implement the baselines), (ii) proposed baselines span across two GCN families, (iii) thanks to the hyperparameter tuning module in Elliot, we allow to easily run ablation studies on the number of explored hops, and (iv) we prepare a Docker image to provide an auto-consistent ecosystem for the running of experiments. Codes, datasets, and a video tutorial to install and launch the application are accessible at: https://github.com/sisinflab/Graph-Demo-CIKM2022.

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
| LightGCN   | LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation                        | He et al.   | 2020     | SIGIR          | https://dl.acm.org/doi/10.1145/3397271.3401063         |
| DGCF       | Disentangled Graph Collaborative Filtering                                                             | Wang et al. | 2020     | SIGIR          | https://dl.acm.org/doi/abs/10.1145/3397271.3401137     |
| LR-GCCF    | Revisiting Graph Based Collaborative Filtering: A Linear Residual Graph Convolutional Network Approach | Chen et al. | 2020     | AAAI           | https://ojs.aaai.org//index.php/AAAI/article/view/5330 |
| UltraGCN   | UltraGCN: Ultra Simplification of Graph Convolutional Networks for Recommendation                      | Mao et al.  | 2021     | CIKM           | https://dl.acm.org/doi/10.1145/3459637.3482291         |
| GFCF       | How Powerful is Graph Convolution for Recommendation?                                                  | Shen et al. | 2021     | CIKM           | https://dl.acm.org/doi/abs/10.1145/3459637.3482264     |

## Demo Datasets
| **Datasets**         | **Users** | **Items** | **Interactions** | **Official Link**                                 | **Download Link**                                                                 |
|----------------------|-----------|-----------|------------------|---------------------------------------------------|-----------------------------------------------------------------------------------|
| Movielens 1M         | 5,915     | 2,753     | 570,622          | https://grouplens.org/datasets/movielens/1m/      | [link](https://drive.google.com/drive/folders/1ZIAFa63TAP76D5qSRaxltN0jBwO3aPbE) |
| Amazon Digital Music | 8,328     | 6,275     | 99,400           | https://jmcauley.ucsd.edu/data/amazon/            | [link](https://drive.google.com/drive/folders/1ABWWOE9PONypZw1qV80VrRu2T7QQicRn) |
| Epinions             | 14,341    | 13,145    | 269,170          | https://snap.stanford.edu/data/soc-Epinions1.html | [link](https://drive.google.com/drive/folders/1uj7X5PdaHTbES-YcoKnQMEuuiLAjC-_k) |

## Running Examples
Here, we provide two possible examples to run and test our application.

First, let us clone this GitHub repository so that we can access to the YAML docker compose file.

```
git clone https://github.com/sisinflab/Graph-Demo-CIKM2022.git
```

```
cd Graph-Demo-CIKM2022/
```

### 1. Train and evaluate NGCF (explicit message-passing)
```
sudo docker compose run demo-cikm-2022
```
Then, select:

- Model: ```ngcf```
- Number of explore hops: ```3```
- Dataset: ```amazon digital music```


### 2. Train and evaluate GFCF (no message-passing)
```
sudo docker compose run demo-cikm-2022
```
Then, select:

- Model: ```gfcf```
- Number of explore hops: ```0```
- Dataset: ```amazon digital music```

## Video Tutorial
If you need a practical guide to install and launch our application, click on the image below to go to the video tutorial on YouTube.

Video link: https://www.youtube.com/watch?v=2Lx3kPO680I

<a href="https://www.youtube.com/watch?v=2Lx3kPO680I"><img src="video.png" align="left"></a>

## Contacts
- Vito Walter Anelli (vitowalter.anelli@poliba.it)
- Tommaso Di Noia (tommaso.dinoia@poliba.it)
- Antonio Ferrara (antonio.ferrara@poliba.it)
- Daniele Malitesta (daniele.malitesta@poliba.it)
- Claudio Pomo (claudio.pomo@poliba.it)
