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


## Video Tutorial
If you need a practical guide to install and launch our application, clik on the image below to go to the video tutorial on YouTube.

<a href="https://www.youtube.com/watch?v=2Lx3kPO680I"><img src="video.png" align="left" width="50%"></a>
