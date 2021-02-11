# PyTorch Image Classification

## Introduction

Simple straightfoward repo for simple task of image classsification. Uses yaml config files to set hyperparameters for training. And loads data through list-style datasets.

## To train

1. Copy `configs/config-example.yaml` and modify accordingly.
2. `python3 train.py --config <path to config>` 

## Special notes

- Augmentations are defined in their own `.yaml` files, supports [torchvision.transforms](https://pytorch.org/docs/stable/torchvision/transforms.html).