# PyTorch Image Classification

## Introduction

Simple straightfoward repo for simple task of image classsification. Uses yaml config files to set hyperparameters for training. And loads data through list-style datasets.

## To Train

1. Copy `configs/config-example.yaml` and modify accordingly.
2. `python3 train.py --config <path to config>`

## To Test

1. Copy `configs/config-example.yaml` and modify accordingly. Make sure pointing to the right path of the trained weights.
2. `python3 test.py --config <path to config>`

## For Inference

1. Copy `configs/config-infer-example.yaml` and modify accordingly. Make sure pointing to the right path of the trained weights.
2. `python3 infer.py --config <path to config>`
3. Look at the example under the `if __name__=='__main__'` portion of `infer.py` and adapt to your own application accordingly. Main thing is instantiating the `Classifier` object with your config yaml file and running its `predict` method.

## Special notes

- Augmentations/preprocessing transformations are defined in their own `.yaml` files, supports [torchvision.transforms](https://pytorch.org/docs/stable/torchvision/transforms.html).