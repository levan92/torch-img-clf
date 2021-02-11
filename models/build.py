from pathlib import Path

import torch
import torch.nn as nn

from .resnets import resnet18, resnet50, resnext50_32x4d

def build_model(arch, weights=None, pretrained=True, num_classes=None, device='cuda:0', inference=False):
    '''
    Params
    ------
    weights: str
        Path to weights to be loaded. Takes precedence over pretrained weights. Defaults to None.
    pretrained: bool 
        To load imagenet pretrained weights or not. If weights are given, then this does not matter. Defaults to True. 
    num_classes: int
        Number of classes. If weights given, this must match what it was trained with. 
    device: str
        Device to move model to. Defaults to 'cuda:0', the first gpu device. 
    inference: bool
        Whether this is for inference or training. In Inference mode, a softmax operation will be added at the end of the model. Defults to False (training mode).
    '''
    if arch == 'resnet18':
        model = resnet18(pretrained=False, progress=True, inference=inference)
        if pretrained:
            state_dict = torch.load('weights/pretrained/resnet18-5c106cde.pth')
            model.load_state_dict(state_dict)
    elif arch == 'resnet50':
        model = resnet50(pretrained=False, progress=True, inference=inference)
        if pretrained:
            state_dict = torch.load('weights/pretrained/resnet50-19c8e357.pth')
            model.load_state_dict(state_dict)
    elif arch == 'resnext50':
        model = resnext50_32x4d(pretrained=False, progress=True, inference=inference)
        if pretrained:
            state_dict = torch.load('weights/pretrained/resnext50_32x4d-7cdf4587.pth')
            model.load_state_dict(state_dict)
    else:
        raise ValueError('Chosen architecture {} not supported'.format(arch))

    if num_classes and num_classes != 1000:
        num_fc_in_features = model.fc.in_features 
        model.fc = nn.Linear(num_fc_in_features, num_classes)

    if weights is not None and Path(weights).is_file():
        model.load_state_dict(torch.load(weights))

    model.to(device)
    print(f'{arch} model built!')
    return model

def build_model_from_config(config):
    return build_model(
                config['model']['architecture'], 
                weights = config['model']['weights'] if config['model']['weights'] != 'pretrained' else None,
                pretrained = config['model']['weights'] == 'pretrained', 
                num_classes = config['datasets']['classes']['num_classes'], 
                device = config['training']['device']
                )
