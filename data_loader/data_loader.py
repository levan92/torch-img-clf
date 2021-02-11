
from torch.utils.data import DataLoader

import torchvision
from torchvision import datasets, transforms

from utils import parse_config

from .image_csv_dataset import ImageCSVDataset
from .transforms import build_transforms

PHASES = ['train', 'val', 'test']

def get_data_loaders_from_config(config):
    '''
    Returns a dict with the phases as keys and respective dataloaders as values.
    '''

    classes_txt = config['datasets']['classes']['classes_txt']

    phase2dataset = {}
    for phase, phase_dict in config['datasets']['sets'].items():
        assert phase in PHASES,f'Phases configured must be either {PHASES}.'
        print()
        print(phase)
        phase_aug_dicts =  parse_config(phase_dict['aug'], verbose=False)
        img_transforms = build_transforms(phase_aug_dicts)
        phase2dataset[phase] = ImageCSVDataset(phase_dict['csv'], classes_txt, transform=img_transforms, target_transform=None)

    phase2shuffle = {'train': True, 'val': False, 'test': False}
    bs = config['training']['batch_size']
    num_workers = config['training']['data_num_workers']

    dataloaders_dict = {}
    for phase, dataset in phase2dataset.items():
        dataloaders_dict[phase] = DataLoader( dataset, batch_size=bs, shuffle=phase2shuffle[phase], num_workers=num_workers, sampler=None )

    return dataloaders_dict, phase2dataset['train'].classes
