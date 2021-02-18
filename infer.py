from pathlib import Path

import torch
from torchvision import transforms
import cv2
import numpy as np

if __name__ == '__main__':
    from utils import parse_config
    from models.build import build_model
    from data_loader.transforms import build_transforms
else:
    from .utils import parse_config
    from .models.build import build_model
    from .data_loader.transforms import build_transforms

def build_model_from_infer_config(config):
    return build_model(
                config['model']['architecture'], 
                weights = config['model']['weights'] if config['model']['weights'] != 'pretrained' else None,
                pretrained = config['model']['weights'] == 'pretrained', 
                num_classes = config['datasets']['classes']['num_classes'], 
                device = config['infer']['device'],
                inference=True,
                )

def build_preproc_transforms():
    '''
    Assumed preprocessing transformations here. Change accordingly if different preproc was used during model training.
    '''
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])

def build_transforms_from_config(preproc_config_path):
    aug_dict = parse_config(preproc_config_path)
    return build_transforms(aug_dict)

def get_classes(classes_txt):
    classes_path = Path(classes_txt)
    assert classes_path.is_file(),f'{classes_path} does not exist.'
    with classes_path.open('r') as rf:
        classes = [cl.strip() for cl in rf.readlines()]
    assert len(set(classes)) == len(classes),'Duplicated classes given'
    return classes

class Classifier:
    def __init__(self, config_path):
        '''
        Params
        ------
        config_path: str
            path to inference config file
        '''
        config = parse_config(config_path)
        self.model = build_model_from_infer_config(config)
        self.model.eval()

        self.classes = get_classes(config['datasets']['classes']['classes_txt'])

        self.device = config['infer']['device']
        self.bgr = config['infer']['bgr']
        self.input_size = config['model']['input_size']
        self.img_transforms = build_preproc_transforms() 
        self.batch_size = config['infer']['batch_size']

        self.predict([np.zeros((224,224,3), dtype=np.uint8)]*self.batch_size)

    def preprocess(self, frame):
        if self.bgr:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, tuple(self.input_size), interpolation=cv2.INTER_LINEAR)
        return self.img_transforms(frame).to(torch.device(self.device))

    def batch_preprocess(self, frames):
        inputs = []
        for frame in frames:
            inputs.append(self.preprocess(frame))
        return torch.stack(inputs)

    def postprocess(self, preds):
        confs, inds = torch.max(preds,1)
        return confs.numpy(), inds.numpy()

    def get_class_preds(self, pred_indices):
        return [self.classes[i] for i in pred_indices]

    def forward(self, inputs):
        with torch.no_grad():
            all_preds = torch.zeros(0,dtype=torch.float, device='cpu')
            for batch in torch.split(inputs, self.batch_size):
                preds = self.model(batch)
                all_preds = torch.cat([all_preds, preds.cpu()])
        return all_preds

    def predict(self, frames):
        '''
        Params
        -----
        frames : Sequence of frames
        
        Returns
        ------
        Confidences, Argmax Indices. 
        To get respective classnames of each index, can use `get_class_preds` method on the indices. 
        '''
        inputs = self.batch_preprocess(frames)
        preds = self.forward(inputs)
        return self.postprocess(preds)

if __name__ == '__main__':    
    import time
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', help='Path to config file', default='configs/config.yaml')
    ap.add_argument('--imgs', nargs='+')
    args = ap.parse_args()

    clf = Classifier(args.config)
    images = [ cv2.imread(p) for p in args.imgs ] 
    pred_confs, pred_inds = clf.predict(images)
    pred_classes = clf.get_class_preds(pred_inds)

    for cl, conf in zip(pred_classes, pred_confs):
        print(cl, conf)

    ''' timing script
    images_orig = [ cv2.imread(p) for p in args.imgs ] 
    reps = 20
    for duplicate in [1, 8, 16, 32, 64, 128, 256, 512, 1024]:
        images = images_orig * duplicate

        print(f'num of images: {len(images)}')

        tic = time.perf_counter()
        for _ in range(reps):
            preds = clf.predict(images)

        toc = time.perf_counter()
        print(preds.shape)
        total_time = (toc - tic) / reps
        time_per = total_time / len(images)
        print('total time', total_time)
#     print('time per img', time_per)
    '''