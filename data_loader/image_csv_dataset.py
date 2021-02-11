import csv
from pathlib import Path

from PIL import Image
from torchvision.datasets.vision import VisionDataset

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)

def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)

class ImageCSVDataset(VisionDataset):
    def __init__(self, csv_path, classes_txt, transform=None, target_transform=None, loader=default_loader):
        '''
        csv_path: str
            Path to csv file with '<image path>,<label>' lines.
        classes_txt: str
            Path to newline-separated list of classes (the order here will affect order in classifier)
        loader: Callable
            Function to load image from image path
        '''
        super().__init__(None,                                
                         transform=transform, 
                         target_transform=target_transform
                        )

        classes_path = Path(classes_txt)
        assert classes_path.is_file(),f'{classes_path} does not exist.'
        with classes_path.open('r') as rf:
            classes = [cl.strip() for cl in rf.readlines()]
        assert len(set(classes)) == len(classes),'Duplicated classes given'
        self.classes = classes
        self.class_to_idx = {classes[i]: i for i in range(len(classes))}

        csvpath = Path(csv_path)
        assert csvpath.is_file(),f'{csvpath} does not exist'
        paths = []
        targets = []
        with csvpath.open('r') as rf:
            for impath, label in csv.reader(rf, delimiter=','):
                assert label in classes,f'Label {label} given in csv file does not appear in given classes txt file'
                paths.append(impath)
                targets.append(self.class_to_idx[label])
        
        self.targets = targets
        self.samples = [tup for tup in zip(paths, targets)]

        self.loader = loader
        print('Image CSV Dataset created!')

    def __len__(self,):
        return len(self.samples)
    
    def __getitem__(self, idx):
        path, target = self.samples[idx]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target


if __name__ =='__main__':
    csv_path  = '/media/dh/HDD/person/KAIST_Multispectral_Pedestrian_Detection_Benchmark/KAIST_DH/kaist_day_train.csv'
    cls_txt = '/media/dh/HDD/person/KAIST_Multispectral_Pedestrian_Detection_Benchmark/KAIST_DH/sensor_classes.txt'
    icd = ImageCSVDataset(csv_path, cls_txt)