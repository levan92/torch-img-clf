import cv2
from tqdm import tqdm

from utils import parse_config
from data_loader.data_loader import get_data_loaders_from_config

def inverse_normalize(tensor, mean, std):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor

def viz(args):
    config = parse_config(args.config)
    dataloaders = get_data_loaders_from_config(config)

    for phase, dataloader in dataloaders.items(): 
        print(phase)
        next_phase = False
        for step, data in enumerate(tqdm(dataloader)):
            inputs, labels = data
            for input_, label in zip(inputs, labels):
                input_ = inverse_normalize(input_,mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
                img = input_.permute(1, 2, 0).numpy()
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                cv2.imshow(f'{label}', img)
                key = cv2.waitKey(0) 
                if key == ord('q'):
                    exit()
                elif key == ord('n'):
                    next_phase = True
                    cv2.destroyAllWindows()
                    break
            if next_phase:
                break
            

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', help='Path to config file', default='configs/config.yaml')
    args = ap.parse_args()

    viz(args)