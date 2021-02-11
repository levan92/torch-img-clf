from pathlib import Path
import yaml

def parse_config(config_path, verbose=True):
    cp = Path(config_path)
    assert cp.is_file(),f'{cp} does not exist.'
    with cp.open('r') as rf:
        config = yaml.load(rf, Loader=yaml.FullLoader)
    if verbose:
        print(config)
    return config

def inverse_normalize(tensor, mean, std):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor

def tensor2numpy(tensor):
    tensor = inverse_normalize(tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    # CWH to HWC
    return tensor.permute(1, 2, 0).numpy()
