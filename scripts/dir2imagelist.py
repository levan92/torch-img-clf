import argparse
from pathlib import Path

from tqdm import tqdm

IMG_EXTS = ['.jpg','.jpeg','.png','.tiff','.tif','.bmp','.gif','.webp']
IMG_EXTS = [x.lower() for x in IMG_EXTS] + [x.upper() for x in IMG_EXTS]
print('Acceptable img extensions:', IMG_EXTS)

ap = argparse.ArgumentParser()
ap.add_argument('--dirs', nargs='+', help='paths to directories, each directory contains images belonging to one class.', required=True)
ap.add_argument('--classes', nargs='+', help='Classnames of corresponding directories. Defaults to int numbering starting from 0')
ap.add_argument('--out', help='output csv file. Defaults to all.csv in current directory.', default='all.csv')
args = ap.parse_args()

out_txt = Path(args.out)
with out_txt.open('w') as wf:
    for i, dir_ in enumerate(args.dirs):
        dir_path = Path(dir_) 
        assert Path(dir_).is_dir()
        classname = args.classes[i] if args.classes else i
        print(dir_)
        print(classname)

        for imagepath in tqdm(list(dir_path.glob('*'))):
            if imagepath.suffix not in IMG_EXTS:
                continue
            wf.write(f'{imagepath},{classname}\n')
