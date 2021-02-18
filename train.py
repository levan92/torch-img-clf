import time
import math
import copy
from pathlib import Path

import yaml
from tqdm import tqdm
import seaborn as sn

import torch
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter

from utils import parse_config
from models.build import build_model_from_config
from data_loader.data_loader import get_data_loaders_from_config
from losses import get_train_val_losses
from optimizers import get_optim_from_config, get_scheduler_from_config
from test import test_model

PHASES = ['train', 'val']

def train_model(model, dataloaders, losses, optimizer, scheduler, writer, config):
    num_epochs = config['training']['num_epochs']
    device = config['training']['device']
    verbose_steps = config['training']['verbose_steps']
    early_stopping_threshold = config['training']['early_stopping']

    phases = ['train', 'val']
    assert all([p in dataloaders.keys() for p in PHASES]),f'Dataloaders does not have all the needed phases {PHASES}.'
    assert all([p in losses.keys() for p in PHASES]),f'Losses does not have all the needed phases {PHASES}.'

    dataset_sizes = { s : len(dataloaders[s].dataset) for s in PHASES }

    total_steps_per_epoch = dataset_sizes['train'] // dataloaders['train'].batch_size + 1

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_loss = math.inf
    best_epoch = 1
    early_stopping_strike = 0

    since = time.time()
    try:
        for epoch in range(num_epochs):
            print('-' * 10)
            print('Epoch {}/{}'.format(epoch+1, num_epochs))
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in PHASES:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0
                running_corrects = 0

                criterion = losses[phase]

                # Iterate over data.
                for step, data in enumerate(dataloaders[phase]):
                    inputs, labels = data
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                            global_step = epoch * total_steps_per_epoch + (step+1)
                            writer.add_scalar(f"Loss/{phase}", loss.item(), global_step)

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                    if phase == 'train' and (step+1) % verbose_steps == 0:
                        num_imgs_so_far = (step+1)*dataloaders['train'].batch_size
                        verbose_loss = running_loss / num_imgs_so_far
                        verbose_acc = running_corrects.double() /num_imgs_so_far

                        print('[{}] Step: {}/{} | Loss: {:.4f} Acc: {:.4f}'.format(phase, step+1, total_steps_per_epoch, verbose_loss, verbose_acc))

                        writer.flush()


                if phase == 'train':
                    scheduler.step()
                    lr_now = scheduler.get_last_lr()
                    print('LR:', lr_now)

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                print('[{}] Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))
                
                writer.add_scalar(f"EpochLoss/{phase}", epoch_loss, epoch)
                writer.add_scalar(f"EpochAccuracy/{phase}", epoch_acc, epoch)
                writer.flush()

                # checkpointing / early stoppping logic
                if phase == 'val': 
                    if epoch_loss < best_loss:
                        best_acc = epoch_acc
                        best_loss = epoch_loss
                        best_model_wts = copy.deepcopy(model.state_dict())
                        best_epoch = epoch + 1
                        early_stopping_strike = 0 # reset
                        print('Best val checkpointed.')
                    else:
                        early_stopping_strike += 1
                        print('Val not best, strike:{}/{}'.format(early_stopping_strike, early_stopping_threshold))

            print()
            if early_stopping_strike >= early_stopping_threshold:
                print('Terminating training as val not best for {} strikes'.format(early_stopping_strike))
                break

    except KeyboardInterrupt:
        print('Training interupted manually!')
    finally:
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val acc: {:4f}'.format(best_acc))
        print('Best val loss: {:4f}'.format(best_loss))
        print('achieved at epoch {}/{}'.format(best_epoch, epoch))

        # load best model weights
        model.load_state_dict(best_model_wts)

    return model, best_acc, best_loss, best_epoch, epoch

def viz_to_tb(dataloader, writer, num_classes, display_num=4):
    from collections import defaultdict
    from torchvision.utils import make_grid
    import numpy as np
    from utils import tensor2numpy
    labels_count = {k:0 for k in range(num_classes)}
    imgs_dict = defaultdict(list)

    dl_iter = iter(dataloader)
    while all([v < display_num for v in labels_count.values()]):
        inputs, labels = next(dl_iter)
        for input_, label in zip(inputs, labels):
            label = int(label)
            if labels_count[label] < display_num:
                imgs_dict[label].append(input_)
            labels_count[label] += 1

    for label, imgs in imgs_dict.items():
        img_grid = make_grid(imgs)
        img_grid = tensor2numpy(img_grid)
        # img_grid = (img_grid * 255).astype(np.uint8)
        writer.add_image(f'example image for label {label}', img_grid, dataformats='HWC')

def main(args):
    config = parse_config(args.config)

    out_dir = Path(config['training']['save_dir']) / config['training']['save_context'] 
    out_dir.mkdir(exist_ok=True, parents=True)
    config_out =  out_dir / f"{config['training']['save_context']}_config.yaml"
    with config_out.open('w') as wf:
        yaml.dump(config, wf)
    print(f'Training config saved to {config_out}.')

    tb_logdir = out_dir / 'logdir'
    writer = SummaryWriter(log_dir=tb_logdir)

    model = build_model_from_config(config)
    dataloaders, classes = get_data_loaders_from_config(config)
    if config['datasets']['viz']:
        viz_to_tb(dataloaders['train'], writer, config['datasets']['classes']['num_classes'])
    losses = get_train_val_losses()
    optimizer = get_optim_from_config(model.parameters(), config)
    scheduler = get_scheduler_from_config(optimizer, config)

    model, best_acc, best_loss, best_epoch, total_epoch = train_model(model, dataloaders, losses, optimizer, scheduler, writer, config)

    weights_dir = out_dir / 'weights' 
    weights_dir.mkdir(parents=True, exist_ok=True)
    save_path = weights_dir / f"{config['training']['save_context']}_bestval_loss{best_loss:0.3f}_acc{best_acc:0.3f}_ep{best_epoch}of{total_epoch}.pth"
    torch.save(model.state_dict(), save_path)
    print(f'Best val weights saved to {save_path}')

    conf_mat, report, _, _, _ = test_model(model, dataloaders['test'], config,  classes=classes)
    
    test_dir = out_dir / 'test'
    test_dir.mkdir(exist_ok=True, parents=True)
    test_out  = test_dir /  f"{config['training']['save_context']}_clfreport.log"
    with test_out.open('w') as wf:
        wf.write(report)

    sn_plot = sn.heatmap(conf_mat, annot=True, fmt='g', xticklabels=classes, yticklabels=classes)
    test_out_cm  = test_dir /  f"{config['training']['save_context']}_confmat.jpg"
    sn_plot.get_figure().savefig(test_out_cm)


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', help='Path to config file', default='configs/config.yaml')
    args = ap.parse_args()

    main(args)