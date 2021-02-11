from torch import nn

def get_train_val_losses():
    losses = {'train': nn.CrossEntropyLoss(),
              'val': nn.CrossEntropyLoss()
              }
    return losses
