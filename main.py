import pathlib

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from skimage import io
from torch.autograd import Variable
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from CovidDataset import CovidDataset


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
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

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc['val']))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


if __name__ == "__main__":
    device = "cuda"

    train_transforms = transforms.Compose([transforms.Resize([224, 224]),
                                           transforms.RandomHorizontalFlip(p=0.5),
                                           transforms.ToTensor(),
                                           transforms.Normalize(0.1187, 0.2295)])

    train_dataset = CovidDataset("./", train=True, transform=train_transforms)
    train_loader = DataLoader(dataset=train_dataset, shuffle=True, num_workers=4, batch_size=11)

    val_dataset = CovidDataset("./", train=False, transform=train_transforms)
    val_loader = DataLoader(dataset=val_dataset, shuffle=True, num_workers=4, batch_size=11)
    print("Train samples: {}\nValidation samples: {}\nTotal samples: {}\n" .format (len(train_dataset), len(val_dataset), len(train_dataset)+len(val_dataset)))
    dataloaders = {'train': train_loader, 'val': val_loader}

    model_ft = models.resnet50(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    # Here the size of each output sample is set to 2.
    model_ft.fc = nn.Linear(num_ftrs, 2)
    model_ft = model_ft.to(device)

    #TODO: try to change loss function
    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.005, momentum=0.9, weight_decay=1e-5) #0.01

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=10, gamma=0.1)

    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=25)
    torch.save(model_ft, "./rete_with_val.pth")
    
    model = torch.load("./rete_with_val.pth")
    model.eval()
    path_mp3 = "img/test/"
    glob = '*.jpg'

    loader = transforms.Compose([transforms.ToTensor()])

    def image_loader(image_name):
        """load image, returns cuda tensor"""
        image = Image.open(image_name)
        image = loader(image).float()
        image = Variable(image, requires_grad=True)
        image = image.unsqueeze(0)  # this is for VGG, may not be needed for ResNet
        return image.cuda()  # assumes that you're using GPU


    for file_path in pathlib.Path("./img/test/").glob("*.jpg"):
        image = image_loader(file_path)
        image = model(image)
        _, pred = torch.max(image, 1)
        m = nn.Softmax(dim=1)
        print(str(file_path), pred.data.cpu().numpy(), m(image).data.cpu().numpy())
