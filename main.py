import pathlib
import torch
from sklearn.svm import SVC
import numpy as np
import torch.nn as nn
import torch.optim as optim
from sklearn.decomposition import PCA
import pandas as pd
from PIL import Image
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import models, transforms
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import time
import copy
from sklearn.neighbors import KNeighborsClassifier
from CovidDataset import CovidDataset
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

def to_onehot(targets, n_classes):
    return torch.eye(n_classes)[targets]

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_loss = 0.0
    plt.figure()
    accuracy = [[], []]
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels, _ in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    #
                    labels_oneshot = to_onehot(labels, 2).to(device)
                    #
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels_oneshot) #labels

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
            if phase == 'val':
                accuracy[1].append(epoch_acc)
            else:
                accuracy[0].append(epoch_acc)

            # deep copy the model, if two model have the same accuracy on val set we pick the one with the highest train acc.
            if phase == 'val' and (epoch_acc > best_acc or (epoch_acc == best_acc and epoch_loss <= best_loss)):
                best_acc = epoch_acc
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    plt.plot([i for i in range(0, num_epochs)], accuracy[1], label="Validation", c='red')
    plt.plot([i for i in range(0, num_epochs)], accuracy[0], label="Train", c='blue')
    plt.legend()
    plt.show()
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


if __name__ == "__main__":
    device = "cuda"
    #a = PrepareDataset()
    #a.createTrain()
    #a.createTest()

    """
    Create parameters from terminal
    """
    parameters = dict()
    for arg in sys.argv[1:]:
        parameter = arg[2:].split("=")[0]
        value = arg[2:].split("=")[1]
        if parameter == 'epoch':
            parameters['epochs'] = int(value)
        elif parameter == 'lr':
            parameters['lr'] = float(value)
        elif parameter == 'gamma':
            parameters['gamma'] = float(value)
        elif parameter == 'step':
            parameters['step_size'] = int(value)
        elif parameter == 'train':
            parameters['train_size'] = float(value)
    if len(parameters) != 5:
        raise ValueError('The number of parameters is wrong!!')

    trans = [transforms.RandomHorizontalFlip(p=0.5), transforms.RandomVerticalFlip(p=0.5), transforms.RandomGrayscale(p=0.5)]
    #TODO: normalize images?
    train_transforms = transforms.Compose([transforms.Resize([224, 224]),
                                           transforms.RandomChoice(trans),
                                           transforms.ToTensor()])

    val_transforms = transforms.Compose([transforms.ToTensor()])

    train_dataset = CovidDataset("./img/", train=True, transform=train_transforms, train_size=parameters['train_size'])
    train_loader = DataLoader(dataset=train_dataset, shuffle=True, num_workers=4, batch_size=11)

    val_dataset = CovidDataset("./img/", train=False, transform=val_transforms, train_size=parameters['train_size'])
    val_loader = DataLoader(dataset=val_dataset, shuffle=True, num_workers=4, batch_size=11)
    print("Train samples: {}\nValidation samples: {}\nTotal samples: {}\n".format(len(train_dataset), len(val_dataset),
                                                                                  len(train_dataset) + len(
                                                                                      val_dataset)))
    dataloaders = {'train': train_loader, 'val': val_loader}

    model_ft = models.resnet34(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    # Here the size of each output sample is set to 2.
    model_ft.fc = nn.Linear(num_ftrs, 2)
    model_ft = model_ft.to(device)
    
    """
    The parameter pos_weight:
    * >1 -> increase recall
    * <1 -> increase precision
    """
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(5.))

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=parameters['lr'], momentum=0.9, weight_decay=1e-5)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=parameters['step_size'], gamma=parameters['gamma'])

    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=parameters['epochs'])
    torch.save(model_ft, "./net_with_bce.pth")

    model = torch.load("./net_with_bce.pth")
    model.to("cpu")
    model.eval()

    ### strip the last layer
    feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])

    #TODO: we can use batch mode instead of one image at time
    def image_loader(image_name):
        #load image, returns cuda tensor
        image = Image.open(image_name)
        image = val_transforms(image).float()
        image = Variable(image, requires_grad=True)
        image = image.unsqueeze(0)  # this is for VGG, may not be needed for ResNet
        return image.cuda()  # assumes that you're using GPU

    data, label, age, gender, \
    medical_history, smoker, patient_symptoms, filename = list(), list(), \
                                                          list(), list(), list(), list(), list(), list()

    with open("labels.csv") as csv:
        for index, line in enumerate(csv):
            if index == 0:
                features_name = line.replace("\n", "").replace("corona_test,", "").split(",")
            else:
                features = line.replace("\n", "").split('"')
                data.append(features[0].split(",")[0])
                #label.append(features[0].split(",")[1])
                age.append(int(features[0].split(",")[2]))
                gender.append(features[0].split(",")[3].lower())

                medical_history.append(features[1][:-1].lower())
                smoker.append(features[2][1:-1].lower())
                patient_symptoms.append(features[3][:-1].lower())
                filename.append(features[4][1:].replace(".mp3", ""))

    data = {features_name[0]: data,
            #features_name[1]: label,
            features_name[1]: age,
            features_name[2]: gender,
            features_name[3]: medical_history,
            features_name[4]: smoker,
            features_name[5]: patient_symptoms,
            features_name[6]: filename}

    df = pd.DataFrame(data, columns=features_name)

    df['smoker'] = LabelEncoder().fit_transform(df['smoker'])

    med_history = list()
    # medical_history, unique values
    for i in df.medical_history:
        for j in i.split(","):
            if j not in med_history and j != 'none':
                med_history.append(i)

    for index, mh in enumerate(med_history):
        df['history_'+str(index)] = df.medical_history.str.contains(mh).astype(int)

    med_sym = list()
    # medical_history, unique values
    for i in df.patient_reported_symptoms:
        for j in i.split(","):
            if j not in med_sym and j != 'none':
                med_sym.append(j)
    for index, mh in enumerate(med_sym):
        df['symptom_' + str(index)] = df.patient_reported_symptoms.str.contains(mh).astype(int)

    df['gender'] = LabelEncoder().fit_transform(df['gender'])

    #DELETE OLD FEATURES
    df.drop(columns=["medical_history", "patient_reported_symptoms"], inplace=True)
    # Table for machine learning algo.
    table_selected = [features_name[1], features_name[2], features_name[4]]
    table_selected.extend(['history_'+str(i) for i, _ in enumerate(med_history)])
    table_selected.extend(['symptom_'+str(i) for i, _ in enumerate(med_sym)])
    """
    Filename explanation: neg-date-CODE-cough-sex-age-nSample.mp3
    """
    def my_func(df_a, truel):
        results = []
        for name in df_a:
            if truel.split("-")[0]+"-"+truel.split("-")[1] in name and \
                    truel.split("-")[4] + "-" + truel.split("-")[5] in name:
                results.append(True)
            else:
                results.append(False)
        return results

    X_train, X_test, y_train, y_test = list(), list(), list(), list()
    for image, label, img_name in train_loader.dataset:
        """
        Feature extractor, combine with the data from csv
        """
        image = image.unsqueeze(0)
        outputs = feature_extractor(image)
        outputs = outputs.view(-1).tolist()

        sample = df.loc[my_func(df['cough_filename'], img_name), table_selected].values.tolist()
        outputs.extend(sample[0])
        X_train.append(outputs)
        y_train.append(label)

    for image, label, img_name in val_loader.dataset:
        image = image.unsqueeze(0)#.to(device)
        outputs = feature_extractor(image)
        outputs = outputs.view(-1).tolist()
        sample = df.loc[my_func(df['cough_filename'], img_name), table_selected].values.tolist()
        outputs.extend(sample[0])
        X_test.append(outputs)
        y_test.append(label)

    """
    TODO: add more classifier. From terminal an user can select the clf.
    """
    knn = KNeighborsClassifier(n_neighbors=7)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    print(classification_report(y_test, y_pred))
    """
    These lines are for the prediction of my sample
    """
    for file_path in pathlib.Path("./img/test/").glob("*.jpg"):
        image = image_loader(file_path).to("cpu")
        img_name = str(file_path).split("\\")[2].replace(".jpg", "")
        outputs = feature_extractor(image)
        outputs = outputs.view(-1).tolist()
        sample = df.loc[my_func(df['cough_filename'], img_name), table_selected].values.tolist()
        outputs.extend(sample[0])
        y_pred = knn.predict([outputs])
        print(str(file_path), y_pred)
