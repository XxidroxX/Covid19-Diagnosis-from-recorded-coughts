from PIL import Image
import os
from torch.utils.data import Dataset
import pathlib
from skimage import io
import sklearn

class CovidDataset(Dataset):

    def __init__(self, root, train=True, transform=None, train_size=0.7):

        self.train = train  # training set or test set
        self.folder = "train"
        self.root = root
        self.transform = transform
        self.data = []
        self.targets = []

        path_mp3 = self.root + "train/"
        glob = '*/*.jpg'

        list_of_samples, list_of_targets = [], []

        for count, file_path in enumerate(pathlib.Path(path_mp3).glob(glob)):
            if str(file_path).split("\\")[2] == "neg":
                classe = 0
            else:
                classe = 1

            img_name = os.path.join(file_path)
            image = io.imread(img_name)
            list_of_samples.append(image)
            list_of_targets.append(classe)

        list_of_samples, list_of_targets = sklearn.utils.shuffle(list_of_samples, list_of_targets)

        if self.train:
            self.data = list_of_samples[:round(train_size*len(list_of_samples))]
            self.targets = list_of_targets[:round(train_size*len(list_of_samples))]
        else:
            self.data = list_of_samples[round(train_size*len(list_of_samples)):]
            self.targets = list_of_targets[round(train_size*len(list_of_samples)):]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.data)
