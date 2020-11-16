from PIL import Image
import os
from torch.utils.data import Dataset
import pathlib
from skimage import io


class CovidDataset(Dataset):
    base_mp3_folder = "./img/"

    def __init__(self, root, train=True, transform=None):

        self.train = train  # training set or test set
        self.folder = "train"
        
        self.transform = transform
        self.data = []
        self.targets = []

        path_mp3 = self.base_mp3_folder + "train/"
        glob = '*/*.jpg'

        for file_path in pathlib.Path(path_mp3).glob(glob):
            img_name = os.path.join(file_path)
            if str(file_path).split("\\")[2] == "neg":
                classe = 0
            else:
                classe = 1

            image = io.imread(img_name)
            self.data.append(image)
            self.targets.append(classe)


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

        """
        if self.target_transform is not None:
            target = self.target_transform(target)
        """
        return img, target

    def __len__(self):
        return len(self.data)