# Covid19 Diagnosis from recorded coughts
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/XxidroxX/Covid19-Diagnosis-from-recorded-coughts/graphs/commit-activity)
[![Ask Me Anything !](https://img.shields.io/badge/Ask%20me-anything-1abc9c.svg)](https://github.com/XxidroxX)
[![forthebadge made-with-python](http://ForTheBadge.com/images/badges/made-with-python.svg)](https://www.python.org/)
[![GitHub license](https://img.shields.io/github/license/XxidroxX/Covid19-Diagnosis-from-recorded-coughts)](https://github.com/XxidroxX/Covid19-Diagnosis-from-recorded-coughts/blob/main/LICENSE)
[![Github all releases](https://img.shields.io/github/downloads/XxidroxX/Covid19-Diagnosis-from-recorded-coughts/total)](https://github.com/XxidroxX/Covid19-Diagnosis-from-recorded-coughts/releases/)
[![GitHub forks](https://img.shields.io/github/forks/XxidroxX/Covid19-Diagnosis-from-recorded-coughts?style=social)](https://github.com/XxidroxX/Covid19-Diagnosis-from-recorded-coughts/network)
[![GitHub stars](https://img.shields.io/github/stars/XxidroxX/Covid19-Diagnosis-from-recorded-coughts?style=social)](https://github.com/XxidroxX/Covid19-Diagnosis-from-recorded-coughts/stargazers)
[![GitHub watchers](https://img.shields.io/github/watchers/XxidroxX/Covid19-Diagnosis-from-recorded-coughts?style=social)](https://github.com/XxidroxX/Covid19-Diagnosis-from-recorded-coughts/watchers/)
[![Open Source Love svg2](https://badges.frapsoft.com/os/v2/open-source.svg?v=103)]()

A simple neural network written with PyTorch to diagnostic covid19 from recorded coughs. Since the dataset is small, we do not achieve a high value of reliability but overall it is not so bad.

## Dataset
Dataset provided by: https://virufy.org/data.html
Locally, I have converted all the recorded audio tracks with Librosa from .mp3 to .jpg images. In this way, we can easily adopt a CNN.

## Network
Since we have few data, ResNet34 is a good choice for the network because a deeper model would lead to overfitting.
For what concerning the data augmentation, I used the classic transformations such as *RandomHorizontalFlip*, *RandomVerticalFlip* and *RandomGrayScale*.

## Installation
Download the files and extract them in a folder. 

## Usage
```console
foo@bar:~$ python path/to/the/main.py --epoch=25 --lr=0.001 --step=8 --gamma=0.2 --train=0.7 --clf=rf
```

## Parameters ##
Standard parameters, feel free to change all of them!
These are my parameters:
| Loss function | Learning rate | # Epochs | Step size | Gamma | pre-trained |
| --- | --- | --- |--- |--- | --------- |
| BCEWithLogits | 0.001 | 25 | 8 | 0.2 | True | 

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
If you want, you can send me your recorded cough to increase the dataset size. 

## TODO
- [x] choose the parameters from the cmd when you run the script without open the file .py
- [ ] better pre-processing phase (e.g. normalization or PCA).
- [ ] Curse of dimensionality. We have more features than samples and this can lead to overfit the model.
- [ ] from terminal we can choose the classifier from a list.
- [ ] dummy variables or simply LabelEncoder()? Dummy var. are in general better but PCA and some classifiers can have a problem with that.
- [ ] use keras instead of pytorch?
- [x] Implementing BCE Loss function
- [x] Code cleanup
