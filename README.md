# Covid19 Diagnosis from recorded coughts
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/XxidroxX/Covid19-Diagnosis-from-recorded-coughts/graphs/commit-activity)
[![Ask Me Anything !](https://img.shields.io/badge/Ask%20me-anything-1abc9c.svg)](https://github.com/XxidroxX)
[![forthebadge made-with-python](http://ForTheBadge.com/images/badges/made-with-python.svg)](https://www.python.org/)
[![GitHub license](https://img.shields.io/github/license/Naereen/StrapDown.js.svg)](https://github.com/XxidroxX/Covid19-Diagnosis-from-recorded-coughts/blob/main/LICENSE)
[![Github all releases](https://img.shields.io/github/downloads/Naereen/StrapDown.js/total.svg)](https://github.com/XxidroxX/Covid19-Diagnosis-from-recorded-coughts/releases/)
[![GitHub forks](https://img.shields.io/github/forks/Naereen/StrapDown.js.svg?style=social&label=Fork&maxAge=2592000)](https://github.com/XxidroxX/Covid19-Diagnosis-from-recorded-coughts/network)
[![GitHub stars](https://img.shields.io/github/stars/Naereen/StrapDown.js.svg?style=social&label=Star&maxAge=2592000)](https://github.com/XxidroxX/Covid19-Diagnosis-from-recorded-coughts/stargazers)
[![GitHub watchers](https://img.shields.io/github/watchers/Naereen/StrapDown.js.svg?style=social&label=Watch&maxAge=2592000)](https://github.com/XxidroxX/Covid19-Diagnosis-from-recorded-coughts/watchers/)
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
```python
python path/to/the/main.py
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
- [ ] choose the parameters from the cmd when you run the script without open the file .py
- [ ] Try with more combinations of parameters.
- [x] Create my first TODO.md  
- [x] Implementing BCE Loss function
- [x] Code cleanup
