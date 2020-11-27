from PIL import Image
from torch.autograd import Variable
from torchvision import transforms


def image_loader(image_name):
    """

    :param image_name:
    :return:
    """
    # load image, returns cuda tensor
    image = Image.open(image_name)
    tr = transforms.ToTensor()
    image = tr(image).float()
    image = Variable(image, requires_grad=True)
    image = image.unsqueeze(0)  # this is for VGG, may not be needed for ResNet
    return image.cuda()  # assumes that you're using GPU

"""
Filename explanation:
* train/val: neg-date-CODE-cough-sex-age-nSample.jpg
* test     : date-CODE-cough-sex-age-nSample.jpg
"""
def my_func(df_a, truel, test=False):
    """

    :param df_a:
    :param truel:
    :param test:
    :return:
    """
    results   = []
    if not test:
        indexes   = [2, 4, 5]
    else:
        indexes = [1, 3, 4]

    for name in df_a:
        if "-"+truel.split("-")[indexes[0]]+"-" in name and \
                truel.split("-")[indexes[1]] + "-" + truel.split("-")[indexes[2]] in name:
            results.append(True)
        else:
            results.append(False)
    return results

def union_features(df, dataset, feature_extractor, table_selected):
    """

    :param df:
    :param dataset:
    :param feature_extractor:
    :param table_selected:
    :return:
    """
    X, y = list(), list()
    for image, label, img_name in dataset:
        """
        Feature extractor, combine with the data from csv
        """
        image = image.unsqueeze(0)
        outputs = feature_extractor(image)
        outputs = outputs.view(-1).tolist()
        sample = df.loc[my_func(df['cough_filename'], img_name), table_selected].values.tolist()
        outputs.extend(sample[0])
        X.append(outputs)
        y.append(label)
    return X, y