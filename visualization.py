from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from PIL import Image
from torch.nn import functional as F
from torchvision import transforms
from torchvision import utils as vutil

from unet import Unet


def showImage(image):
    npimage = image.numpy() / 255.0
    plt.figure(figsize=(8, 8))
    plt.axis('off')
    plt.imshow(np.transpose(npimage, (1, 2, 0)), interpolation='nearest')


def preprocessing(image):
    transform = transforms.Compose([
        transforms.Resize([512, 512]),
        transforms.ToTensor(),
    ])

    input_image = transform(image).unsqueeze(0)
    return input_image


activation = {}  # 保存获取的输出


def get_activation(name):
    def hook_func(module, input, output):
        data = output.detach()
        data = data.permute(1, 0, 2, 3)
        activation[name] = data
    return hook_func


def main():
    img_path = "img/repeater1.jpg"
    image = Image.open(img_path)
    # showImage(image)
    imageTensor = preprocessing(image)
    print(imageTensor.shape)

    unet = Unet()
    model = unet.net
    model.eval()
    for name, module in model.named_modules():
        if (isinstance(module, nn.MaxPool2d)):
            module.register_forward_hook(get_activation(name))
    _ = model(imageTensor)

    # plot
    for key, value in activation.items():
        vutil.save_image(value, f"{key}.png", pad_value=0.5)


if __name__ == "__main__":
    main()
