#----------------------------------------------------#
# 进行预测
#----------------------------------------------------#

import os

from PIL import Image
from scipy.io import savemat

from unet import Unet

if __name__ == "__main__":
    #-------------------------------------------------------------------------#
    #  对应种类的颜色，到__init__函数里修改self.colors即可
    #-------------------------------------------------------------------------#
    unet = Unet()

    images = os.listdir("./img")
    for image in images:
        image_path = os.path.join("./img", image)
        image = Image.open(image_path)
        r_image = unet.detect_image(image)
        r_image.show()
