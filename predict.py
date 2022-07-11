# 进行预测

import os

from PIL import Image
from scipy.io import savemat

from unet import Unet

if __name__ == "__main__":
    unet = Unet()
    image_paths = os.listdir("./img")
    for image_path in image_paths:
        path = os.path.join("./img", image_path)
        image = Image.open(path)
        r_image = unet.detect_image(image)
        # jamming_pos = Unet.get_jamming_pos(unet, image_path)
        r_image.show()
