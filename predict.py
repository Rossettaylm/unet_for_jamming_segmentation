#----------------------------------------------------#
#进行预测
#----------------------------------------------------#
import time

import cv2
import numpy as np
from PIL import Image
from scipy.io import savemat

from unet import Unet

if __name__ == "__main__":
    #-------------------------------------------------------------------------#
    #  对应种类的颜色，到__init__函数里修改self.colors即可
    #-------------------------------------------------------------------------#
    unet = Unet()

    # image_path = "img/repeater10.jpg"
    image_path = input("请输入测试的图片路径(输出exit退出): ")
    do = True
    while do or image_path != "exit":
        do = False
        try:
            if image_path == "exit":
                break
            image = Image.open(image_path)
        except:
            print('Open Error! Try again!')
            image_path = input("请输入测试的图片路径(输出exit退出): ")
            continue
        else:
            r_image = unet.detect_image(image)
            jamming_pos = Unet.get_jamming_pos(unet, image_path)
            savemat('jamming_pos.mat', {'jamming_pos': jamming_pos})
            r_image.show()
            image_path = input("请输入测试的图片路径(输出exit退出): ")
