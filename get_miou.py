import os

from PIL import Image
from tqdm import tqdm

from unet import Unet
from utils.utils_metrics import compute_mIoU

if __name__ == "__main__":
    num_classes = 3
    name_classes = ["_background_", "target", "jamming"]
    VOCdevkit_path = 'VOCdevkit'

    image_ids = open(os.path.join(
        VOCdevkit_path, "VOC2007/ImageSets/Segmentation/val.txt"), 'r').read().splitlines()
    gt_dir = os.path.join(VOCdevkit_path, "VOC2007/SegmentationClass/")
    pred_dir = "miou_out"

    if not os.path.exists(pred_dir):
        os.makedirs(pred_dir)

    print("Load model.")
    unet = Unet()
    print("Load model done.")

    print("Get predict result.")
    for image_id in tqdm(image_ids):
        image_path = os.path.join(
            VOCdevkit_path, "VOC2007/JPEGImages/"+image_id+".jpg")
        image = Image.open(image_path)
        image = unet.get_miou_png(image)
        image.save(os.path.join(pred_dir, image_id + ".png"))
    print("Get predict result done.")

    print("Get miou.")
    compute_mIoU(gt_dir, pred_dir, image_ids,
                 num_classes, name_classes)  # 执行计算mIoU的函数
    print("Get miou done.")
