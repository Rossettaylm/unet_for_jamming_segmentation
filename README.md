## 训练步骤：
1. 将打完标签的数据集放到datesets/before，运行json_to_dateset.py，生成的数据集在datesets/JPEGImages和datasets/SegmentationClass文件夹下
2. 将上述两个文件夹复制到VOCdevkit/VOC2007下，运行voc_annotation.py，在VOCdevkit/VOC2007/ImageSets文件夹底下生成训练文件信息
3. 进行训练，运行train.py文件

## 预测步骤
1. 将待识别的文件放到img目录底下
2. 在unet.py文件中修改训练生成的权值文件路径，默认在logs文件夹底下
3. 运行predict.py文件