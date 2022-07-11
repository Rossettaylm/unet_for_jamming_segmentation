import os

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader

from nets.unet import Unet
from nets.unet_training import weights_init
from utils.callbacks import LossHistory
from utils.dataloader import UnetDataset, unet_dataset_collate
from utils.utils_fit import fit_one_epoch

if __name__ == "__main__":
    # 是否使用Cuda
    Cuda = False
    # 分类个数+1
    num_classes = 3
    #----------------------------------------------------------------------------------------------------------------------------#
    #   是否使用主干网络的预训练权重，此处使用的是主干的权重，因此是在模型构建的时候进行加载的。
    #   如果设置了model_path，则主干的权值无需加载，pretrained的值无意义。
    #   如果不设置model_path，pretrained = True，此时仅加载主干开始训练。
    #   如果不设置model_path，pretrained = False，Freeze_Train = Fasle，此时从0开始训练，且没有冻结主干的过程。
    # --------------------------------------------------------------------------------------------------------------------------
    pretrained = True
    #----------------------------------------------------------------------------------------------------------------------------#
    #   权值文件的下载请看README，可以通过网盘下载。模型的 预训练权重 对不同数据集是通用的，因为特征是通用的。
    #   模型的 预训练权重 比较重要的部分是 主干特征提取网络的权值部分，用于进行特征提取。
    #   预训练权重对于99%的情况都必须要用，不用的话主干部分的权值太过随机，特征提取效果不明显，网络训练的结果也不会好
    #   训练自己的数据集时提示维度不匹配正常，预测的东西都不一样了自然维度不匹配
    #
    #   如果训练过程中存在中断训练的操作，可以将model_path设置成logs文件夹下的权值文件，将已经训练了一部分的权值再次载入。
    #   同时修改下方的 冻结阶段 或者 解冻阶段 的参数，来保证模型epoch的连续性。
    #
    #   当model_path = ''的时候不加载整个模型的权值。
    #
    #   此处使用的是整个模型的权重，因此是在train.py进行加载的，pretrain不影响此处的权值加载。
    #   如果想要让模型从主干的预训练权值开始训练，则设置model_path = ''，pretrain = True，此时仅加载主干。
    #   如果想要让模型从0开始训练，则设置model_path = ''，pretrain = Fasle，Freeze_Train = Fasle，此时从0开始训练，且没有冻结主干的过程。
    #   一般来讲，从0开始训练效果会很差，因为权值太过随机，特征提取效果不明显。
    #
    #   网络一般不从0开始训练，至少会使用主干部分的权值，有些论文提到可以不用预训练，主要原因是他们 数据集较大 且 调参能力优秀。
    #   如果一定要训练网络的主干部分，可以了解imagenet数据集，首先训练分类模型，分类模型的 主干部分 和该模型通用，基于此进行训练。
    #----------------------------------------------------------------------------------------------------------------------------#
    model_path = "model_data/unet_voc.pth"
    # 输入图片的大小
    input_shape = [512, 512]

    Freeze_Train = True
    # 冻结阶段参数
    Init_Epoch = 0
    Freeze_Epoch = 20
    Freeze_batch_size = 4
    Freeze_lr = 1e-4

    # 解冻阶段训练参数
    UnFreeze_Epoch = 40
    Unfreeze_batch_size = 2
    Unfreeze_lr = 1e-5

    # 数据集路径
    VOCdevkit_path = 'VOCdevkit'

    dice_loss = True
    focal_loss = False
    cls_weights = np.ones([num_classes], np.float32)
    num_workers = 4

    model = Unet(num_classes=num_classes, pretrained=pretrained).train()
    if not pretrained:
        weights_init(model)
    if model_path != '':
        print('Load weights {}.'.format(model_path))
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location=device)
        pretrained_dict = {k: v for k, v in pretrained_dict.items(
        ) if np.shape(model_dict[k]) == np.shape(v)}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    model_train = model.train()
    if Cuda:
        model_train = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        model_train = model_train.cuda()

    # 输出loss_history
    loss_history = LossHistory("logs/")

    #  读取数据集对应的txt
    with open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/train.txt"), "r") as f:
        train_lines = f.readlines()

    with open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/val.txt"), "r") as f:
        val_lines = f.readlines()

    # 开始训练
    batch_size = Freeze_batch_size
    lr = Freeze_lr
    start_epoch = Init_Epoch
    end_epoch = Freeze_Epoch

    epoch_step = len(train_lines) // batch_size
    epoch_step_val = len(val_lines) // batch_size

    optimizer = optim.Adam(model_train.parameters(), lr)
    lr_scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=1, gamma=0.96)

    train_dataset = UnetDataset(
        train_lines, input_shape, num_classes, True, VOCdevkit_path)
    val_dataset = UnetDataset(
        val_lines, input_shape, num_classes, False, VOCdevkit_path)
    gen = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                     drop_last=True, collate_fn=unet_dataset_collate)
    gen_val = DataLoader(val_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                         drop_last=True, collate_fn=unet_dataset_collate)

    # 冻结一定部分训练
    if Freeze_Train:
        for param in model.vgg.parameters():
            param.requires_grad = False

    for epoch in range(start_epoch, end_epoch):
        fit_one_epoch(model_train, model, loss_history, optimizer, epoch,
                      epoch_step, epoch_step_val, gen, gen_val, end_epoch, Cuda, dice_loss, focal_loss, cls_weights, num_classes)
        lr_scheduler.step()

    batch_size = Unfreeze_batch_size
    lr = Unfreeze_lr
    start_epoch = Freeze_Epoch
    end_epoch = UnFreeze_Epoch

    epoch_step = len(train_lines) // batch_size
    epoch_step_val = len(val_lines) // batch_size

    optimizer = optim.Adam(model_train.parameters(), lr)
    lr_scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=1, gamma=0.96)

    train_dataset = UnetDataset(
        train_lines, input_shape, num_classes, True, VOCdevkit_path)
    val_dataset = UnetDataset(
        val_lines, input_shape, num_classes, False, VOCdevkit_path)
    gen = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                     drop_last=True, collate_fn=unet_dataset_collate)
    gen_val = DataLoader(val_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                         drop_last=True, collate_fn=unet_dataset_collate)

    if Freeze_Train:
        for param in model.vgg.parameters():
            param.requires_grad = True

    for epoch in range(start_epoch, end_epoch):
        fit_one_epoch(model_train, model, loss_history, optimizer, epoch,
                      epoch_step, epoch_step_val, gen, gen_val, end_epoch, Cuda, dice_loss, focal_loss, cls_weights, num_classes)
        lr_scheduler.step()
