import torch
import torch.nn as nn

from nets.vgg import VGG16


class unetUp(nn.Module):
    def __init__(self, in_size, out_size):
        super(unetUp, self).__init__()
        self.conv1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs1, inputs2):
        outputs = torch.cat([inputs1, self.up(inputs2)], 1)
        outputs = self.conv1(outputs)
        outputs = self.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = self.relu(outputs)
        return outputs


class Unet(nn.Module):
    def __init__(self, num_classes=21, in_channels=3, pretrained=False):
        super(Unet, self).__init__()
        self.vgg = VGG16(pretrained=pretrained, in_channels=in_channels)
        in_filters = [192, 384, 768, 1024]
        out_filters = [64, 128, 256, 512]
        # upsampling
        # 64,64,512
        self.up_concat4 = unetUp(in_filters[3], out_filters[3])
        # 128,128,256
        self.up_concat3 = unetUp(in_filters[2], out_filters[2])
        # 256,256,128
        self.up_concat2 = unetUp(in_filters[1], out_filters[1])
        # 512,512,64
        self.up_concat1 = unetUp(in_filters[0], out_filters[0])

        self.final = nn.Conv2d(out_filters[0], num_classes, 1)

    def forward(self, inputs):
        feat1 = self.vgg.features[:4](inputs)  # 输出：(512, 512, 64)
        feat2 = self.vgg.features[4:9](feat1)  # 输出：(256, 256, 128)
        feat3 = self.vgg.features[9:16](feat2)  # 输出：(128, 128, 256)
        feat4 = self.vgg.features[16:23](feat3)  # 输出：(64, 64, 512)
        feat5 = self.vgg.features[23:-1](feat4)  # 输出：(32, 32, 512)

        """特征层进行上采样后和上一特征层进行堆叠，再通过卷积提取特征，调整通道数"""
        # cat((64, 64, 512), upsample(32, 32, 512)) -> conv(64, 64, 1024) -> (64, 64, 512)
        up4 = self.up_concat4(feat4, feat5)

        # cat((128, 128, 256), upsample(64, 64, 512)) -> conv(128, 128, 768) -> (128, 128, 256)
        up3 = self.up_concat3(feat3, up4)

        # cat((256, 256, 128), upsample(128, 128, 256)) -> conv(256, 256, 384) -> (256, 256, 128)
        up2 = self.up_concat2(feat2, up3)

        # cat((512, 512, 64), upsample(256, 256, 128)) -> conv(512, 512, 192) -> (512, 512, 64)
        up1 = self.up_concat1(feat1, up2)

        # 通过1×1卷积层下降通道数将(512, 512, 64) -> (512, 512, num_classes)表示对输入图片的每一个像素进行分类
        final = self.final(up1)

        return final

    def _initialize_weights(self, *stages):
        for modules in stages:
            for module in modules.modules():
                if isinstance(module, nn.Conv2d):
                    nn.init.kaiming_normal_(module.weight)
                    if module.bias is not None:
                        module.bias.data.zero_()
                elif isinstance(module, nn.BatchNorm2d):
                    module.weight.data.fill_(1)
                    module.bias.data.zero_()
