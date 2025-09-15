import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from thop import profile, clever_format

import os, sys, time

# #{ include this project packages

current_dir = os.path.dirname(os.path.abspath(__file__))

project_root = os.path.join(current_dir, '..')

sys.path.append(project_root)

# #}

from .stdcnet import STDCNet1446, STDCNet813
# from modules.bn import InPlaceABNSync as BatchNorm2d

# #{ Conv2dBNReLU

class Conv2dBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, *args, **kwargs):
        super(Conv2dBNReLU, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size = kernel_size,
            stride = stride,
            padding = padding,
            bias = False
        )

        self.bn = nn.BatchNorm2d(out_channels)
        # self.bn = BatchNorm2d(out_channels, activation='none')
        self.relu = nn.ReLU()
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

# #}

# #{ BiSeNetOutput

class BiSeNetOutput(nn.Module):
    def __init__(self, in_channels, mid_chan, num_classes, *args, **kwargs):
        super(BiSeNetOutput, self).__init__()
        self.conv = Conv2dBNReLU(in_channels, mid_chan, kernel_size=3, stride=1, padding=1)
        self.conv_out = nn.Conv2d(mid_chan, num_classes, kernel_size=1, bias=False)
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.conv_out(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params

# #}

# #{ AttentionRefinement

class AttentionRefinement(nn.Module):
    def __init__(self, in_channels, out_channels, *args, **kwargs):
        super(AttentionRefinement, self).__init__()
        self.conv = Conv2dBNReLU(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv_atten = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)
        self.bn_atten = nn.BatchNorm2d(out_channels)
        # self.bn_atten = BatchNorm2d(out_channels, activation='none')

        self.sigmoid_atten = nn.Sigmoid()
        self.init_weight()

    def forward(self, x):
        feat = self.conv(x)
        atten = F.avg_pool2d(feat, feat.size()[2:])
        atten = self.conv_atten(atten)
        atten = self.bn_atten(atten)
        atten = self.sigmoid_atten(atten)
        out = torch.mul(feat, atten)
        return out

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

# #}

# #{ ContextPath

class ContextPath(nn.Module):
    def __init__(self, backbone='CatNetSmall', pretrain_model='', use_conv_last=False, *args, **kwargs):
        super(ContextPath, self).__init__()

        self.backbone_name = backbone
        if backbone == 'STDCNet1446':
            self.backbone = STDCNet1446(pretrain_model=pretrain_model, use_conv_last=use_conv_last)
            self.arm16 = AttentionRefinement(512, 128)
            inplanes = 1024
            if use_conv_last:
                inplanes = 1024
            self.arm32 = AttentionRefinement(inplanes, 128)
            self.conv_head32 = Conv2dBNReLU(128, 128, kernel_sise=3, stride=1, padding=1)
            self.conv_head16 = Conv2dBNReLU(128, 128, kernel_size=3, stride=1, padding=1)
            self.conv_avg = Conv2dBNReLU(inplanes, 128, kernel_size=1, stride=1, padding=0)

        elif backbone == 'STDCNet813':
            self.backbone = STDCNet813(pretrain_model=pretrain_model, use_conv_last=use_conv_last)
            self.arm16 = AttentionRefinement(512, 128)
            inplanes = 1024
            if use_conv_last:
                inplanes = 1024
            self.arm32 = AttentionRefinement(inplanes, 128)
            self.conv_head32 = Conv2dBNReLU(128, 128, kernel_size=3, stride=1, padding=1)
            self.conv_head16 = Conv2dBNReLU(128, 128, kernel_size=3, stride=1, padding=1)
            self.conv_avg = Conv2dBNReLU(inplanes, 128, kernel_size=1, stride=1, padding=0)
        else:
            print('backbone is not in backbone lists')
            exit(0)

        self.init_weight()

    def forward(self, x):
        H0, W0 = x.size()[2:]

        feat2, feat4, feat8, feat16, feat32 = self.backbone(x)
        # print('feat2', feat2.size())
        # print('feat4', feat4.size())
        # print('feat8', feat8.size())
        # print('feat16', feat16.size())
        # print('feat32', feat32.size())

        H8, W8 = feat8.size()[2:]
        H16, W16 = feat16.size()[2:]
        H32, W32 = feat32.size()[2:]

        # print('feat32[2:]', feat32.size()[2:])

        avg = F.avg_pool2d(feat32, feat32.size()[2:])
        # print('avg', avg.size())

        avg = self.conv_avg(avg)
        avg_up = F.interpolate(avg, (H32, W32), mode='nearest')

        feat32_arm = self.arm32(feat32)
        feat32_sum = feat32_arm + avg_up
        feat32_up = F.interpolate(feat32_sum, (H16, W16), mode='nearest')
        feat32_up = self.conv_head32(feat32_up)

        feat16_arm = self.arm16(feat16)
        feat16_sum = feat16_arm + feat32_up
        feat16_up = F.interpolate(feat16_sum, (H8, W8), mode='nearest')
        feat16_up = self.conv_head16(feat16_up)

        return feat2, feat4, feat8, feat16, feat16_up, feat32_up # x8, x16

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params

# #}

# #{ FeatureFusion

class FeatureFusion(nn.Module):
    def __init__(self, in_channels, out_channels, *args, **kwargs):
        super(FeatureFusion, self).__init__()
        self.convblk = Conv2dBNReLU(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv1 = nn.Conv2d(out_channels,
                out_channels // 4,
                kernel_size = 1,
                stride = 1,
                padding = 0,
                bias = False)
        self.conv2 = nn.Conv2d(out_channels // 4,
                out_channels,
                kernel_size = 1,
                stride = 1,
                padding = 0,
                bias = False)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.init_weight()

    def forward(self, fsp, fcp):
        fcat = torch.cat([fsp, fcp], dim=1)
        feat = self.convblk(fcat)
        atten = F.avg_pool2d(feat, feat.size()[2:])
        atten = self.conv1(atten)
        atten = self.relu(atten)
        atten = self.conv2(atten)
        atten = self.sigmoid(atten)
        feat_atten = torch.mul(feat, atten)
        feat_out = feat_atten + feat
        return feat_out

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params

# #}

# #{ BiSeNet

class BiSeNet(nn.Module):
    def __init__(self, backbone, num_classes, pretrain_model='', use_boundary_2=False, use_boundary_4=False, use_boundary_8=False, use_boundary_16=False, use_conv_last=False, heat_map=False, *args, **kwargs):
        super(BiSeNet, self).__init__()

        self.use_boundary_2 = use_boundary_2
        self.use_boundary_4 = use_boundary_4
        self.use_boundary_8 = use_boundary_8
        self.use_boundary_16 = use_boundary_16
        # self.heat_map = heat_map
        self.cp = ContextPath(backbone, pretrain_model, use_conv_last=use_conv_last)

        if backbone == 'STDCNet1446':
            conv_out_inplanes = 128
            sp2_inplanes = 32
            sp4_inplanes = 64
            sp8_inplanes = 256
            sp16_inplanes = 512
            inplane = sp8_inplanes + conv_out_inplanes

        elif backbone == 'STDCNet813':
            conv_out_inplanes = 128
            sp2_inplanes = 32
            sp4_inplanes = 64
            sp8_inplanes = 256
            sp16_inplanes = 512
            inplane = sp8_inplanes + conv_out_inplanes

        else:
            print("backbone is not in backbone lists")
            exit(0)

        self.ffm = FeatureFusion(inplane, 256)
        self.conv_out = BiSeNetOutput(256, 256, num_classes)
        self.conv_out16 = BiSeNetOutput(conv_out_inplanes, 64, num_classes)
        self.conv_out32 = BiSeNetOutput(conv_out_inplanes, 64, num_classes)

        self.conv_out_sp16 = BiSeNetOutput(sp16_inplanes, 64, 1)

        self.conv_out_sp8 = BiSeNetOutput(sp8_inplanes, 64, 1)
        self.conv_out_sp4 = BiSeNetOutput(sp4_inplanes, 64, 1)
        self.conv_out_sp2 = BiSeNetOutput(sp2_inplanes, 64, 1)
        self.init_weight()

    def forward(self, x):
        H, W = x.size()[2:]

        # print('X ', x.size())
        feat_res2, feat_res4, feat_res8, feat_res16, feat_cp8, feat_cp16 = self.cp(x)

        feat_out_sp2 = self.conv_out_sp2(feat_res2)
        # print('feat_out_sp2', feat_out_sp2.size())

        feat_out_sp4 = self.conv_out_sp4(feat_res4)
        # print('feat_out_sp4', feat_out_sp4.size())

        feat_out_sp8 = self.conv_out_sp8(feat_res8)
        # print('feat_out_sp8', feat_out_sp8.size())

        feat_out_sp16 = self.conv_out_sp16(feat_res16)
        # print('feat_out_sp16', feat_out_sp16.size())

        feat_fuse = self.ffm(feat_res8, feat_cp8)
        # print('feat_fuse', feat_fuse.size())

        feat_out = self.conv_out(feat_fuse)
        feat_out16 = self.conv_out16(feat_cp8)
        feat_out32 = self.conv_out32(feat_cp16)

        feat_out = F.interpolate(feat_out, (H, W), mode='bilinear', align_corners=True)
        feat_out16 = F.interpolate(feat_out16, (H, W), mode='bilinear', align_corners=True)
        feat_out32 = F.interpolate(feat_out32, (H, W), mode='bilinear', align_corners=True)

        if self.use_boundary_2 and self.use_boundary_4 and self.use_boundary_8:
            return feat_out, feat_out16, feat_out32, feat_out_sp2, feat_out_sp4, feat_out_sp8

        if (not self.use_boundary_2) and self.use_boundary_4 and self.use_boundary_8:
            return feat_out, feat_out16, feat_out32, feat_out_sp4, feat_out_sp8

        if (not self.use_boundary_2) and (not self.use_boundary_4) and self.use_boundary_8:
            return feat_out, feat_out16, feat_out32, feat_out_sp8

        if (not self.use_boundary_2) and (not self.use_boundary_4) and (not self.use_boundary_8):
            return feat_out, feat_out16, feat_out32

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params = [], [], [], []
        for name, child in self.named_children():
            child_wd_params, child_nowd_params = child.get_params()
            if isinstance(child, (FeatureFusion, BiSeNetOutput)):
                lr_mul_wd_params += child_wd_params
                lr_mul_nowd_params += child_nowd_params
            else:
                wd_params += child_wd_params
                nowd_params += child_nowd_params
        return wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params

# #}

if __name__ == "__main__":

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    bisenet = BiSeNet('STDCNet813', 19)
    weights_path = '../checkpoints/STDC1-Seg/model_maxmIOU75.pth'
    bisenet.load_state_dict(torch.load(weights_path))

    dummy_input = torch.randn(3, 3, 512, 1024)

    # flops, params = profile(bisenet, inputs=(dummy_input, ))
    # flops, params = clever_format([flops, params], '%.3f')

    # print(f'FLOPs: {flops}')
    # print(f'Parameters: {params}')

    bisenet = bisenet.to(device)
    dummy_input = dummy_input.to(device)
    output = bisenet(dummy_input)

    size = dummy_input.size()[-2:]

    logits = F.interpolate(
        output[0],
        size=size,
        mode='bilinear',
        align_corners=True
    )

    probs = torch.softmax(logits, dim=1)
    preds = torch.argmax(probs, dim=1)
    print('Output size:', preds.size())

    start = time.time()
    output = bisenet(dummy_input)
    end = time.time()

    print('Inference time', end - start)
