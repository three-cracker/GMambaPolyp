from lib.pvtv2 import pvt_v2_b2
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from torch.nn import Conv2d
import warnings
import torch
from torch import nn
from lib.vmamba.vmuet import SS2D


class GSA(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(GSA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.conv = nn.Conv2d(2, 1, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        x = self.sigmoid(avg_out + max_out)

        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)

        return self.sigmoid(out)

class GSDC(nn.Module):
    def __init__(self, in_planes):
        super(GSDC, self).__init__()
        self.gsa = GSA(in_planes)

    def forward(self, x, y, sub):
        _fu = self.gsa(torch.add(x, y))
        _x = _fu * x
        _y = _fu * y
        out = torch.add(torch.add(_x, _y), sub)
        
        return out

class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            # nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            # nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        size = x.shape[-2:]
        x = super(ASPPPooling, self).forward(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

class GCSS2D(nn.Module):
    def __init__(self, in_ch, d_conv=3, d_state=16):
        super().__init__()
        self.ss2d = SS2D(in_ch, d_conv=d_conv, d_state=d_state)
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.project = nn.Sequential(
            nn.Conv2d(in_ch * 2, in_ch, 1, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        ss_out = self.ss2d(x)
        
        pool_max = self.avgpool(x)
        pool_max = F.interpolate(pool_max, size=ss_out.size()[2:], mode='bilinear', align_corners=False)

        out = self.project(torch.cat([ss_out, pool_max], dim=1))

        return out

class RCSA(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid()
        )
        self.spatial_att = nn.Sequential(
            nn.Conv2d(in_channels, 1, 7, padding=3, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        channel_att = self.channel_att(x)
        spatial_att = self.spatial_att(x)
        att = channel_att * spatial_att
        return x * att + x

class MSSSM(nn.Module):
    def __init__(self, in_channels, branch_ratio=0.25):
        super().__init__()
        gc = int(in_channels * branch_ratio)
        self.split_indexes = (in_channels-3*gc, gc, gc, gc)
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels - 3 * gc, in_channels - 3 * gc, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels - 3 * gc),
            nn.ReLU(inplace=True)
        )
        
        self.conv2 = GCSS2D(gc, d_conv=3, d_state=8)
        
        self.conv3 = ASPPConv(gc, gc, 12)
        
        self.conv4 = GCSS2D(gc, d_conv=7, d_state=24)
        
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels, 1, bias=False),
            nn.ReLU(inplace=True)
        )
        
        self.project = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )
        self.csa = RCSA(in_channels)

    def forward(self, x):
        x_1, x_2, x_3, x_4 = torch.split(x, self.split_indexes, dim=1)
        
        # 处理各分支
        out1 = self.conv1(x_1)
        out2 = self.conv2(x_2)
        x_3 = x_3 + x_1
        out3 = self.conv3(x_3)
        x_4 = x_4 + x_2
        out4 = self.conv4(x_4)

        fused = torch.cat([out1, out2, out3, out4], dim=1)
        
        global_feat = self.global_pool(x)
        global_feat = F.interpolate(global_feat, size=fused.shape[2:], mode='bilinear', align_corners=False)
        
        y = torch.cat([fused, global_feat], dim=1)

        y = x + self.project(y)

        y = self.csa(y)
        return y

class conv(nn.Module):
    """
    Linear Embedding
    """

    def __init__(self, input_dim=512, embed_dim=768, k_s=3):
        super().__init__()

        self.proj = nn.Sequential(nn.Conv2d(input_dim, embed_dim, 3, padding=1, bias=False), nn.ReLU(),
                                  nn.Conv2d(embed_dim, embed_dim, 3, padding=1, bias=False), nn.ReLU())

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    return F.interpolate(input, size, scale_factor, mode, align_corners)


class GMambaPolyp(nn.Module):
    def __init__(self, dim=32, dims=[64, 128, 320, 512]):
        super(GMambaPolyp, self).__init__()

        self.backbone = pvt_v2_b2()  # [64, 128, 320, 512]
        path = '/home/featurize/work/.ssh/PGCF/lib/pvt_v2_b2.pth'
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = dims[0], dims[1], dims[2], dims[3]

        self.MSSSM_c4 = MSSSM(c4_in_channels)
        self.MSSSM_c3 = MSSSM(c3_in_channels)
        self.MSSSM_c2 = MSSSM(c2_in_channels)
        self.MSSSM_c1 = MSSSM(c1_in_channels)
        

        self.linear_c4 = conv(input_dim=c4_in_channels, embed_dim=dim)
        self.linear_c3 = conv(input_dim=c3_in_channels, embed_dim=dim)
        self.linear_c2 = conv(input_dim=c2_in_channels, embed_dim=dim)
        self.linear_c1 = conv(input_dim=c1_in_channels, embed_dim=dim)
        self.linear_fuse = ConvModule(in_channels=dim * 4, out_channels=dim, kernel_size=1,
                                      norm_cfg=dict(type='BN', requires_grad=True))
        self.linear_fuse34 = ConvModule(in_channels=dim, out_channels=dim, kernel_size=1,
                                        norm_cfg=dict(type='BN', requires_grad=True))
        self.linear_fuse2 = ConvModule(in_channels=dim, out_channels=dim, kernel_size=1,
                                       norm_cfg=dict(type='BN', requires_grad=True))
        self.linear_fuse1 = ConvModule(in_channels=dim, out_channels=dim, kernel_size=1,
                                       norm_cfg=dict(type='BN', requires_grad=True))

        self.linear_pred = Conv2d(dim, 1, kernel_size=1)
        self.dropout = nn.Dropout(0.1)
        self.linear_pred1 = Conv2d(dim, 1, kernel_size=1)
        self.dropout1 = nn.Dropout(0.1)
        self.linear_pred2 = Conv2d(dim, 1, kernel_size=1)
        self.dropout2 = nn.Dropout(0.1)
        self.linear_pred_f = Conv2d(3, 1, kernel_size=1)

        self.gsdc_3 = GSDC(dim)
        self.gsdc_2 = GSDC(dim)
        self.gsdc_1 = GSDC(dim)



    def forward(self, x):
        # backbone
        pvt = self.backbone(x)
        c1, c2, c3, c4 = pvt
        n, _, h, w = c4.shape

        _c4 = self.MSSSM_c4(c4)  # [1, 64, 11, 11]
        _c3 = self.MSSSM_c3(c3)  # [1, 64, 22, 22]
        _c2 = self.MSSSM_c2(c2)  # [1, 64, 44, 44]
        _c1 = self.MSSSM_c1(c1)

        _c4 = self.linear_c4(_c4).permute(0, 2, 1).reshape(n, -1, _c4.shape[2], _c4.shape[3])
        _c3 = self.linear_c3(_c3).permute(0, 2, 1).reshape(n, -1, _c3.shape[2], _c3.shape[3])
        _c2 = self.linear_c2(_c2).permute(0, 2, 1).reshape(n, -1, _c2.shape[2], _c2.shape[3])
        _c1 = self.linear_c1(_c1).permute(0, 2, 1).reshape(n, -1, _c1.shape[2], _c1.shape[3])

        _c4 = resize(_c4, size=_c3.size()[2:], mode='bilinear', align_corners=False)
        sub3 = abs(_c3 - _c4)
        L34 = self.linear_fuse34(self.gsdc_3(_c3, _c4, sub3))
        O34 = L34

        _c3 = resize(_c3, size=_c2.size()[2:], mode='bilinear', align_corners=False)
        sub2 = abs(_c2 - _c3)
        L34 = resize(L34, size=_c2.size()[2:], mode='bilinear', align_corners=False)
        L2 = self.linear_fuse34(self.gsdc_2(_c2, L34, sub2))
        O2 = L2

        _c2 = resize(_c2, size=_c1.size()[2:], mode='bilinear', align_corners=False)
        sub1 = abs(_c1 - _c2)
        L2 = resize(L2, size=_c1.size()[2:], mode='bilinear', align_corners=False)
        _c = self.linear_fuse34(self.gsdc_1(_c1, L2, sub1))

        x = self.dropout(_c)
        x = self.linear_pred(x)
        O2 = self.dropout2(O2)
        O2 = self.linear_pred2(O2)
        O34 = self.dropout1(O34)
        O34 = self.linear_pred1(O34)
        return x, O2, O34
