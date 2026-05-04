import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from timm.models import create_model
from .backbones.model_dinov2 import dinov2
import numpy as np
from torch.nn import init
from torch.nn.parameter import Parameter
from sample4geo.Utils import init
# from lpn import get_part_pool
import math
class Gem_heat(nn.Module):
    def __init__(self, dim=768, p=3, eps=1e-6):
        super(Gem_heat, self).__init__()
        self.p = nn.Parameter(torch.ones(dim) * p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3):
        p = F.softmax(p).unsqueeze(-1)
        x = torch.matmul(x, p)
        x = x.view(x.size(0), x.size(1))
        return x


def position(H, W, is_cuda=True):
    if is_cuda:
        loc_w = torch.linspace(-1.0, 1.0, W).cuda().unsqueeze(0).repeat(H, 1)
        loc_h = torch.linspace(-1.0, 1.0, H).cuda().unsqueeze(1).repeat(1, W)
    else:
        loc_w = torch.linspace(-1.0, 1.0, W).unsqueeze(0).repeat(H, 1)
        loc_h = torch.linspace(-1.0, 1.0, H).unsqueeze(1).repeat(1, W)
    loc = torch.cat([loc_w.unsqueeze(0), loc_h.unsqueeze(0)], 0).unsqueeze(0)
    return loc


def stride(x, stride):
    b, c, h, w = x.shape
    return x[:, :, ::stride, ::stride]


def init_rate_half(tensor):
    if tensor is not None:
        tensor.data.fill_(0.5)


def init_rate_0(tensor):
    if tensor is not None:
        tensor.data.fill_(0.)


# class BasicConv(nn.Module):
#     def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
#                  bn=True, bias=False):
#         super(BasicConv, self).__init__()
#         self.out_channels = out_planes
#         self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
#                               dilation=dilation, groups=groups, bias=bias)
#         self.conv2 =KAN_Convolutional_Layer(n_convs = out_planes, kernel_size=(kernel_size,kernel_size), stride=(stride,stride), padding=(padding,padding),
#                               dilation=(dilation,dilation),device='cuda')
#         self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
#         self.relu = nn.ReLU() if relu else None

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.conv2(x)
#         if self.bn is not None:
#             x = self.bn(x)
#         if self.relu is not None:
#             x = self.relu(x)
#         return x

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=False,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.SiLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class ZPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class AttentionGate(nn.Module):
    def __init__(self):
        super(AttentionGate, self).__init__()
        kernel_size = 7
        self.compress = ZPool()
        self.conv = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.conv(x_compress)
        scale = torch.sigmoid_(x_out)
        return x * scale


class TripletAttention(nn.Module):
    def __init__(self):
        super(TripletAttention, self).__init__()
        self.cw = AttentionGate()
        self.hc = AttentionGate()

    def forward(self, x):
        x_perm1 = x.permute(0, 2, 1, 3).contiguous()
        x_out1 = self.cw(x_perm1)
        x_out11 = x_out1.permute(0, 2, 1, 3).contiguous()
        x_perm2 = x.permute(0, 3, 2, 1).contiguous()
        x_out2 = self.hc(x_perm2)
        x_out21 = x_out2.permute(0, 3, 2, 1).contiguous()
        return x_out11, + x_out21


class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, droprate, relu=False, bnorm=True, num_bottleneck=512, linear=True,
                 return_f=False):
        super(ClassBlock, self).__init__()
        self.return_f = return_f
        add_block = []

        if linear:
            add_block += [nn.Linear(input_dim, num_bottleneck)]
        else:
            num_bottleneck = input_dim
        if bnorm:
            add_block += [nn.BatchNorm1d(num_bottleneck)]
        if True:
            add_block += [nn.SiLU(inplace=True)]
        if droprate > 0:
            add_block += [nn.Dropout(p=droprate)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)
        # classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier

    def forward(self, x):
        x = self.add_block(x)
        if self.training:
            if self.return_f:
                f = x
                x = self.classifier(x)
                return x, f
            else:
                x = self.classifier(x)
                return x
        else:
            return x


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, std=0.001)
        nn.init.constant_(m.bias.data, 0.0)


class MLP1D(nn.Module):
    """
    The non-linear neck in byol: fc-bn-relu-fc
    """
    def __init__(self, in_channels, hid_channels, out_channels,
                 norm_layer=None, bias=False, num_mlp=2):
        super(MLP1D, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        mlps = []
        for _ in range(num_mlp-1):
            mlps.append(nn.Conv1d(in_channels, hid_channels, 1, bias=bias))
            mlps.append(norm_layer(hid_channels))
            mlps.append(nn.SiLU(inplace=True))
            in_channels = hid_channels
        mlps.append(nn.Conv1d(hid_channels, out_channels, 1, bias=bias))
        self.mlp = nn.Sequential(*mlps)

    def init_weights(self, init_linear='kaiming'): # origin is 'normal'
        init.init_weights(self, init_linear)

    def forward(self, x):
        x = self.mlp(x)
        return x


class build_convnext(nn.Module):
    def __init__(self, num_classes, block=4, return_f=False, resnet=False):
        super(build_convnext, self).__init__()
        self.return_f = return_f
        resnet=False
        super(build_convnext, self).__init__()
       
        dino_name = 'dinov2'
        print('using model_type: {} as a backbone'.format(dino_name))

        self.in_planes = 768
        self.convnext = dinov2(pretrained=True)

        self.num_classes = num_classes
        self.classifier1 = ClassBlock(self.in_planes, num_classes, 0.5, return_f=return_f)
        self.block = block
        self.tri_layer = TripletAttention()
        for i in range(self.block):
            name = 'classifier_mcb' + str(i + 1)
            setattr(self, name, ClassBlock(self.in_planes, num_classes, 0.5, return_f=self.return_f))

        # define for Domain Space Alignment Module
        in_channels = 768
        hid_channels = 2048
        out_channels = 256
        norm_layer = None
        num_layers = 2
        self.proj = MLP1D(in_channels, hid_channels, out_channels, norm_layer, num_mlp=num_layers)
        self.proj.init_weights()
        self.proj_obj = MLP1D(in_channels, hid_channels, out_channels, norm_layer, num_mlp=num_layers)
        self.proj_obj.init_weights()
        self.scale = 1.
        self.l2_norm = True
        self.num_heads = 8
        # self.KANConv = KAN_Convolutional_Layer(n_convs = 1, kernel_size=(3,3),stride = (1,1),padding = (1,1))
        # self.KANLinear = KANLinear(1024, 701)
        self.GateLinear = nn.Linear(in_channels, in_channels + 6 + 1)
        # self.weight_H = nn.Parameter(torch.tensor(0))
        # self.weight_W = nn.Parameter(torch.tensor(0.05))z

    def forward(self, x):
        # -- backbone feature extractor
        part_features, gap_feature = self.convnext(x)

        b = part_features.size(0)
        part_features = part_features[:, 1:, :]  # 裁剪到1369（37x37）
        part_features = part_features.reshape(b, 768, 37, 37)  # [16, 768, 37, 37]

        # -- Training
        if self.training:

            # 1. Domain Space Alignment Module
            b, c, h, w = part_features.shape #torch.Size([4, 1024, 12, 12])
            lpn_lst = self.get_part_pool(part_features)
            output = self.GateLinear(part_features.permute(0, 2, 3, 1)).permute(0, 3, 1, 2) #torch.Size([4, 1030, 12, 12])
            
            z, gating = torch.split(output, (768, 6 + 1), 1)
            pfeat_align = 0
            for i in range(len(lpn_lst)):
                # print(lpn_lst[i].size)
                # print("Gating min:", gating.min().item(), "max:", gating.max().item(), "mean:", gating.mean().item())

                # print("lpn_lst[i] min:", lpn_lst[i].min().item(), "max:", lpn_lst[i].max().item(), "mean:", lpn_lst[i].mean().item())
                # print("part_features min:", part_features.min().item(), "max:", part_features.max().item())
                lpn_lst[i] = F.normalize(lpn_lst[i], dim = 1)
                lpn_lst[i] = ((lpn_lst[i] * gap_feature)).unsqueeze(-1).unsqueeze(-1) #TransFG
                lpn_lst[i] = lpn_lst[i] * F.normalize(part_features * 1.05,dim = 1)
                pfeat_align = pfeat_align + lpn_lst[i] * gating[:, i : i + 1]
            W1, W2 = self.tri_layer(pfeat_align)
            pfeat_align = (1.05 * W1 + W2) / 2
            pfeat = pfeat_align.flatten(2) #torch.Size([b, 1024, 144])
              # (bs, c, h*w)
            W = self.proj(pfeat)  # transfer 1024 to 256
          
            # print("w.shape is {0}".format(W.shape))            
            # W = F.normalize(W, dim=1) if self.l2_norm else W
            W *= (1/self.scale)
            W = F.softmax(W, dim=2) #torch.Size([4, 1024, 144])
            # print("w.shape is {0}".format(W.shape))
            # pfeat_align = pfeat + W
            pfeat_align = torch.cat([pfeat, W], dim=1)
            
            # pfeat_align = pfeat

            # 2. triplet attention
            # W1, W2 = self.tri_layer(part_features) #torch.Size([4, 1024, 12, 12])
            pfeat = pfeat.reshape(b, 768, 37, 37)
            tri_features = [pfeat, W1, W2]  #(bs, 1024, 12, 12)
            convnext_feature = self.classifier1(gap_feature)  # class: (bs, 701); feature: (bs, 512)
            tri_list = []
            self.block = min(self.block, len(tri_features)) 
            for i in range(self.block):
                tri_list.append(tri_features[i].mean([-2, -1]))  # average pooling, 一张图变一个像素
            triatten_features = torch.stack(tri_list, dim=2)  # 把另外两个轴旋转的特征体
            if self.block == 0:
                y = []
            else:
                y = self.part_classifier(self.block, triatten_features,
                                         cls_name='classifier_mcb')  # 把另外两个轴旋转的feature也做分类
    
            ### lpn
            z = []
            triatten_features_lpn = []

            # import threading

            # # 获取当前线程
            # current_thread = threading.current_thread()

            ###
            # lpn_feature = get_part_pool(part_features)
            # for i in lpn_feature:
            #     #             # 打印当前线程的信息
            #     # print(f"Current Thread Name: {current_thread.name}")
            #     z.append(self.KANLinear(i))
            #     z = z + [convnext_feature[0]]
            ###
            # 
            y = y + [convnext_feature]  # 三个分支连起来


            if True:  # return_f是triplet loss的设置，0.3
                cls, features = [], []
                for i in y:
                    cls.append(i[0])
                    features.append(i[1])
                # for i in z:
                #     # print(i.shape)
                #     cls.append(i)
                return pfeat_align, cls, features, gap_feature, part_features

        # -- Eval
        else:
            # ffeature = convnext_feature.view(convnext_feature.size(0), -1, 1)
            # y = torch.cat([y, ffeature], dim=2)
            pass

        return gap_feature, part_features
    


    def get_part_pool(self,x, pool='avg'):
        """
        参数:
            x: 输入张量 [batch, channels, height, width]
            pool: 池化类型 ('avg' 或 'max')
        
        返回:
            List[torch.Tensor]: 按 [12, 11_diff(11-10), 10_diff(10-7), 7] 顺序输出
        """
        result = []
        pooling = nn.AdaptiveAvgPool2d((1,1)) if pool == 'avg' else nn.AdaptiveMaxPool2d((1,1))
        _, _, H, W = x.shape
        c_h, c_w = H // 2, W // 2  # 中心坐标

        # 尺寸层级定义 [outer, inner]
        size_pairs = [
            (12, 11),  # 12x12 → 11x11 (仅用于获取12_full)
            (11, 10),  # 11x11 → 10x10
            (10, 7)    # 10x10 → 7x7
        ]

        # 1. 首先处理完整的12x12
        x_12 = x[:, :, c_h-6:c_h+6, c_w-6:c_w+6]
        if x_12.shape[2:] != (12, 12):
            x_12 = F.interpolate(x_12, size=(12,12), mode='bilinear')
        result.append(torch.squeeze(pooling(x_12)))

        # 2. 处理差值特征
        for i, (outer, inner) in enumerate(size_pairs[1:]):  # 跳过12→11
            # 提取外层区域
            outer_half = outer // 2
            x_outer = x[:, :,
                    max(0, c_h-outer_half):min(H, c_h+outer_half + (outer % 2)),
                    max(0, c_w-outer_half):min(W, c_w+outer_half + (outer % 2))]
            if x_outer.shape[2:] != (outer, outer):
                x_outer = F.interpolate(x_outer, size=(outer,outer), mode='bilinear')

            # 提取内层区域
            inner_half = inner // 2
            x_inner = x[:, :,
                    max(0, c_h-inner_half):min(H, c_h+inner_half + (inner % 2)),
                    max(0, c_w-inner_half):min(W, c_w+inner_half + (inner % 2))]
            if x_inner.shape[2:] != (inner, inner):
                x_inner = F.interpolate(x_inner, size=(inner,inner), mode='bilinear')

            # 零填充内层到外层尺寸
            pad = (outer - inner) // 2
            x_pad = F.pad(x_inner, (pad, pad, pad, pad), "constant", 0)
            if x_pad.shape[2:] != (outer, outer):
                x_pad = F.interpolate(x_pad, size=(outer,outer), mode='bilinear')

            # 计算差值特征
            x_diff = x_outer - x_pad
            result.append(torch.squeeze(pooling(x_diff)))

            # 最后一个层级添加完整7x7
            if i == len(size_pairs[1:])-1:
                x_7 = x_inner  # 直接使用已提取的7x7
                result.append(torch.squeeze(pooling(x_7)))

        return result


    def part_classifier(self, block, x, cls_name='classifier_mcb'):
        part = {}
        predict = {}
        for i in range(block):
            part[i] = x[:, :, i].view(x.size(0), -1)
            name = cls_name + str(i + 1)
            c = getattr(self, name)
            predict[i] = c(part[i])
        y = []
        for i in range(block):
            y.append(predict[i])
        if not self.training:
            return torch.stack(y, dim=2)
        return y

    def fine_grained_transform(self):

        pass


def make_convnext_model(num_class, block=4, return_f=False, resnet=False):
    print('===========building convnext===========')
    model = build_convnext(num_class, block=block, return_f=return_f, resnet=resnet)
    return model
