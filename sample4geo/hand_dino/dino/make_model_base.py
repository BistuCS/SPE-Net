import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from timm.models import create_model
from .backbones.model_dinov2 import dinov2
import numpy as np
from torch.nn import init
from torch.nn.parameter import Parameter

import torch.nn.init as init

class ForegroundBackgroundSelector(nn.Module):
    def __init__(self, in_dim=768):
        super(ForegroundBackgroundSelector, self).__init__()
        
        # 线性分类器，用于预测每个 patch 是前景还是背景（输出一个值，sigmoid 后为概率）
        self.linear = nn.Linear(in_dim, 1)
        
        # Kaiming 初始化权重
        init.kaiming_normal_(self.linear.weight, mode='fan_out', nonlinearity='relu')
        if self.linear.bias is not None:
            init.constant_(self.linear.bias, 0.0)

    def forward(self, patch_tokens):  # patch_tokens: [B, 1369, 768]
        # 1. 计算每个 patch 是前景的概率
        logits = self.linear(patch_tokens)      # [B, 1369, 1]
        probs = torch.sigmoid(logits)           # [B, 1369, 1]  -> 概率接近1表示前景，接近0表示背景

        # 2. 得到前景权重和背景权重
        foreground_weights = probs              # [B, 1369, 1]
        background_weights = 1.0 - probs        # [B, 1369, 1]

        # 3. 对 patch 特征加权求和，得到前景和背景的特征向量
        foreground_feature = (patch_tokens * foreground_weights).sum(dim=1) / (foreground_weights.sum(dim=1) + 1e-6)  # [B, 768]
        background_feature = (patch_tokens * background_weights).sum(dim=1) / (background_weights.sum(dim=1) + 1e-6)  # [B, 768]

        return foreground_feature, background_feature

class FeatureFusion(nn.Module):
    def __init__(self, embed_dim=768, fusion_dim=768):
        super(FeatureFusion, self).__init__()

        # 输入是三份特征拼接后的 [B, 3*768]
        self.fusion_layer = nn.Linear(3 * embed_dim, fusion_dim)

        # 使用 Kaiming 初始化
        nn.init.kaiming_normal_(self.fusion_layer.weight, mode='fan_out', nonlinearity='relu')
        if self.fusion_layer.bias is not None:
            nn.init.constant_(self.fusion_layer.bias, 0.0)

        self.norm = nn.LayerNorm(fusion_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, cls_token_feature, foreground_feature, background_feature):
        # cls_token_feature: from forward_head (e.g., [B, 768])
        # foreground_feature / background_feature: from patch selection (e.g., [B, 768])

        concat = torch.cat([cls_token_feature, foreground_feature, background_feature], dim=1)  # [B, 2304]
        fused = self.fusion_layer(concat)  # [B, 768]
        fused = self.norm(fused)
        fused = self.relu(fused)
        return fused

class build_convnext(nn.Module):
    def __init__(self):
        super(build_convnext, self).__init__()
       
        dino_name = 'dinov2'
        print('using model_type: {} as a backbone'.format(dino_name))
        if 'base' in dino_name:
            self.in_planes = 768
        self.convnext = dinov2(pretrained=True)
        self.selector = ForegroundBackgroundSelector()
        self.fusion = FeatureFusion()

    def forward(self, x):
        # -- backbone feature extractor
        part_feature, gap_feature = self.convnext(x)
        fg, bg = self.selector(part_feature) 
        # fus = self.fusion(gap_feature, fg, bg)
        # print(gap_feature.shape)
        # print(fg.shape)
        # print(bg.shape)
        return gap_feature, fg, bg

def make_convnext_model():
    print('===========building convnext===========')
    model = build_convnext()
    return model