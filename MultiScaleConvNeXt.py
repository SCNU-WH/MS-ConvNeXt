import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchvision.models import convnext_tiny


class MultiScaleConvNeXt(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.base = convnext_tiny(weights='IMAGENET1K_V1')
        in_channels = self.base.classifier[2].in_features

        self.branch = MultiScaleBranch()
        self.branch_residual = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=1),
            nn.BatchNorm2d(64)
        )

        self.attention = ChannelAttention(in_channels=64, ratio=8)  # 通道注意力

        self.fusion = nn.Sequential(
            nn.Conv2d(768 + 64, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.GELU()
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, config.num_classes)
        )

        self.ce_loss = nn.CrossEntropyLoss()
        self.fl_loss = FocalLoss(gamma=2.0)
        self.loss_weight = 0.5
        self.train_acc = Accuracy(task="multiclass", num_classes=config.num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=config.num_classes)
        self.train_recall = Recall(task="multiclass", num_classes=config.num_classes, average='macro')
        self.val_recall = Recall(task="multiclass", num_classes=config.num_classes, average='macro')
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.train_recalls = []
        self.val_recalls = []

    def forward(self, x):
        x_base = self.base.features(x)  # B, 768, 7, 7
        x_branch = self.branch(x)  # B, 64, 224, 224
        residual = self.branch_residual(x)
        x_branch = x_branch + residual

        # 应用通道注意力并降采样
        x_branch = self.attention(x_branch)  # 通道加权
        x_branch = nn.functional.adaptive_avg_pool2d(x_branch, (7, 7))  # 降采样至7x7

        # 特征融合
        fused = torch.cat([x_base, x_branch], dim=1)  # B, 768+64=832, 7, 7
        features = self.fusion(fused)  # B, 512, 7, 7
        logits = self.classifier(features)
        return logits, features