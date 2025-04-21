import torch
import torch.nn as nn


class MultiScaleBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.GELU()
        )
        self.scale2 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=7, padding=3),
            nn.BatchNorm2d(64),
            nn.GELU()
        )
        self.scale3 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(4),
            nn.Conv2d(32, 64, kernel_size=9, padding=4),
            nn.BatchNorm2d(64),
            nn.GELU()
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(64 * 3, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.GELU()
        )

    def forward(self, x):
        s1 = self.scale1(x)
        s2 = self.scale2(x)
        s3 = self.scale3(x)
        # 上采样到s1的尺寸
        s2 = nn.functional.interpolate(s2, size=s1.shape[2:], mode='bilinear', align_corners=True)
        s3 = nn.functional.interpolate(s3, size=s1.shape[2:], mode='bilinear', align_corners=True)
        multi_scale = torch.cat([s1, s2, s3], dim=1)
        return self.fusion(multi_scale)