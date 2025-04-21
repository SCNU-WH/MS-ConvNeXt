import torch.nn as nn


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=1.0):
        super().__init__()
        self.gamma = gamma  # 难样本聚焦参数
        self.alpha = alpha  # 全局平衡系数

    def forward(self, logits, targets):
        prob = nn.functional.softmax(logits, dim=1).gather(1, targets.unsqueeze(1)).squeeze()
        ce_loss = nn.functional.cross_entropy(logits, targets, reduction='none')
        fl_loss = self.alpha * (1 - prob) ** self.gamma * ce_loss
        return fl_loss.mean()  # 平均损失

