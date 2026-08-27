import torch
from torch import nn
import torch.nn.functional as F


class KLDivergenceLoss(nn.Module):
    """
    基于 soft label 的 KL 散度损失模块。
    """
    def __init__(self,lambda_weight = 2):
        """
        lambda_weight: Control the weight decay of index positions, the greater the lambda_weight, the more severe the weight decay
        """
        super(KLDivergenceLoss,self).__init__()
        self.lambda_weight = lambda_weight

    def forward(self, logits, targets, target_ratings):
        """
        计算 KL 散度损失。
        :param logits: 模型输出的 logits，形状为 (batch_size, num_classes)
        :param targets:  future k interacted items  (batch_size, k)
        :param target_ratings: ratings of future k interactions (batch_size, k)
        :return: KL 散度损失标量
        """
        batch_size, num_classes = logits.shape
        device = logits.device

        # positive indicator
        positive_index = torch.zeros((batch_size, num_classes), device=logits.device)   # (batch_size, num_class)
        positive_index.scatter_(1, targets, 1)  # 仅对有效类别填充 1
        positive_index[:,0] = 0

        mask = targets > 0  # 0 是无效类别，忽略

        rating_weights = target_ratings.float()

        # 计算索引权重（位置靠后的类别权重更小）
        position_weights = torch.exp(-self.lambda_weight * torch.arange(targets.shape[1], device=device))
        class_weights = mask.float() * position_weights * rating_weights # 仅对有效类别赋权重
        class_weights = class_weights / class_weights.sum(1).unsqueeze(1)
        soft_probs = torch.zeros_like(positive_index)
        soft_probs.scatter_add_(
            dim=1,
            index=targets,
            src=class_weights
        )

        log_probs = F.log_softmax(logits, dim=-1)  # 计算 log_softmax
        loss = F.kl_div(log_probs, soft_probs, reduction='batchmean')  # 计算 KL 散度
        return loss


# Example usage
if __name__ == "__main__":
    batch_size = 2
    num_classes = 50

    # Simulated model output logits
    pred = torch.randn(batch_size, num_classes, requires_grad=True)

    # Generate soft labels (assumed to be pre-processed)
    targets = torch.tensor([[1,25,14,0,0], [2,4,1,10,30]],dtype = torch.long)
    # targets = torch.tensor([[1], [2]],dtype = torch.long)

    target_ratings =  torch.tensor([[5,5,4,0,0], [5,4,1,3,5]],dtype = torch.long)

    # Initialize loss function
    loss_fn = KLDivergenceLoss(lambda_weight = 2)

    # Compute loss
    loss = loss_fn(pred, targets,target_ratings)
    print("Soft Label KL Loss:", loss.item())

