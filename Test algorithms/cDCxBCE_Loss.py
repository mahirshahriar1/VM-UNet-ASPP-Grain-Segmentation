import torch
import torch.nn.functional as F

def continuous_dice_loss(output, target):
    """
    Calculates the Continuous Dice Loss for a set of prediction and target.

    Args:
    output (torch.Tensor): Model's probabilistic map output, values between [0, 1].
    target (torch.Tensor): Ground truth binary tensor, values {0, 1}.

    Returns:
    torch.Tensor: Continuous Dice Loss.
    """
    # Flatten the tensors to calculate the loss on a per-image basis
    output_flat = output.view(output.size(0), -1)
    target_flat = target.view(target.size(0), -1)

    # Calculate intersection and cardinality (or area)
    intersection = torch.sum(output_flat * target_flat, dim=1)
    cardinality_output = torch.sum(output_flat, dim=1)
    cardinality_target = torch.sum(target_flat, dim=1)

    # Calculate c as per the definition
    c_numerator = torch.sum(output_flat * target_flat, dim=1)
    c_denominator = torch.sum(output_flat * torch.sign(target_flat), dim=1)

    # Avoid division by zero
    c_denominator = torch.where(c_denominator == 0, torch.ones_like(c_denominator), c_denominator)
    c = c_numerator / c_denominator

    # Compute continuous Dice coefficient
    cDC = (2. * intersection) / (c * cardinality_target + cardinality_output)

    # Compute loss as 1 - cDC
    cDL = 1 - cDC

    return cDL.mean()

##  Better to use dice_weight > ce_weight , {0.8,0.2}/{0.9,0.1}
def combined_dice_ce_loss(output, target, dice_weight=0.5, ce_weight=0.5):
    ce_loss = F.binary_cross_entropy(output, target)
    dice_loss = continuous_dice_loss(output, target)
    combined_loss = dice_weight * dice_loss + ce_weight * ce_loss
    return combined_loss
