import torch

def continuous_dice_coefficient(gt, pred):
    """
    Paper: https://arxiv.org/ftp/arxiv/papers/1906/1906.11031.pdf

    Calculates the Continuous Dice Coefficient for ground truth and predicted map tensors.
    
    Args:
    gt (torch.Tensor): Binary ground truth tensor.
    pred (torch.Tensor): Probabilistic map tensor, values between [0, 1].

    Returns:
    float: Continuous Dice Coefficient.
    """
    size_of_intersect = torch.sum(gt * pred)
    size_of_gt = torch.sum(gt)
    size_of_pred = torch.sum(pred)

    if size_of_intersect > 0:
        c = torch.sum(gt * pred) / torch.sum(gt * torch.sign(pred))
    else:
        c = 1.0

    cDC = (2 * size_of_intersect) / (c * size_of_gt + size_of_pred)
    return cDC.item()



# # Example usage with dummy data
# ground_truth = torch.tensor([0, 1, 1, 0], dtype=torch.float32)  # Binary ground truth
# predicted_map = torch.tensor([0.5, 0.8, 0.7, 0.1], dtype=torch.float32)  # Probabilistic map

# continuous_dice = continuous_dice_coefficient(ground_truth, predicted_map)
# print(f"Continuous Dice Coefficient: {continuous_dice}")
