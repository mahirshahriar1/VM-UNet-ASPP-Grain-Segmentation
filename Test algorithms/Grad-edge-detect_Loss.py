import torch
import torch.nn.functional as F

def compute_gradients(image):
    """
    Compute the gradients of an image using finite differences.
    
    Parameters:
    image (torch.Tensor): Tensor of shape [b, c, h, w] representing the image.
    
    Returns:
    grad_x (torch.Tensor): Gradient of the image along the x direction.
    grad_y (torch.Tensor): Gradient of the image along the y direction.
    """
    grad_x = image[:, :, :, 2:] - image[:, :, :, :-2]
    grad_y = image[:, :, 2:, :] - image[:, :, :-2, :]
    
    return grad_x, grad_y

def gradient_edge_loss(pred, target):
    """
    Compute the gradient edge loss between predicted and ground truth images.
    
    Parameters:
    pred (torch.Tensor): Tensor of shape [b, c, h, w] representing the predicted image.
    target (torch.Tensor): Tensor of shape [b, c, h, w] representing the ground truth image.
    
    Returns:
    loss (torch.Tensor): The mean absolute difference between the gradients of the predicted and target images.
    """
    pred_grad_x, pred_grad_y = compute_gradients(pred)
    target_grad_x, target_grad_y = compute_gradients(target)
    
    diff_x = torch.abs(pred_grad_x - target_grad_x)
    diff_y = torch.abs(pred_grad_y - target_grad_y)
    
    # Need to pad the difference tensors to match the original dimensions
    diff_x = F.pad(diff_x, (1, 1, 0, 0), mode='constant', value=0)
    diff_y = F.pad(diff_y, (0, 0, 1, 1), mode='constant', value=0)
    
    loss = torch.mean(diff_x + diff_y)
    
    return loss

# Example usage:
# pred = torch.randn((b, c, h, w))  # Replace with the predicted image tensor
# target = torch.randn((b, c, h, w))  # Replace with the ground truth image tensor

# loss = gradient_edge_loss(pred, target)
# print("Gradient Edge Loss:", loss.item())
