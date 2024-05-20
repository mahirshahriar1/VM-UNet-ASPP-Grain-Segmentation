import torch
import torch.nn.functional as F

def sobel_filters():
    """
    Define the Sobel filters for x and y directions.
    
    Returns:
    sobel_x (torch.Tensor): Sobel filter for the x direction.
    sobel_y (torch.Tensor): Sobel filter for the y direction.
    """
    sobel_x = torch.tensor([[1, 0, -1],
                            [2, 0, -2],
                            [1, 0, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    
    sobel_y = torch.tensor([[-1, -2, -1],
                            [ 0,  0,  0],
                            [ 1,  2,  1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    
    return sobel_x, sobel_y

def compute_sobel_gradients(image, sobel_x, sobel_y):
    """
    Compute the Sobel gradients of an image.
    
    Parameters:
    image (torch.Tensor): Tensor of shape [b, c, h, w] representing the image.
    sobel_x (torch.Tensor): Sobel filter for the x direction.
    sobel_y (torch.Tensor): Sobel filter for the y direction.
    
    Returns:
    grad_x (torch.Tensor): Gradient of the image along the x direction.
    grad_y (torch.Tensor): Gradient of the image along the y direction.
    """
    grad_x = F.conv2d(image, sobel_x, padding=1, groups=image.shape[1])
    grad_y = F.conv2d(image, sobel_y, padding=1, groups=image.shape[1])
    
    return grad_x, grad_y

def sobel_edge_loss(pred, target):
    """
    Compute the Sobel edge loss between predicted and ground truth images.
    
    Parameters:
    pred (torch.Tensor): Tensor of shape [b, c, h, w] representing the predicted image.
    target (torch.Tensor): Tensor of shape [b, c, h, w] representing the ground truth image.
    
    Returns:
    loss (torch.Tensor): The mean absolute difference between the Sobel gradients of the predicted and target images.
    """
    sobel_x, sobel_y = sobel_filters()
    sobel_x, sobel_y = sobel_x.to(pred.device), sobel_y.to(pred.device)
    
    pred_grad_x, pred_grad_y = compute_sobel_gradients(pred, sobel_x, sobel_y)
    target_grad_x, target_grad_y = compute_sobel_gradients(target, sobel_x, sobel_y)
    
    diff_x = torch.abs(pred_grad_x - target_grad_x)
    diff_y = torch.abs(pred_grad_y - target_grad_y)
    
    loss = torch.mean(diff_x + diff_y)
    
    return loss

# Example usage:
# pred = torch.randn((b, c, h, w), device=device)  # Replace with the predicted image tensor
# target = torch.randn((b, c, h, w), device=device)  # Replace with the ground truth image tensor

# loss = sobel_edge_loss(pred, target)
# print("Sobel Edge Loss:", loss.item())
