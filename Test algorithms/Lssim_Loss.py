import torch
import torch.nn.functional as F
from torchmetrics.functional import structural_similarity_index_measure as ssim

class ImprovedSSIMLoss(torch.nn.Module):
    def __init__(self):
        super(ImprovedSSIMLoss, self).__init__()
        # Define the Laplacian filter for edge enhancement
        self.laplacian_filter = torch.tensor([[0, 1, 0], 
                                              [1, -5, 1], 
                                              [0, 1, 0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
    def forward(self, y_true, y_pred):
        # Edge enhance the ground truth image
        y_true_edge_enhanced = self.edge_enhance(y_true)
        
        # Compute SSIM between the edge enhanced ground truth and prediction
        ssim_loss = 1 - ssim(y_true_edge_enhanced, y_pred, data_range=1.0)
        
        return ssim_loss
    
    def edge_enhance(self, img):
        # Apply the Laplacian filter to each channel
        batch_size, channels, height, width = img.shape
        laplacian_filter = self.laplacian_filter.to(img.device)
        
        edge_enhanced_img = F.conv2d(img.view(batch_size * channels, 1, height, width), laplacian_filter, padding=1)
        edge_enhanced_img = edge_enhanced_img.view(batch_size, channels, height, width)
        
        # Add the edge enhanced image to the original image
        enhanced_img = img + edge_enhanced_img
        
        return enhanced_img

# # Example usage
# batch_size, channels, height, width = 1, 3, 256, 256
# y_true = torch.randn(batch_size, channels, height, width, device='cuda')
# y_pred = torch.randn(batch_size, channels, height, width, device='cuda')

# ssim_loss = ImprovedSSIMLoss()
# loss = ssim_loss(y_true, y_pred)
# print(f"Improved SSIM Loss: {loss.item()}")
