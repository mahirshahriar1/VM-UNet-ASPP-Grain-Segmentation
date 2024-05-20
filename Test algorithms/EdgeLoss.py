import torch
import torch.nn as nn

class TotalLoss(nn.Module):
    """
    The weighting factor 𝜆1 is selected based on trial and error method.
    A value of 0.1 for 𝜆1 is found to be the most appropriate when MAE
    loss is active and a value of 1 for 𝜆1 when BerHu loss is active. The
    values for 𝜆2 and 𝜆3 were kept as 1 as this was found to be optimum.
    """
    def __init__(self, pix_loss_fn=nn.L1Loss(), ssim_loss_fn=None, edge_loss_fn=None):
        super(TotalLoss, self).__init__()

        # Default loss functions if none are provided
        self.Loss1 = pix_loss_fn if pix_loss_fn else nn.L1Loss()
        self.Loss2 = ssim_loss_fn if ssim_loss_fn else ssim
        self.Loss3 = edge_loss_fn if edge_loss_fn else nn.MSELoss()

        # Set lambda1 based on the type of pixel loss function
        if isinstance(self.Loss1, nn.L1Loss):
            self.lambda1 = 0.1
        elif isinstance(self.Loss1, BerHuLoss):
            self.lambda1 = 1.0
        else:
            self.lambda1 = 1.0

        self.lambda2 = 1.0
        self.lambda3 = 1.0

    def forward(self, y_pred, y_true):
        pix_loss = self.Loss1(y_pred, y_true)
        ssim_loss = 1 - self.Loss2(y_pred, y_true)  # SSIM loss
        edge_loss = self.Loss3(y_pred, y_true)

        total_loss = (self.lambda1 * pix_loss) + (self.lambda2 * ssim_loss) + (self.lambda3 * edge_loss)
        return total_loss
    
# criterion = TotalLoss(pix_loss_fn= BerHuLoss(),ssim_loss_fn = ImprovedSSIMLoss(),edge_loss_fn = laplacian_edge_loss)