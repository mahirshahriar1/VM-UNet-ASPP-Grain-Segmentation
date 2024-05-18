import torch
import torch.nn as nn
import torch.fft as fft
import torch.nn.functional as F

class FourierEdgeLoss(nn.Module):
    def __init__(self):
        super(FourierEdgeLoss, self).__init__()

    def forward(self, pred, target):
        # Fourier transform to frequency domain
        pred_freq = fft.fftn(pred, dim=(-2, -1))
        target_freq = fft.fftn(target, dim=(-2, -1))

        # Apply high-pass filter (enhance high frequencies which represent edges)
        high_pass_filter = self._high_pass_filter(pred.shape[-2], pred.shape[-1])
        pred_filtered = pred_freq * high_pass_filter
        target_filtered = target_freq * high_pass_filter

        # Inverse Fourier transform to spatial domain
        pred_edges = fft.ifftn(pred_filtered, dim=(-2, -1)).abs()
        target_edges = fft.ifftn(target_filtered, dim=(-2, -1)).abs()

        # Compute binary cross-entropy loss between edges
        edge_loss = F.mse_loss(pred_edges, target_edges)

        return edge_loss

    def _high_pass_filter(self, height, width):
        x = torch.linspace(-0.5, 0.5, steps=width).unsqueeze(0).repeat(height, 1)
        y = torch.linspace(-0.5, 0.5, steps=height).unsqueeze(1).repeat(1, width)
        radius = torch.sqrt(x**2 + y**2)
        filter = 1 - torch.exp(-radius**2 / (2 * (0.1 ** 2)))
        return filter.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions

import monai

class CombinedLoss(monai.losses.DiceCELoss):
    def __init__(self, lambda_edge=0.5, lambda_fft=0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sobel_edge_loss = SobelEdgeLoss()
        self.fourier_edge_loss = FourierEdgeLoss()
        self.lambda_edge = lambda_edge
        self.lambda_fft = lambda_fft

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        total_loss = super().forward(input, target)
        sobel_edge_loss = self.sobel_edge_loss(input, target)
        fourier_edge_loss = self.fourier_edge_loss(input, target)
        combined_loss = total_loss + self.lambda_edge * sobel_edge_loss + self.lambda_fft * fourier_edge_loss
        return combined_loss
