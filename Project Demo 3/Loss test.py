import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class PerceptualLoss(nn.Module):
    def __init__(self, layers=['conv1_2', 'conv2_2', 'conv3_3', 'conv4_3'], weights=None):
        super(PerceptualLoss, self).__init__()
        vgg = models.vgg16(pretrained=True).features
        self.layers = layers
        self.weights = weights if weights is not None else [1.0 / len(layers)] * len(layers)
        self.vgg_slices = nn.ModuleList()
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

        slice_layers = {
            'conv1_2': 4,
            'conv2_2': 9,
            'conv3_3': 16,
            'conv4_3': 23,
        }

        for layer_name in layers:
            slice_idx = slice_layers[layer_name]
            self.vgg_slices.append(vgg[:slice_idx + 1])

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, pred, target):
        if pred.size(1) == 1:  # If input is single-channel, convert to 3-channel
            pred = pred.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)

        device = pred.device
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)

        pred = (pred - self.mean) / self.std
        target = (target - self.mean) / self.std

        loss = 0.0
        for slice_net, weight in zip(self.vgg_slices, self.weights):
            slice_net = slice_net.to(device)
            pred_features = slice_net(pred)
            target_features = slice_net(target)
            loss += weight * F.mse_loss(pred_features, target_features)

        return loss

class SobelEdgeLoss(nn.Module):
    def __init__(self):
        super(SobelEdgeLoss, self).__init__()
        sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)
        self.bce_loss = nn.BCELoss()

    def forward(self, pred, target):
        device = pred.device
        sobel_x = self.sobel_x.to(device)
        sobel_y = self.sobel_y.to(device)

        pred_edges_x = F.conv2d(pred, sobel_x, padding=1)
        pred_edges_y = F.conv2d(pred, sobel_y, padding=1)
        pred_edges = torch.sqrt(pred_edges_x ** 2 + pred_edges_y ** 2 + 1e-7)

        target_edges_x = F.conv2d(target, sobel_x, padding=1)
        target_edges_y = F.conv2d(target, sobel_y, padding=1)
        target_edges = torch.sqrt(target_edges_x ** 2 + target_edges_y ** 2 + 1e-7)

        pred_edges = torch.sigmoid(pred_edges)
        target_edges = torch.sigmoid(target_edges)

        edge_loss = self.bce_loss(pred_edges, target_edges)

        return edge_loss

class CombinedLoss(nn.Module):
    def __init__(self, lambda_edge=0.5):
        super(CombinedLoss, self).__init__()
        self.bce_loss = nn.BCELoss()
        self.edge_loss = SobelEdgeLoss()
        self.perceptual_loss = PerceptualLoss()
        self.lambda_edge = lambda_edge

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce_loss(input, target)
        edge_loss = self.edge_loss(input, target)
        perceptual_loss = self.perceptual_loss(input, target)
        combined_loss = bce_loss + self.lambda_edge * edge_loss + perceptual_loss
        return combined_loss
criterion = CombinedLoss() 