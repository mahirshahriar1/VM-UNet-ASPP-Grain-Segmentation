import torch
import torch.nn as nn

class BerHuLoss(nn.Module):
    def __init__(self):
        super(BerHuLoss, self).__init__()

    def forward(self, y_pred, y_true):
        abs_error = torch.abs(y_pred - y_true)
        max_error = torch.max(abs_error)
        threshold = 0.2 * max_error

        # Applying the BerHu loss function
        mask = abs_error <= threshold
        loss = torch.where(
            mask,
            abs_error,
            (abs_error ** 2 + threshold ** 2) / (2 * threshold)
        )

        return torch.mean(loss)

# # Example usage:
# if __name__ == "__main__":
#     # Dummy data
#     y_pred = torch.randn(4, 3, 256, 256)
#     y_true = torch.randn(4, 3, 256, 256)

#     # Loss computation
#     berhu_loss = BerHuLoss()
#     loss = berhu_loss(y_pred, y_true)
#     print(f'BerHu Loss: {loss.item()}')
