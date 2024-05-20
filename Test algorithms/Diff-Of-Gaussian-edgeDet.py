import numpy as np
from scipy.ndimage import convolve

def gaussian_kernel(size, sigma):
    """Generates a Gaussian kernel."""
    ax = np.linspace(-(size - 1) / 2., (size - 1) / 2., size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sigma))
    return kernel / np.sum(kernel)

def dog_edge_loss(y_true, y_pred):
    """
    Calculate the Difference of Gaussian (DoG) edge loss.

    Parameters:
    y_true (numpy array): Ground truth depth maps of shape [b, c, h, w].
    y_pred (numpy array): Predicted depth maps of shape [b, c, h, w].

    Returns:
    float: The DoG edge loss.
    """
    # Define Gaussian kernels
    gaussian1 = np.array([[1,  2, 1],
                          [2,  4, 2],
                          [1,  2, 1]]) / 16.0

    gaussian2 = np.array([[1,  4,  6,  4, 1],
                          [4, 16, 24, 16, 4],
                          [6, 24, 36, 24, 6],
                          [4, 16, 24, 16, 4],
                          [1,  4,  6,  4, 1]]) / 256.0

    def apply_filter(image, kernel):
        return convolve(image, kernel, mode='reflect')

    # Initialize the loss
    loss = 0.0

    # Iterate over the batch and channels
    for i in range(y_true.shape[0]):  # Batch size
        for j in range(y_true.shape[1]):  # Channels
            # Apply Gaussian filters
            y_true_g1 = apply_filter(y_true[i, j], gaussian1)
            y_true_g2 = apply_filter(y_true[i, j], gaussian2)
            y_pred_g1 = apply_filter(y_pred[i, j], gaussian1)
            y_pred_g2 = apply_filter(y_pred[i, j], gaussian2)

            # Compute the DoG edge loss
            term1 = np.abs(y_pred_g1 - y_pred_g2)
            term2 = np.abs(y_true_g1 - y_true_g2)
            loss += np.mean(np.abs(term1 - term2))

    # Normalize the loss by the batch size and number of channels
    loss /= (y_true.shape[0] * y_true.shape[1])

    return loss

# # Example Usage
# # Example ground truth and predicted depth-maps
# y_true = np.random.rand(4, 1, 64, 64)  # Example batch of ground truth depth maps
# y_pred = np.random.rand(4, 1, 64, 64)  # Example batch of predicted depth maps

# # Calculate DoG edge loss
# loss = dog_edge_loss(y_true, y_pred)
# print(f"DoG Edge Loss: {loss}")
