# pip install scipy
import numpy as np
from scipy.ndimage import gaussian_filter, laplace

def log_edge_loss(y_true, y_pred, sigma=1.0):
    """
    Calculate the Laplacian of Gaussian (LoG) edge loss.

    Parameters:
    y_true (numpy array): Ground truth depth maps of shape [b, c, h, w].
    y_pred (numpy array): Predicted depth maps of shape [b, c, h, w].
    sigma (float): Standard deviation for Gaussian kernel.

    Returns:
    float: The LoG edge loss.
    """
    # Ensure the input arrays are numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    def laplacian_of_gaussian(image, sigma):
        # Apply Gaussian filter
        smoothed = gaussian_filter(image, sigma=sigma)
        # Apply Laplacian filter
        log_image = laplace(smoothed)
        return log_image
    
    # Initialize the loss
    loss = 0.0
    
    # Iterate over the batch and channels
    for i in range(y_true.shape[0]):  # Batch size
        for j in range(y_true.shape[1]):  # Channels
            log_true = laplacian_of_gaussian(y_true[i, j], sigma)
            log_pred = laplacian_of_gaussian(y_pred[i, j], sigma)
            
            # Compute the edge loss as the mean absolute difference
            loss += np.mean(np.abs(log_true - log_pred))
    
    # Normalize the loss by the batch size and number of channels
    loss /= (y_true.shape[0] * y_true.shape[1])
    
    return loss

# # Example Usage
# # Example ground truth and predicted depth-maps
# y_true = np.random.rand(4, 1, 64, 64)  # Example batch of ground truth depth maps
# y_pred = np.random.rand(4, 1, 64, 64)  # Example batch of predicted depth maps

# # Calculate LoG edge loss
# loss = log_edge_loss(y_true, y_pred)
# print(f"LoG Edge Loss: {loss}")
