import cv2
import numpy as np
import torch
import torch.nn.functional as F


def generate_boundary_mask(target):
    """
    Generates a binary boundary mask from the segmentation target.

    Args:
        target (torch.Tensor): A tensor of shape (B, H, W) containing class labels.

    Returns:
        torch.Tensor: A binary mask of shape (B, H, W) with boundaries marked as 1.
    """

    target_np = target.cpu().numpy().astype(np.uint8)
    
    boundary_masks = []
    
    for i in range(target_np.shape[0]):  # Iterate over each image in the batch
        # Apply Sobel filter to find edges
        sobel_x = cv2.Sobel(target_np[i], cv2.CV_64F, 1, 0, ksize=3)  # Sobel filter in X direction
        sobel_y = cv2.Sobel(target_np[i], cv2.CV_64F, 0, 1, ksize=3)  # Sobel filter in Y direction
        gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        gradient_magnitude = (gradient_magnitude - gradient_magnitude.min()) / (gradient_magnitude.max() - gradient_magnitude.min())
        _, boundary_mask = cv2.threshold(gradient_magnitude, 0.1, 1, cv2.THRESH_BINARY)  # Adjust threshold as needed

        boundary_masks.append(boundary_mask)
    boundary_masks_array = np.array(boundary_masks)  # Convert to a single numpy array

    boundary_masks_tensor = torch.tensor(boundary_masks_array, dtype=torch.float32).to(target.device)

    return boundary_masks_tensor

def boundary_loss(predicted, target, weight=8):
    target_boundary = generate_boundary_mask(target).unsqueeze(1)

    # Calculate binary cross-entropy loss
    bce_loss = F.binary_cross_entropy_with_logits(predicted, target_boundary, reduction='none')
    weights = weight * target_boundary + (1 - target_boundary)  # Weight boundaries more than non-boundaries

    # Apply weights to the loss
    weighted_loss = bce_loss * weights

    # Return the mean loss
    return weighted_loss.mean()
