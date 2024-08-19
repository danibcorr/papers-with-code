# --------------------------------------------------------------------------------------------
# LIBRARIES
# --------------------------------------------------------------------------------------------

import numpy as np
from numba import njit

# --------------------------------------------------------------------------------------------
# DEFINITIONS OF FUNCTIONS
# --------------------------------------------------------------------------------------------

@njit
def sample_beta_distribution(size: int, concentration_0: float = 0.2, concentration_1: float = 0.2) -> np.ndarray:

   """
   Samples from a beta distribution using the gamma distribution method.

   Args:
       size (int): Number of samples to generate.
       concentration_0 (float): Alpha parameter of the beta distribution.
       concentration_1 (float): Beta parameter of the beta distribution.

   Returns:
       np.ndarray: Samples from the beta distribution.
   """

   gamma_1_sample = np.array([np.random.gamma(concentration_1) for _ in range(size)])
   gamma_2_sample = np.array([np.random.gamma(concentration_0) for _ in range(size)])

   return gamma_1_sample / (gamma_1_sample + gamma_2_sample)

@njit
def clip(value: float, min_value: float, max_value: float) -> float:

   """
   Clips a value to be within a specified range.

   Args:
       value (float): The value to be clipped.
       min_value (float): The minimum allowed value.
       max_value (float): The maximum allowed value.

   Returns:
       float: The clipped value.
   """

   return max(min_value, min(value, max_value))

@njit
def get_box(lambda_value: float, height: int, width: int) -> tuple:

   """
   Generates a random bounding box for CutMix.

   Args:
       lambda_value (float): The mixing ratio from the beta distribution.
       height (int): Height of the image.
       width (int): Width of the image.

   Returns:
       tuple: Coordinates of the bounding box (x1, y1, x2, y2).
   """

   cut_rat = np.sqrt(1.0 - lambda_value)
   cut_w = np.int32(width * cut_rat)
   cut_h = np.int32(height * cut_rat)
   cut_x = np.random.randint(width)
   cut_y = np.random.randint(height)
   
   boundaryx1 = clip(cut_x - cut_w // 2, 0, width)
   boundaryy1 = clip(cut_y - cut_h // 2, 0, height)
   boundaryx2 = clip(cut_x + cut_w // 2, 0, width)
   boundaryy2 = clip(cut_y + cut_h // 2, 0, height)
   
   return boundaryx1, boundaryy1, boundaryx2, boundaryy2

@njit
def cutmix_one_sample(train_ds_one: tuple, train_ds_two: tuple, alpha: float, beta: float) -> tuple:

   """
   Applies CutMix augmentation to a single pair of samples.

   Args:
       train_ds_one (tuple): First sample (image, label).
       train_ds_two (tuple): Second sample (image, label).
       alpha (float): Alpha parameter for beta distribution.
       beta (float): Beta parameter for beta distribution.

   Returns:
       tuple: Augmented image and interpolated label.
   """

   (image1, label1), (image2, label2) = train_ds_one, train_ds_two
   height, width, _ = image1.shape
   lambda_value = sample_beta_distribution(1, alpha, beta)[0]
   bbx1, bby1, bbx2, bby2 = get_box(lambda_value, height, width)
   
   # Apply CutMix
   image1[:, bbx1:bbx2, bby1:bby2] = image2[:, bbx1:bbx2, bby1:bby2]
   
   # Calculate the true lambda (proportion of image1)
   lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (height * width))
   
   # Interpolate labels
   label = lambda_value * label1 + (1 - lambda_value) * label2

   return image1, label

@njit
def cut_mix(dataset_one: tuple, dataset_two: tuple, alpha: float, beta: float) -> tuple:

   """
   Applies CutMix augmentation to two datasets.

   Args:
       dataset_one (tuple): First dataset (images, labels).
       dataset_two (tuple): Second dataset (images, labels).
       alpha (float): Alpha parameter for beta distribution.
       beta (float): Beta parameter for beta distribution.

   Returns:
       tuple: Augmented images and interpolated labels.
   """

   (ds_one, labels_one) = dataset_one
   (ds_two, labels_two) = dataset_two
   images = np.empty_like(ds_one)
   labels = np.empty_like(labels_one)

   for i in range(ds_one.shape[0]):
    
       images[i], labels[i] = cutmix_one_sample((ds_one[i], labels_one[i]), (ds_two[i], labels_two[i]), alpha, beta)
   
   return images, labels