"""
Lung Segmentation Utility Script
--------------------------------

Description:
    This module provides three main functions:
        1. is_dir_path()  – Validate directory paths.
        2. segment_lung() – Perform lung region segmentation on CT slices.
        3. count_params() – Count the number of trainable parameters in a model.

Citation / Source Acknowledgement:
    The `segment_lung()` function was adapted from the preprocessing tutorial
    used in the **Kaggle Data Science Bowl 2017** competition:
    https://www.kaggle.com/c/data-science-bowl-2017
    Original method credited to community contributors.
    Modified and documented for research and educational purposes.
"""

import argparse
import os
import numpy as np
from medpy.filter.smoothing import anisotropic_diffusion
from scipy.ndimage import median_filter
from skimage import measure, morphology
import scipy.ndimage as ndimage
from sklearn.cluster import KMeans


def is_dir_path(string):
    """
    Check whether the provided path is a valid directory.

    Args:
        string (str): Path to validate.

    Returns:
        str: The same path if it is a valid directory.

    Raises:
        NotADirectoryError: If the path does not exist or is not a directory.
    """
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)


def segment_lung(img):
    """
    Segment the lung region from a CT image slice.
    (Adapted from the Kaggle Data Science Bowl 2017 tutorial.)

    Processing steps:
        1. Normalize image intensity (subtract mean, divide by standard deviation).
        2. Remove extreme pixel values to suppress noise.
        3. Apply median and anisotropic diffusion filters for denoising.
        4. Use K-Means (k=2) clustering to determine a threshold separating lung vs background.
        5. Perform morphological operations (erosion + dilation) to clean small regions.
        6. Label connected regions and keep those within anatomical lung boundaries.
        7. Generate a binary mask for the lungs and apply it to the normalized image.

    Args:
        img (numpy.ndarray): A single CT slice, typically 512×512 pixels.

    Returns:
        numpy.ndarray: Segmented lung image with background suppressed.
    """
    # Step 1: Normalize by mean and standard deviation
    mean = np.mean(img)
    std = np.std(img)
    img = (img - mean) / std

    # Step 2: Clip extreme values based on the central 300×300 region
    middle = img[100:400, 100:400]
    mean = np.mean(middle)
    max_val = np.max(img)
    min_val = np.min(img)
    img[img == max_val] = mean
    img[img == min_val] = mean

    # Step 3: Median filter + anisotropic diffusion smoothing
    img = median_filter(img, size=3)
    img = anisotropic_diffusion(img)

    # Step 4: Estimate threshold using K-Means clustering (2 clusters)
    kmeans = KMeans(n_clusters=2).fit(np.reshape(middle, [np.prod(middle.shape), 1]))
    centers = sorted(kmeans.cluster_centers_.flatten())
    threshold = np.mean(centers)
    thresh_img = np.where(img < threshold, 1.0, 0.0)

    # Step 5: Morphological operations to refine the mask
    eroded = morphology.erosion(thresh_img, np.ones([4, 4]))
    dilation = morphology.dilation(eroded, np.ones([10, 10]))

    # Step 6: Label connected regions and filter valid ones
    labels = measure.label(dilation)
    regions = measure.regionprops(labels)
    valid_labels = []
    for prop in regions:
        B = prop.bbox  # (min_row, min_col, max_row, max_col)
        if B[2] - B[0] < 475 and B[3] - B[1] < 475 and B[0] > 40 and B[2] < 472:
            valid_labels.append(prop.label)

    # Step 7: Construct final binary lung mask
    mask = np.zeros([512, 512], dtype=np.int8)
    for L in valid_labels:
        mask += np.where(labels == L, 1, 0)
    mask = morphology.dilation(mask, np.ones([10, 10]))

    # Return masked image (lungs only)
    return mask * img


def count_params(model):
    """
    Count the number of trainable parameters in a PyTorch model.

    Args:
        model (torch.nn.Module): Model to inspect.

    Returns:
        int: Total number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
