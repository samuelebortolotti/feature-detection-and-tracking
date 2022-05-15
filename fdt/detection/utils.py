"""Utils module.
It contains useful miscellaneous functions
"""
import cv2
import numpy as np


def draw_features_keypoints(image: np.ndarray, keypoints: np.ndarray) -> np.ndarray:
    r"""Draw the feature keypoints on the image passed

    Args:
      image (np.ndarray): image [BGR]
      keypoints (np.ndarray): features keypoints

    Returns:
      np.ndarray: image with the descriptors highlighted on top of it
    """
    # Include the Keypoints
    img_kp = cv2.drawKeypoints(
        image,
        keypoints,
        image,
        color=(0, 0, 255),
        flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT,
    )

    return img_kp
