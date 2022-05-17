"""Utils module.
It contains useful miscellaneous functions
"""
import cv2
import numpy as np
from typing import Tuple


def draw_features_keypoints(
    image: np.ndarray, keypoints: np.ndarray, color: Tuple[int, int, int] = (0, 0, 255)
) -> np.ndarray:
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
        color=color,
        flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT,
    )

    return img_kp


def draw_lk_keypoints(
    frame: np.ndarray,
    lk_features: np.ndarray,
    radius: int,
    color: Tuple[int, int, int],
    thickness: int,
    line_type: int,
) -> np.ndarray:
    """Function which draws the keypoints returned by the Lucas-Kanade optical flow
    Args:
      frame (np.ndarray): frame on which to draw the keypoints
      lk_features (np.ndarray): keypoints found thanks to lucas-kanade optical flow
      radius (int): radius of the circle for the keypoints
      color (Typle[int, int, int]): color of the keypoints
      line_type (int): keypoints circle line type

    Returns:
      np.ndarray: new frame with the keypoints on top of it
    """
    lk_features = lk_features.astype(int)
    for corner in lk_features:
        x, y = corner.ravel()
        frame = cv2.circle(
            frame,
            (x, y),
            radius=radius,
            color=color,
            thickness=thickness,
            lineType=line_type,
        )
    return frame
