"""Feature matcher
"""
import cv2
from cv2 import BFMatcher
import numpy as np
from typing import List


def match_features(
    matcher: BFMatcher,
    first_descriptors: np.ndarray,
    second_descriptors: np.ndarray,
    max_matching_distance: int = 150,
) -> List[np.ndarray]:
    """Method which returns the effective matches between the objects employing a brute force
    feature matcher

    Args:
      matcher (BFMatcher): brute force matcher
      first_descriptors (np.ndarray): fist image descriptors
      second_descriptors (np.ndarray): second image descriptors
      max_matching_distance (int): maximum distance to consider when matching points

    Returns:
      List[np.ndarray]: list of effective matches found by the algorithm
    """
    # matching point
    matching_points = matcher.match(first_descriptors, second_descriptors)
    # filter matches
    effective_matches = list(
        filter(lambda x: x.distance < max_matching_distance, matching_points)
    )
    # return the effective_matches
    return effective_matches
