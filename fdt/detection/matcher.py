"""Feature matcher
"""
from argparse import _SubParsersAction as Subparser
from argparse import Namespace
import cv2
from cv2 import BFMatcher
import imutils
from fdt.detection.blob import blob
from fdt.detection.harris import harris
from fdt.detection.orb import orb
from fdt.detection.sift import sift
import numpy as np
import os
from typing import List, Union, Any
from . import METHODS


def from_matches_to_points(
    keypoints: List[cv2.KeyPoint], matches: List[cv2.DMatch]
) -> List[cv2.KeyPoint]:
    """Method which converts the matched points from the °cv2.DMatch° class to
    `cv2.KeyPoint`

    Args:
      keypoints (List[cv2.KeyPoint]): list of keypoints
      matches (List[cv2.DMatch]): list of matches

    Returns:
        List[cv2.KeyPoint]: list of keypoints
    """
    points = []
    for match in matches:
        points.append(keypoints[match.trainIdx].pt)
    return points


def match_features(
    matcher: BFMatcher,
    first_descriptors: np.ndarray,
    second_descriptors: np.ndarray,
    second_keypoints: np.ndarray,
    max_matching_distance: int = 150,
) -> Union[List[cv2.KeyPoint], List[cv2.DMatch]]:
    """Method which returns the effective matches between the objects employing a brute force
    feature matcher

    Args:
      matcher (BFMatcher): brute force matcher
      first_descriptors (np.ndarray): fist image descriptors
      second_descriptors (np.ndarray): second image descriptors
      second_keypoints (np.ndarray): second image keypoints
      max_matching_distance (int): maximum distance to consider when matching points

    Returns:
      List[np.ndarray]: list of effective matches found by the algorithm
      List[cv2.DMatch]: matching points
    """
    # matching point
    matching_points = matcher.match(first_descriptors, second_descriptors)
    # filter matches by distance
    effective_matches = list(
        filter(lambda x: x.distance < max_matching_distance, matching_points)
    )
    # return the effective_matches
    return (
        from_matches_to_points(second_keypoints, effective_matches),
        effective_matches,
    )


def draw_features_matched(
    reference_image: np.ndarray,
    current_image: np.ndarray,
    reference_keypoints: np.ndarray,
    current_keypoints: np.ndarray,
    good_matches: np.ndarray,
    draw_matches_flags: int = 2,
):
    """Method which draws the good matches found on the current image having as a reference the
    target image

    Args:
      reference_image (np.ndarray): reference frame
      current_image (np.ndarray): current frame
      reference_keypoints (np.ndarray): reference keypoints
      good_matches (List[np.ndarray]): good matching pairs
      draw_matches_flags (int): drawMatches opencv function [default = 2]

    Returns:
      np.ndarray: image on which the good matching pairs are projected
    """
    # Draw the best matches
    matches_image = cv2.drawMatches(
        reference_image,
        reference_keypoints,
        current_image,
        current_keypoints,
        good_matches,
        draw_matches_flags,
    )
    return matches_image


def configure_subparsers(subparsers: Subparser) -> None:
    """Configure a new subparser for running the Kalman feature tracking

    Args:
      subparser (Subparser): argument parser
    """

    """
    Subparser parameters
    Args:
      method (str): feature detector
      nfeatures (int): number of features for the feature detector [default: 100]
      video (str): video file path if you are willing to run the algorithm on a video
      matchingdist (int): matching distance for the matcher [default=150]
      flann (bool): whether to use the flann based matcher [default=False]
      frameupdate (int): after how many frame to recalibrate the features [default=50]
    """

    parser = subparsers.add_parser("matcher", help="Feature matcher")
    parser.add_argument(
        "method",
        type=str,
        default="sift",
        choices=METHODS,
        help="Which feature to use in order to perform the matching",
    )
    parser.add_argument(
        "--n-features",
        "-NF",
        type=int,
        default=100,
        help="Number of features to retain",
    )
    parser.add_argument(
        "--flann", "-F", action="store_true", help="Use the FLANN matcher"
    )
    parser.add_argument(
        "--matching-distance", "-MD", type=int, default=150, help="Matching distance"
    )
    parser.add_argument(
        "--video", "-V", type=str, help="Video on which to run the Kalman filter"
    )
    parser.add_argument(
        "--frame-update",
        "-FU",
        type=int,
        default=50,
        help="After how many frames to recalibrate",
    )
    # set the main function to run when Kalman is called from the command line
    parser.set_defaults(func=main)


def main(args: Namespace) -> None:
    """Checks the command line arguments and then runs the matcher

    Args:
      args (Namespace): command line arguments
    """
    print("\n### Feature matcher ###")
    print("> Parameters:")
    for p, v in zip(args.__dict__.keys(), args.__dict__.values()):
        print("\t{}: {}".format(p, v))
    print("\n")

    # check wether the video exists
    assert args.video is None or os.path.exists(
        args.video
    ), "Video passed does not exist"

    # call the feature matcing algorithm
    feature_matching(
        camera_index=args.camera,
        video=args.video,
        method=args.method,
        n_features=args.n_features,
        matching_distance=args.matching_distance,
        update_every_n_frame=args.frame_update,
        flann=args.flann,
    )


def feature_matching(
    camera_index: int,
    video: str,
    method: str,
    n_features: int,
    matching_distance: int,
    update_every_n_frame: int,
    flann: bool,
) -> None:
    """Matches the features using a BFMatcher and the first frame as a reference according
    to the method passed

    Args:
      camera_index (int): camera index
      video (str): video path if any
      method (str): feature detector method
      n_features (int): number of features to extract
      matchingdist (int): matching distance for the matcher
      update_every_n_frame (int): after how many frame to recalibrate the features
      flann (bool): whether to use the flann based matcher
    """
    # video capture input
    cap_input = video
    if video is None:
        cap_input = camera_index

    # open the capture
    cap = cv2.VideoCapture(cap_input)

    # set the parameters of the matcher according to the chosen feature detection method
    if method == "sift":
        # SIFT
        feature_extract = sift
        # configuration of the feature extractor
        feature_extract_conf = {"n_features": n_features}
        feature_matching_mode = cv2.NORM_L2
        FLANN_INDEX_KDTREE = 1
        feature_index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    elif method == "orb":
        # ORB
        feature_extract = orb
        # configuration of the feature extractor
        feature_extract_conf = {"n_features": n_features}
        feature_matching_mode = cv2.NORM_HAMMING
        # FLANN index LSH
        FLANN_INDEX_LSH = 6
        feature_index_params = dict(
            algorithm=FLANN_INDEX_LSH,
            table_number=6,
            key_size=12,
            multi_probe_level=1,
        )
    elif method == "harris":
        # Harris corner detector
        feature_extract = harris
        # load config from `config` file
        feature_extract_conf = {
            "block_size": None,
            "k_size": None,
            "k": None,
            "tresh": None,
            "config_file": True,
        }
        # as it employs SIFT descriptors the brute force matcher will employ the L2 NORM
        feature_matching_mode = cv2.NORM_L2
        # mode for the FLANN matcher
        FLANN_INDEX_KDTREE = 1
        # feature parameters for FLANN
        feature_index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    else:
        # Blob detector
        feature_extract = blob
        # load config from `config` file
        feature_extract_conf = {
            "blob_param_dict": {
                "filterByColor": None,
                "blobColor": None,
                "filterByArea": None,
                "minArea": None,
                "maxArea": None,
                "filterByCircularity": None,
                "minCircularity": None,
                "maxCircularity": None,
                "filterByConvexity": None,
                "minConvexity": None,
                "maxConvexity": None,
                "filterByInertia": None,
                "minInertiaRatio": None,
                "maxInertiaRatio": None,
                "minThreshold": None,
                "maxThreshold": None,
                "thresholdStep": None,
                "minDistBetweenBlobs": None,
                "minRepeatability": None,
            },
            "config_file": True,
        }
        # as it employs SIFT descriptors the brute force matcher will employ the L2 NORM
        feature_matching_mode = cv2.NORM_L2
        # mode for the FLANN matcher
        FLANN_INDEX_KDTREE = 1
        # feature parameters for FLANN
        feature_index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)

    # Brute force point matcher
    if not flann:
        matcher = cv2.BFMatcher(feature_matching_mode, crossCheck=True)
    else:
        # FLANN matcher
        search_params = dict(checks=100)
        matcher = cv2.FlannBasedMatcher(feature_index_params, search_params)

    # capture the first frame
    ret, reference_frame = cap.read()
    # assert an exception whenever the return code is zero
    assert ret, "Error in reading the first frame from Video Capture"

    # extract the features from the first frame
    keypoints_to_match, descriptors_to_match = feature_extract(reference_frame, **feature_extract_conf)

    # frame_counter
    frame_counter = 0

    # processing the video
    while cap.isOpened():

        # read the frame
        ret, frame = cap.read()
        # assert an exception whenever the return code is zero
        assert ret, "Error in reading the frame from Video Capture"

        # Since it may be too big, but keeping the aspect ratio the same
        frame = imutils.resize(frame, width=550)

        # whether to update the feature matched
        if frame_counter % update_every_n_frame == 0:
            # new reference frame
            reference_frame = frame.copy()
            # extract the new reference keypoints and reference descriptors
            keypoints_to_match, descriptors_to_match = feature_extract(
                frame, **feature_extract_conf
            )

        # extract current keypoints and descriptors
        keypoints, descriptors = feature_extract(frame, **feature_extract_conf)

        # perform the brute force matching
        _, matching_points = match_features(
            matcher,
            first_descriptors=descriptors_to_match,
            second_descriptors=descriptors,
            second_keypoints=keypoints,
            max_matching_distance=matching_distance,
        )

        # show the matches frame
        cv2.imshow(
            f"{method} feature matching",
            draw_features_matched(
                reference_image=reference_frame,
                current_image=frame,
                reference_keypoints=keypoints_to_match,
                current_keypoints=keypoints,
                good_matches=matching_points,
            ),
        )

        # to wait
        wait = 1

        # exit when q is pressed
        if cv2.waitKey(wait) == ord("q"):
            break

        # increase the frame counter
        frame_counter += 1

    # Close windows
    cap.release()
    cv2.destroyAllWindows()
