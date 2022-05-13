""" SIFT feature detector module
"""
from argparse import _SubParsersAction as Subparser
from argparse import Namespace
import cv2
from fdt.detection.utils import draw_features_keypoints
from fdt.plotter import plot_image
import numpy as np
import os
from typing import Tuple


def configure_subparsers(subparsers: Subparser) -> None:
    """Configure a new subparser for running the SIFT feature detector.

    Args:
      subparser (Subparser): argument parser
    """

    """
    Subparser parameters
    Args:
      image (str): image path
      nfeatures (int) [Optional]: number of features to retain [default = 500]
    """
    parser = subparsers.add_parser("sift", help="SIFT feature detector")
    parser.add_argument(
        "image", type=str, help="Image path on which to run the SIFT feature detector"
    )
    parser.add_argument(
        "--nfeatures", "-NF", type=int, default=500, help="Number of features to retain"
    )
    # set the main function to run when SIFT is called from the command line
    parser.set_defaults(func=main)


def main(args: Namespace) -> None:
    r"""Checks the command line arguments and then runs the SIFT algorithm
    on an image, showing the result.
    This is employed only for visualization purposes

    Args:
      args (Namespace): command line arguments

    Raises:
      AssertionError: if the image at `image` does not exists
    """
    print("\n### SIFT feature detector ###")
    print("> Parameters:")
    for p, v in zip(args.__dict__.keys(), args.__dict__.values()):
        print("\t{}: {}".format(p, v))
    print("\n")

    # assert an exception if the image does not exists
    assert os.path.exists(args.image), "Image passed does not exist"

    # read the colored version of the image
    image_bgr = cv2.imread(args.image)

    # call the SIFT algorithm
    sift_kp, sift_desc = sift(frame=image_bgr, n_features=args.nfeatures)

    # draw the keypoints
    sift_image = draw_features_keypoints(image_bgr, sift_kp)

    # plot sift_image
    plot_image(sift_image, f"SIFT descriptors {os.path.basename(args.image)}")


def sift(frame: np.ndarray, n_features: int) -> Tuple[np.ndarray, np.ndarray]:
    """Apply the SIFT feature detector on a frame

    Args:
      frame (np.ndarray): frame [BGR]

    Returns:
      Tuple[np.ndarray, np.ndarray]: SIFT keypoints and descriptors of the frame
    """
    # load the frame as grayscale
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect SIFT features
    sift = cv2.SIFT_create(n_features)

    # finds the keypoint and the descriptors
    keypoints, descriptors = sift.detectAndCompute(frame_gray, None)

    # return keypoints and descriptors
    return keypoints, descriptors
