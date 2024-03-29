""" ORB feature detector module
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
    """Configure a new subparser for running the ORB feature detector.

    Args:
      subparser (Subparser): argument parser
    """

    """
    Subparser parameters
    Args:
      image (str): image path
      n-features (int): number of features to retain [default = 500]
    """
    parser = subparsers.add_parser("orb", help="ORB feature detector")
    parser.add_argument(
        "image", type=str, help="Image path on which to run the ORB feature detector"
    )
    parser.add_argument(
        "--n-features",
        "-NF",
        type=int,
        default=500,
        help="Number of features to retain",
    )
    # set the main function to run when ORB is called from the command line
    parser.set_defaults(func=main)


def main(args: Namespace) -> None:
    r"""Checks the command line arguments and then runs the ORB algorithm
    on an image, showing the result.
    This is employed only for visualization purposes

    Args:
      args (Namespace): command line arguments

    Raises:
      AssertionError: if the image at `image` does not exists
    """
    print("\n### ORB feature detector ###")
    print("> Parameters:")
    for p, v in zip(args.__dict__.keys(), args.__dict__.values()):
        print("\t{}: {}".format(p, v))
    print("\n")

    # assert an exception if the image does not exists
    assert os.path.exists(args.image), "Image passed does not exist"

    # read the colored version of the image
    image_bgr = cv2.imread(args.image, cv2.IMREAD_COLOR)

    # call the ORB algorithm
    orb_kp, _ = orb(frame=image_bgr, n_features=args.n_features)

    # draw the keypoints
    orb_image = draw_features_keypoints(image_bgr, orb_kp)

    # plot orb_image
    plot_image(orb_image, f"ORB descriptors {os.path.basename(args.image)}")


def orb(frame: np.ndarray, n_features: int) -> Tuple[cv2.KeyPoint, np.ndarray]:
    """Apply the ORB feature detector on a frame

    Args:
      frame (np.ndarray): frame [BGR]
      n_features (int): number of features to retain

    Returns:
      Tuple[cv2.KeyPoint, np.ndarray]: ORB keypoints and descriptors of the frame
    """
    # load the frame as grayscale
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ORB
    orb = cv2.ORB_create(n_features)

    # finds the keypoint and the descriptors
    keypoints, descriptors = orb.detectAndCompute(frame_gray, None)

    return keypoints, descriptors
