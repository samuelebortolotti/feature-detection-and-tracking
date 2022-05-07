""" SIFT feature detector module
"""
from argparse import _SubParsersAction as Subparser
from argparse import Namespace
import cv2
from fdt.plotter import plot_image
import numpy as np
import os


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
    r"""Checks the command line arguments and then runs the sift algorithm

    Args:
      args (Namespace): command line arguments
    """

    print("\n### SIFT feature detector ###")
    print("> Parameters:")
    for p, v in zip(args.__dict__.keys(), args.__dict__.values()):
        print("\t{}: {}".format(p, v))
    print("\n")

    # call the sift algorithm
    sift_image = sift(image_path=args.image, n_features=args.nfeatures)

    # plot sift_image
    plot_image(sift_image, f"SIFT descriptors {os.path.basename(args.image)}")


def sift(image_path: str, n_features: int) -> np.ndarray:
    r"""Apply the SIFT feature detector on an image

    Args:
      image_path (str): image path

    Returns:
      np.ndarray: gray scale image containing the SIFT keypoints

    Raises:
      AssertionError: if the image at `image_path` does not exists
    """
    # assert if the path does not exits
    assert os.path.exists(image_path), "Image passed does not exists"
    # loat the image as grayscale image
    image_bgr = cv2.imread(image_path)
    image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # Step 1: detect SIFT features
    sift = cv2.SIFT_create(n_features)
    # finds the keypoint
    keypoints, descriptors = sift.detectAndCompute(image_gray, None)

    # Step 2: shows the SIFT keypoints
    img_sfit = cv2.drawKeypoints(
        image_gray,
        keypoints,
        image_gray,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )

    return img_sfit
