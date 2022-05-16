""" Harris Corner detector
"""
from argparse import _SubParsersAction as Subparser
from argparse import Namespace
import cv2
from fdt.detection.utils import draw_features_keypoints
from fdt.plotter import plot_image
import numpy as np
import os


def configure_subparsers(subparsers: Subparser) -> None:
    """Configure a new subparser for running the Harris corner detector

    Args:
      subparser (Subparser): argument parser
    """

    """
    Subparser parameters
    Args:
      image (str): image path
      block-size (int): the size of neighbourhood considered for corner detection [default: 2]
      k-size (int): aperture parameter of the Sobel derivative used [default: 3]
      k (float): Harris detector free parameter in the equation [default 0.04]
      thresh (float): Harris detector good point threshold (the selected points are harris*thres) [default: 0.5]
    """
    parser = subparsers.add_parser("harris", help="Harris corner detector")
    parser.add_argument(
        "image", type=str, help="Image path on which to run the Harris corner detector"
    )
    parser.add_argument(
        "--block-size",
        "-BS",
        type=int,
        default=2,
        help="the size of neighbourhood considered for corner detection"
    )
    parser.add_argument(
        "--k-size",
        "-KS",
        type=int,
        default=3,
        help="aperture parameter of the Sobel derivative used"
    )
    parser.add_argument(
        "--k",
        "-K",
        type=float,
        default=0.04,
        help="Harris detector free parameter in the equation"
    )
    parser.add_argument(
        "--tresh",
        "-T",
        type=float,
        default=0.5,
        help="Harris detector good point threshold (the selected points are harris*thres)"
    )
    # set the main function to run when Harris is called from the command line
    parser.set_defaults(func=main)


def main(args: Namespace) -> None:
    r"""Checks the command line arguments and then runs the Harris corner detector
    on an image, showing the result.
    This is employed only for visualization purposes

    Args:
      args (Namespace): command line arguments

    Raises:
      AssertionError: if the image at `image` does not exists
    """
    print("\n### Harris feature detector ###")
    print("> Parameters:")
    for p, v in zip(args.__dict__.keys(), args.__dict__.values()):
        print("\t{}: {}".format(p, v))
    print("\n")

    # assert an exception if the image does not exists
    assert os.path.exists(args.image), "Image passed does not exist"

    # read the colored version of the image
    image_bgr = cv2.imread(args.image)

    # call the Harris corner detector algorithm
    harris_kp = harris(frame=image_bgr, block_size=args.block_size, k_size=args.k_size, k=args.k, tresh=args.tresh)

    # draw the keypoints
    harris_img = draw_features_keypoints(image_bgr, harris_kp)

    # plot harris_img
    plot_image(harris_img, f"Harris descriptors {os.path.basename(args.image)}")


def harris(frame: np.ndarray, block_size: int, k_size: int, k: float, tresh: float) -> np.ndarray:
    """Apply the Harris corner detector on a frame

    Args:
      frame (np.ndarray): frame [BGR]

    Returns:
      Tuple[np.ndarray, np.ndarray]: Harris keypoints and descriptors of the frame
    """
    # load the frame as grayscale
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # run the Harris corner detector
    harris = cv2.cornerHarris(frame_gray, block_size, k_size, k)

    # I will use the dilate method to mark the corners in the returned image. 
    # Basically, it adds pixels to the corners of objects in an image
    harris = cv2.dilate(harris, None)

    # Threshold for an optimal value, it may vary depending on the image.
    keypoints = np.argwhere(harris > tresh * harris.max())

    # convert numpy keypoints into opencv KeyPoints
    keypoints = [cv2.KeyPoint(int(k[0]), int(k[1]), 1) for k in keypoints]

    # return keypoints
    return keypoints