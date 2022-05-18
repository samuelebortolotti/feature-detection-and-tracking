""" Harris Corner detector
"""
from argparse import _SubParsersAction as Subparser
from argparse import Namespace
import cv2
import fdt.config.harris_conf as config
from fdt.detection.utils import draw_features_keypoints
from fdt.plotter import plot_image
import numpy as np
import os
from typing import Optional, Tuple


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
      config-file (bool): use the automatic configuration, provided in the `config` folder for the non-specified arguments
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
        help="the size of neighbourhood considered for corner detection",
    )
    parser.add_argument(
        "--k-size",
        "-KS",
        type=int,
        default=3,
        help="aperture parameter of the Sobel derivative used",
    )
    parser.add_argument(
        "--k",
        "-K",
        type=float,
        default=0.04,
        help="Harris detector free parameter in the equation",
    )
    parser.add_argument(
        "--tresh",
        "-T",
        type=float,
        default=0.5,
        help="Harris detector good point threshold (the selected points are harris*thres)",
    )
    parser.add_argument(
        "--config-file",
        "-CF",
        action="store_true",
        help="Whether to load the configuration from the configuration file",
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
    harris_kp, _ = harris(
        frame=image_bgr,
        block_size=args.block_size,
        k_size=args.k_size,
        k=args.k,
        tresh=args.tresh,
        config_file=args.config_file,
    )

    # draw the keypoints
    harris_img = draw_features_keypoints(image_bgr, harris_kp)

    # plot harris_img
    plot_image(harris_img, f"Harris descriptors {os.path.basename(args.image)}")


def load_params(
    block_size: int, k_size: int, k: float, tresh: float, conf_file: bool
) -> Tuple[int, int, float, float]:
    """Loads the parameters from the `config` file if they are not provided and the config_file flag is
    specified

    Args:
      block_size (int): the size of neighbourhood considered for corner detection
      k_size (int): aperture parameter of the Sobel derivative used
      k (float): Harris detector free parameter in the equation
      thresh (float): Harris detector good point threshold (the selected points are harris*thres)
      conf_file (bool): use the automatic configuration, provided in the `config` folder for the non-specified arguments

    Returns:
      Tuple[int, int, float, float]: respectively Harris block size, k size, k and treshold multiplier
    """
    # if the file is not set then it makes no sense to load anything
    if not conf_file:
        return block_size, k_size, k, tresh

    # set the configuration data
    block_size_l = (
        config.current_conf["block_size"] if block_size is None else block_size
    )
    k_size_l = config.current_conf["k_size"] if k_size is None else k_size
    k_l = config.current_conf["k"] if k is None else k
    tresh_l = config.current_conf["tresh"] if tresh is None else tresh

    # override the configuration with those loaded
    return block_size_l, k_size_l, k_l, tresh_l


def harris(
    frame: np.ndarray,
    block_size: int,
    k_size: int,
    k: float,
    tresh: float,
    config_file=bool,
) -> Tuple[cv2.KeyPoint, np.ndarray]:
    """Apply the Harris corner detector on a frame

    Args:
      block_size (int): the size of neighbourhood considered for corner detection
      k_size (int): aperture parameter of the Sobel derivative used
      k (float): Harris detector free parameter in the equation
      thresh (float): Harris detector good point threshold (the selected points are harris*thres)
      config_file (bool): use the automatic configuration, provided in the `config` folder for the non-specified arguments

    Returns:
      Tuple[cv2.KeyPoint, np.ndarray]: Harris keypoints and descriptors of the frame [**Note**, the Harris corner detector has no
      descriptor, thus I have employed SIFT for computing only the descriptors based on the Keypoints detected by Harris]
    """

    # load parameters
    block_size, k_size, k, tresh = load_params(
        block_size, k_size, k, tresh, config_file
    )

    # load the frame as grayscale
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # run the Harris corner detector
    harris = cv2.cornerHarris(frame_gray, block_size, k_size, k)

    # SIFT is employed only for descriptor extraction purposes
    sift_desc_extractor = cv2.SIFT_create()

    # dilate method to mark the corners in the returned image, basically, it adds pixels to the corners
    harris = cv2.dilate(harris, None)

    # threshold for the optimal corners, it may vary depending on the image.
    keypoints = np.argwhere(harris > tresh * harris.max())

    # convert numpy.ndarray points into opencv KeyPoints
    keypoints = [cv2.KeyPoint(int(k[1]), int(k[0]), 1) for k in keypoints]

    # use SIFT so as to extract the descriptors
    keypoints, descriptors = sift_desc_extractor.compute(frame_gray, keypoints)

    # return keypoints and not existing descriptors
    return keypoints, descriptors
