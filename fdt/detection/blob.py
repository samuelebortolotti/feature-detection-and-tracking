""" Simple Blob Detector
"""
from argparse import _SubParsersAction as Subparser
from argparse import Namespace
import cv2
import fdt.config.blob_conf as config
from fdt.detection.utils import draw_features_keypoints
from fdt.plotter import plot_image
import numpy as np
import os
from typing import Tuple, Optional


def configure_subparsers(subparsers: Subparser) -> None:
    """Configure a new subparser for running the Simple Blob Detector

    Args:
      subparser (Subparser): argument parser
    """

    """
    Subparser parameters
    Args:
      image (str): image path
    """
    parser = subparsers.add_parser("blob", help="Simple blob detector")
    parser.add_argument(
        "image", type=str, help="Image path on which to run the Simple Blob detector"
    )
    parser.add_argument(
        "--blob-color",
        "-BC",
        type=int,
    )
    parser.add_argument(
        "--filter-by-area",
        "-FBA",
        type=bool,
    )
    parser.add_argument(
        "--min-area",
        "-MIA",
        type=float,
    )
    parser.add_argument(
        "--max-area",
        "-MXA",
        type=float,
    )
    parser.add_argument(
        "--filter-by-circularity",
        "-FBC",
        type=bool,
    )
    parser.add_argument(
        "--min-circularity" "-MIC",
        type=float,
    )
    parser.add_argument(
        "--max-circularity" "-MXC",
        type=float,
    )
    parser.add_argument(
        "--filter-by-convexity" "-FBX",
        type=bool,
    )
    parser.add_argument(
        "--min-convexity" "-MIX",
        type=float,
    )
    parser.add_argument(
        "--max-convexity" "-MXX",
        type=float,
    )
    parser.add_argument(
        "--filter-by-inertia" "-MXX",
        type=bool,
    )
    parser.add_argument(
        "--min-inertia" "-MII",
        type=float,
    )
    parser.add_argument(
        "--max-inertia" "-MXI",
        type=float,
    )
    parser.add_argument(
        "--min-threshold" "-MIT",
        type=float,
    )
    parser.add_argument(
        "--max-threshold" "-MXT",
        type=float,
    )
    parser.add_argument(
        "--threshold-step" "-TS",
        type=int,
    )
    parser.add_argument(
        "--min-dist-between-blobs" "-MDBB",
        type=float,
    )
    parser.add_argument(
        "--min-repetability" "-MR",
        type=float,
    )
    parser.add_argument(
        "--config-file",
        "-CF",
        action="store_true",
        help="Whether to load the configuration from the configuration file",
    )
    # set the main function to run when blob is called from the command line
    parser.set_defaults(func=main)


def main(args: Namespace) -> None:
    r"""Checks the command line arguments and then runs the Simple Blob Detector
    on an image, showing the result.
    This is employed only for visualization purposes

    Args:
      args (Namespace): command line arguments

    Raises:
      AssertionError: if the image at `image` does not exists
    """
    print("\n### Simple Blob feature detector ###")
    print("> Parameters:")
    for p, v in zip(args.__dict__.keys(), args.__dict__.values()):
        print("\t{}: {}".format(p, v))
    print("\n")

    # assert an exception if the image does not exists
    assert os.path.exists(args.image), "Image passed does not exist"

    # read the colored version of the image
    image_bgr = cv2.imread(args.image)

    # call the Blob detector
    blob_kp = blob(
        frame=image_bgr,
        filterByColor=args.filter_by_color,
        blobColor=args.blob_color,
        filterByArea=args.filter_by_area,
        minArea=args.min_area,
        maxArea=args.max_area,
        filterByCircularity=args.filter_by_circularity,
        minCircularity=args.min_circularity,
        maxCircularity=args.max_circularity,
        filterByConvexity=args.filter_by_convexity,
        minConvexity=args.min_convexity,
        maxConvexity=args.max_convexity,
        filterByInertia=args.filter_by_inertia,
        minInertiaRatio=args.min_inertia,
        maxInertiaRatio=args.max_inertia,
        minThreshold=args.min_threshold,
        maxThreshold=args.max_treshold,
        thresholdStep=args.threshold_step,
        minDistBetweenBlobs=args.min_dist_between_blobs,
        minRepeatability=args.min_repetability,
        config_file=args.config_file,
    )

    # draw the keypoints
    blob_img = draw_features_keypoints(image_bgr, blob_kp)

    # plot blob_img
    plot_image(
        blob_img, f"Simple Blob Detector descriptors {os.path.basename(args.image)}"
    )


def load_blob_params(
    filterByColor: bool,
    blobColor: int,
    filterByArea: bool,
    minArea: float,
    maxArea: float,
    filterByCircularity: bool,
    minCircularity: float,
    maxCircularity: float,
    filterByConvexity: bool,
    minConvexity: float,
    maxConvexity: float,
    filterByInertia: bool,
    minInertiaRatio: float,
    maxInertiaRatio: float,
    minThreshold: float,
    maxThreshold: float,
    thresholdStep: int,
    minDistBetweenBlobs: float,
    minRepeatability: float,
    config_file: bool,
    conf_file: bool,
):
    """TODO

    Args:
      frame (np.ndarray): frame [BGR]

    Returns:
      Tuple[np.ndarray, np.ndarray]: Harris keypoints and descriptors of the frame
    """
    # if the file is not set then it makes no sense to load anything
    if not conf_file:
        return (
            filterByColor,
            blobColor,
            filterByArea,
            minArea,
            maxArea,
            filterByCircularity,
            minCircularity,
            maxCircularity,
            filterByConvexity,
            minConvexity,
            maxConvexity,
            filterByInertia,
            minInertiaRatio,
            maxInertiaRatio,
            minThreshold,
            maxThreshold,
            thresholdStep,
            minDistBetweenBlobs,
            minRepeatability
        )

    filterByColor_l = config.current_conf["filterByColor"] if filterByColor is None else filterByColor
    blobColor_l = config.current_conf["blobColor"] if blobColor is None else blobColor
    filterByArea_l = config.current_conf["filterByArea"] if filterByArea is None else filterByArea
    minArea_l = config.current_conf["minArea"] if minArea is None else minArea
    maxArea_l = config.current_conf["maxArea"] if maxArea is None else maxArea
    filterByCircularity_l = config.current_conf["filterByCircularity"] if filterByCircularity is None else filterByCircularity
    minCircularity_l = config.current_conf["minCircularity"] if minCircularity is None else minCircularity
    maxCircularity_l = config.current_conf["maxCircularity"] if maxCircularity is None else maxCircularity
    filterByConvexity_l = config.current_conf["filterByConvexity"] if filterByConvexity is None else filterByConvexity
    minConvexity_l = config.current_conf["minConvexity"] if minConvexity is None else minConvexity
    maxConvexity_l = config.current_conf["maxConvexity"] if maxConvexity is None else maxConvexity
    filterByInertia_l = config.current_conf["filterByInertia"] if filterByInertia is None else filterByInertia
    minInertiaRatio_l = config.current_conf["minInertiaRatio"] if minInertiaRatio is None else minInertiaRatio
    maxInertiaRatio_l = config.current_conf["maxInertiaRatio"] if maxInertiaRatio is None else maxInertiaRatio
    minThreshold_l = config.current_conf["minThreshold"] if minThreshold is None else minThreshold
    maxThreshold_l = config.current_conf["maxThreshold"] if maxThreshold is None else maxThreshold
    thresholdStep_l = config.current_conf["thresholdStep"] if thresholdStep is None else thresholdStep
    minDistBetweenBlobs_l = config.current_conf["minDistBetweenBlobs"] if minDistBetweenBlobs is None else minDistBetweenBlobs
    minRepeatability_l = config.current_conf["minRepeatability"] if minRepeatability is None else minRepeatability

    # override the configuration with those loaded
    return (
        filterByColor_l,
        blobColor_l,
        filterByArea_l,
        minArea_l,
        maxArea_l,
        filterByCircularity_l,
        minCircularity_l,
        maxCircularity_l,
        filterByConvexity_l,
        minConvexity_l,
        maxConvexity_l,
        filterByInertia_l,
        minInertiaRatio_l,
        maxInertiaRatio_l,
        minThreshold_l,
        maxThreshold_l,
        thresholdStep_l,
        minDistBetweenBlobs_l,
        minRepeatability_l
    )


def blob(
    frame: np.ndarray,
    filterByColor: bool,
    blobColor: int,
    filterByArea: bool,
    minArea: float,
    maxArea: float,
    filterByCircularity: bool,
    minCircularity: float,
    maxCircularity: float,
    filterByConvexity: bool,
    minConvexity: float,
    maxConvexity: float,
    filterByInertia: bool,
    minInertiaRatio: float,
    maxInertiaRatio: float,
    minThreshold: float,
    maxThreshold: float,
    thresholdStep: int,
    minDistBetweenBlobs: float,
    minRepeatability: float,
    config_file: bool,
) -> Tuple[np.ndarray, np.array]:
    """Apply the Simple Blob Detector on a frame

    Args:
      frame (np.ndarray): frame [BGR]

    Returns:
      Tuple[np.ndarray, np.ndarray]: Simple Blob keypoints and descriptors of the frame
    """
    # load the frame as grayscale
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # set blob detector parameters
    blob_params = cv2.SimpleBlobDetector_Params()

    load_blob_params(blob_params)

    # set up the simple blob detector
    blob_detector = cv2.SimpleBlobDetector_create(blob_params)

    # run the simple blob detector
    keypoints = blob_detector.detect(frame_gray)

    # return keypoints
    return keypoints, None
