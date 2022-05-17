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
from typing import Any, Dict, Tuple


def configure_subparsers(subparsers: Subparser) -> None:
    """Configure a new subparser for running the Simple Blob Detector

    Args:
      subparser (Subparser): argument parser
    """

    """
    Subparser parameters
    Args:
      image (str): image path
      filterByColor (bool): wether to consider a specific colour as feature [default = False], since it is set to "false", then it finds bright and dark blobs, both.
      blobColor (int): blob colour, basically images are converted to many binary b/w layers. Then 0 searches for dark blobs, 255 searches for bright blobs. [default = 0]
      filterByArea (bool): Extracted blobs have an area between minArea (inclusive) and maxArea (exclusive) [default = True]
      minArea (float): min area of the blob to look for. Notice, this is highly depending on image resolution and dice size [default = 3.0]
      maxArea (float): maximum area of the blob to look for. Highly depending on image resolution. [default = 400.0]
      filterByCircularity (bool): whether to filter by the circluarity shape of the objects in the scene [default = True]
      minCircularity (float): 0 is rectangular, 1 is round. Not set because the dots are not always round when they are damaged, for example. [default = 0.0]
      maxCircularity (float): max circularity default [default = 3.40282346638b52886e38], in this case it has been set to infinity somehow.
      filterByConvexity (bool): whether to filter by convexity [default =  False]
      minConvexity (float): min convexity [default = 0.0]
      maxConvexity (float): max convexity [default = 3.4028234663852886e38], once again this is basically infinity
      filterByInertia (bool): basically a second way to find round blobs [default = True]
      minInertiaRatio (float): minimum ineria ratio, where 1 is round and 0 is basically everything [default = 0.55]
      maxInertiaRatio (float): maximum inertia ratio [default = 3.4028234663852886e38], basically it is infinity once again
      minThreshold (float): from where to start filtering the image [default = 0.0]
      maxThreshold (float): where to end filtering the image [default = 255.0]
      thresholdStep (int): steps to perform [default = 5]
      minDistBetweenBlobs (float): a distance used in order to avoid overlapping blobs, must be bigger than 0 for obvious reasons [default = 3.0]
      minRepeatability (float): if the same blob center is found at different threshold values (within a minDistBetweenBlobs), then it increases a counter for that blob.
      if the counter for each blob is >: minRepeatability, then it's a stable blob, and produces a KeyPoint, otherwise the blob is discarded [default = 2]
    """
    parser = subparsers.add_parser("blob", help="Simple blob detector")
    parser.add_argument(
        "image", type=str, help="Image path on which to run the Simple Blob detector"
    )
    parser.add_argument(
        "--filter-by-color",
        "-FBCO",
        type=bool,
        default=False,
        help="Wether to consider a specific colour"
    )
    parser.add_argument(
        "--blob-color",
        "-BC",
        type=int,
        default=0,
        help="Specific color to consider when filtering by color"
    )
    parser.add_argument(
        "--filter-by-area",
        "-FBA",
        type=bool,
        default=True,
        help="Whether to filter by area"
    )
    parser.add_argument(
        "--min-area",
        "-MIA",
        type=float,
        default=3.0,
        help="Min area to consider when filtering by area"
    )
    parser.add_argument(
        "--max-area",
        "-MXA",
        type=float,
        default=400.0,
        help="Max area to consider when filtering by area"
    )
    parser.add_argument(
        "--filter-by-circularity",
        "-FBC",
        type=bool,
        default=True,
        help="Whether to filter by circularity"
    )
    parser.add_argument(
        "--min-circularity",
        "-MIC",
        type=float,
        default=0.0,
        help="Min circularity to consider when filtering by circularity"
    )
    parser.add_argument(
        "--max-circularity",
        "-MXC",
        type=float,
        default=3.4028234663852886e38,
        help="Max circularity to consider when filtering by circularity"
    )
    parser.add_argument(
        "--filter-by-convexity",
        "-FBX",
        type=bool,
        default=False,
        help="Whether to filter by convexity"
    )
    parser.add_argument(
        "--min-convexity",
        "-MIX",
        type=float,
        default=0.0,
        help="Min convexity to consider when filtering by convexity"
    )
    parser.add_argument(
        "--max-convexity",
        "-MXX",
        type=float,
        default=3.4028234663852886e38,
        help="Max convexity to consider when filtering by convexity"
    )
    parser.add_argument(
        "--filter-by-inertia",
        "-FBI",
        type=bool,
        default=True,
        help="Whether to filter by inertia"
    )
    parser.add_argument(
        "--min-inertia",
        "-MII",
        type=float,
        default=0.55,
        help="Min inertia to consider when filtering by inertia"
    )
    parser.add_argument(
        "--max-inertia",
        "-MXI",
        type=float,
        default=3.4028234663852886e38,
        help="Max inertia to consider when filtering by inertia"
    )
    parser.add_argument(
        "--min-threshold",
        "-MIT",
        type=float,
        default=0.0,
        help="Min treshold to consider when filtering"
    )
    parser.add_argument(
        "--max-threshold",
        "-MXT",
        type=float,
        default=255.0,
        help="Max treshold to consider when filtering"
    )
    parser.add_argument(
        "--threshold-step",
        "-TS",
        type=int,
        default=5,
        help="Step to perform"
    )
    parser.add_argument(
        "--min-dist-between-blobs",
        "-MDBB",
        type=float,
        default=3.0,
        help="Min distance between blobs"
    )
    parser.add_argument(
        "--min-repeatability",
        "-MR",
        type=int,
        default=2,
        help="Repeatability for stable keypoints"
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

    # prepare the blob parameters
    blob_param_dict = {
        "filterByColor": args.filter_by_color,
        "blobColor": args.blob_color,
        "filterByArea": args.filter_by_area,
        "minArea": args.min_area,
        "maxArea": args.max_area,
        "filterByCircularity": args.filter_by_circularity,
        "minCircularity": args.min_circularity,
        "maxCircularity": args.max_circularity,
        "filterByConvexity": args.filter_by_convexity,
        "minConvexity": args.min_convexity,
        "maxConvexity": args.max_convexity,
        "filterByInertia": args.filter_by_inertia,
        "minInertiaRatio": args.min_inertia,
        "maxInertiaRatio": args.max_inertia,
        "minThreshold": args.min_threshold,
        "maxThreshold": args.max_threshold,
        "thresholdStep": args.threshold_step,
        "minDistBetweenBlobs": args.min_dist_between_blobs,
        "minRepeatability": args.min_repeatability,
    }

    # call the blob detector
    blob_kp, _ = blob(
        frame=image_bgr,
        blob_param_dict=blob_param_dict,
        config_file=args.config_file,
    )

    # draw the keypoints
    blob_img = draw_features_keypoints(image_bgr, blob_kp)

    # plot blob_img
    plot_image(
        blob_img, f"Simple Blob Detector descriptors {os.path.basename(args.image)}"
    )


def params_init(attributes: Dict[str, Any]) -> cv2.SimpleBlobDetector_Params:
    """Initializes the `cv2.SimpleBlobDetector_Params` object.

    Args:
      attributes (Dict[str, Any]): dictionary containing the named parameter for the `cv2.SimpleBlobDetector`
    Returns:
      cv2.SimpleBlobDetector_Params: parameters for the `cv2.SimpleBlobDetector`
    """
    # create the parameter object
    blob_params = cv2.SimpleBlobDetector_Params()
    # set the attributes of the SimpleBlobDetector
    for attribute, value in attributes.items():
        setattr(blob_params, attribute, value)
    return blob_params


def load_blob_params(blob_param_dict: Dict[str, Any], config_file: bool) -> cv2.SimpleBlobDetector_Params:
    """Loads the parameters from the `config` file if they are not provided and the config_file flag is
    specified

    Args:
      blob_param_dict (Dict[str, Any]): dictionary containing the named parameter for the `cv2.SimpleBlobDetector`
      conf_file (bool): use the automatic configuration, provided in the `config` folder for the non-specified arguments

    Returns:
      cv2.SimpleBlobDetector_Params: parameters for the `cv2.SimpleBlobDetector`
    """
    # if the file is not set then it makes no sense to load anything
    if not config_file:
        return params_init(attributes=blob_param_dict)

    # set the parameters
    for attribute, value in blob_param_dict.items():
        if value is None:
            blob_param_dict[attribute] = config.current_conf[attribute]

    # override the configuration with those loaded
    return params_init(attributes=blob_param_dict)


def blob(
    frame: np.ndarray,
    blob_param_dict: Dict[str, Any],
    config_file: bool,
) -> Tuple[cv2.KeyPoint, np.array]:

    """Apply the Simple Blob Detector on a frame

    Args:
      frame (np.ndarray): frame [BGR]
      blob_param_dict (Dict[str, Any]): dictionary containing the named parameter for the `cv2.SimpleBlobDetector`
      conf_file (bool): use the automatic configuration, provided in the `config` folder for the non-specified arguments

    Returns:
      Tuple[cv2.KeyPoint, np.ndarray]: Simple Blob keypoints and descriptors of the frame [**Note**, the Blob feature detector has no
      descriptor, thus None is returned. This is done for compatibility reasons with other feature extractors]
    """
    # load the frame as grayscale
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # set blob detector parameters
    blob_params = cv2.SimpleBlobDetector_Params()

    # load parameters of the blob detector
    blob_params = load_blob_params(
        blob_param_dict=blob_param_dict, config_file=config_file
    )

    # set up the simple blob detector
    blob_detector = cv2.SimpleBlobDetector_create(blob_params)

    # run the simple blob detector
    keypoints = blob_detector.detect(frame_gray)

    # return keypoints and not existing descriptors
    return keypoints, None
