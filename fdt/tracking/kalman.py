""" Kalman feature tracking module
"""
from argparse import _SubParsersAction as Subparser
from argparse import Namespace
import cv2
from fdt.detection import METHODS
from fdt.plotter import plot_image
import numpy as np
import os


def configure_subparsers(subparsers: Subparser) -> None:
    """Configure a new subparser for running the Kalman feature tracking

    Args:
      subparser (Subparser): argument parser
    """

    """
    Subparser parameters
    Args:
      method (str): feature detector
    """

    parser = subparsers.add_parser("kalman", help="Kalman feature tracker")
    parser.add_argument(
        "method",
        type=str,
        default="sift",
        choices=METHODS,
        help="Which feature detector to employ",
    )
    # set the main function to run when Kalman is called from the command line
    parser.set_defaults(func=main)


def main(args: Namespace) -> None:
    r"""Checks the command line arguments and then runs kalman

    Args:
      args (Namespace): command line arguments
    """

    print("\n### Kalman feature tracker ###")
    print("> Parameters:")
    for p, v in zip(args.__dict__.keys(), args.__dict__.values()):
        print("\t{}: {}".format(p, v))
    print("\n")

    # call the kalman algorithm
    kalman(camera_index=args.camera, method=args.method)


def kalman(camera_index: int, method: str) -> None:
    r"""Apply the Kalman filter to track the object in the scene

    Args:
      camera_index (int): camera index
      methods (str): methods to employ in order to track the features
    """

    # video capture
    cap = cv2.VideoCapture(camera_index)

    # processing the video
    while cap.isOpened():

        # read the frame
        ret, frame = cap.read()
        # assert an exception whenever the return code is zero
        assert ret, "Error in reading the frame from Video Capture"

        # show the frame
        cv2.imshow(f"Kalman + {method} object tracking", frame)

        # exit when q is pressed
        if cv2.waitKey(33) == ord("q"):
            break

    # Close windows
    cap.release()
    cv2.destroyAllWindows()
