"""Main module of the `feature-detection-and-extraction` project

This project has been developed by Samuele Bortolotti as an assignment for the Computer Vision course
of the master's degree program in Computer Science at University of Trento
"""

import argparse
from argparse import Namespace
import fdt.detection as detectors
import fdt.tracking as trackers
import matplotlib


def get_args() -> Namespace:
    """Parse command line arguments.

    Returns:
      Namespace: command line arguments
    """

    # main parser
    parser = argparse.ArgumentParser(
        prog="Feature-detection-and-extraction",
        description="""
        Feature detection and extraction:
        Second assignment of the Computer Vision course at University of Trento
        """,
    )

    # subparsers: so as to have command line options for sub-functions
    subparsers = parser.add_subparsers(help="sub-commands help")
    # configure detector subparsers
    detectors.sift.configure_subparsers(subparsers)
    detectors.orb.configure_subparsers(subparsers)
    # configure trackers subparsers
    trackers.kalman.configure_subparsers(subparsers)

    # open cv camera
    parser.add_argument(
        "--camera", "-C", type=int, default=0, help="Camera index [default: 0]"
    )
    # matplotlib interactive backend
    parser.add_argument(
        "--matplotlib-backend",
        "-mb",
        choices=matplotlib.rcsetup.interactive_bk,
        default="QtAgg",
        help="Matplotlib interactive backend [default: QtAgg]",
    )

    # parse the command line arguments
    parsed_args = parser.parse_args()

    # if function not passed, then print the usage and exit the program
    if "func" not in parsed_args:
        parser.print_usage()
        parser.exit(1)

    return parsed_args


def main(args: Namespace) -> None:
    """Main function

    It runs the `func` function passed to the parser with the respective
    parameters

    Args:
      args (Namespace): command line arguments
    """
    # set matplotlib backend
    matplotlib.use(args.matplotlib_backend)
    # execute the function `func` with args as arguments
    args.func(
        args,
    )


if __name__ == "__main__":
    """
    Main
    Calls the main function with the command line arguments passed as parameters
    """
    main(get_args())
