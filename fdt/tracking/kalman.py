""" Kalman feature tracking module
"""
from argparse import _SubParsersAction as Subparser
from argparse import Namespace
import cv2
from fdt.detection import METHODS
from fdt.plotter import plot_image
from typing import Optional, Tuple
import numpy as np
import os


class KalmanTracker:
    """Wrapper of the open-cv Kalman filter.

    It is employed in order to track the movement of the objects.
    """

    def __init__(
        self,
        dynamic_params: int,
        measure_params: int,
        control_params: int,
        A: np.ndarray,
        w: np.ndarray,
        H: np.ndarray,
        v: np.ndarray,
        B: Optional[np.ndarray],
    ) -> None:
        """Creates a Kalman Filter wrapper.

        Args:
          A (np.ndarray): state transition matrix
          w (np.ndarray): process noise
          H (np.ndarray): measurement matrix
          v (np.ndarray): measurement noise
          B (np.ndarray): additional and optional control input
        """
        # kalman
        self.kalman = cv2.KalmanFilter(dynamic_params, measure_params, control_params)

        # set the kalman parameters
        self.kalman.measurementMatrix = (
            H if H is not None else np.eye(measure_params, dynamic_params)
        )
        # if None is passed, identity matrix is used in order to track the points using np.eye
        self.kalman.transitionMatrix = A if A is not None else np.eye(measure_params)
        self.kalman.processNoiseCov = w if w is not None else np.eye(measure_params)
        self.kalman.measurementNoiseCov = v if v is not None else np.eye(dynamic_params)
        self.kalman.controlMatrix = B

    def predict_next_position(self, x: np.ndarray, y: np.ndarray) -> Tuple[int, int]:
        """Function which predicts the next position using the Kalman filter equation

        Args:
          x (np.ndarray): x coordinate of the current measurement
          y (np.ndarray): y coordinate of the current measurement

        Returns:
          Tuple[int, int]: next position prediction
        """
        # current measurement
        z = np.array([np.float32(x)], [np.float32(y)])
        # Kalman update
        self.kalman.correct(z)
        # Kalman filter prediction # Kalman filter prediction
        prediction = self.kalman.predict()
        pred_x, pred_y = int(prediction[0]), int(prediction[1])
        return pred_x, pred_y


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
    """Checks the command line arguments and then runs kalman

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
    """Apply the Kalman filter to track the object in the scene

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
        if cv2.waitKey(1) == ord("q"):
            break

    # Close windows
    cap.release()
    cv2.destroyAllWindows()
