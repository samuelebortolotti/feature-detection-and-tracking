""" Kalman feature tracking module
"""
from argparse import _SubParsersAction as Subparser
from argparse import Namespace
from ..config.kalman_config import current_conf
import cv2
from fdt.detection import METHODS
from fdt.detection.matcher import (
    match_features,
)
from fdt.detection.orb import orb
from fdt.detection.utils import draw_features_keypoints
from fdt.detection.sift import sift
import imutils
from typing import Optional, Tuple, Any
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
        A: np.ndarray[Any, np.dtype[np.float32]],
        w: np.ndarray[Any, np.dtype[np.float32]],
        H: np.ndarray[Any, np.dtype[np.float32]],
        v: np.ndarray[Any, np.dtype[np.float32]],
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
            H
            if H is not None
            else np.eye(measure_params, dynamic_params, dtype=np.float32)
        )
        # if None is passed, identity matrix is used in order to track the points using np.eye
        self.kalman.transitionMatrix = (
            A if A is not None else np.eye(dynamic_params, dtype=np.float32)
        )
        self.kalman.processNoiseCov = (
            w if w is not None else np.eye(dynamic_params, dtype=np.float32)
        )
        self.kalman.measurementNoiseCov = (
            v if v is not None else np.eye(measure_params, dtype=np.float32)
        )
        self.kalman.controlMatrix = B

    def predict_next_position(self, x: np.float32, y: np.float32) -> Tuple[int, int]:
        """Function which predicts the next position using the Kalman filter equation

        Args:
          x (np.ndarray): x coordinate of the current measurement
          y (np.ndarray): y coordinate of the current measurement

        Returns:
          Tuple[int, int]: next position prediction
        """
        # current measurement
        z = np.array([[x], [y]], dtype=np.float32)
        # Kalman update
        self.kalman.correct(z)
        # Kalman filter prediction # Kalman filter prediction
        prediction = self.kalman.predict()
        # return the predictions
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
      method (str): feature detector, default [default: sift]
      n-features (int): number of features for the feature detector [default: 100]
      video (str): video file path if you are willing to run the algorithm on a video
      flann (bool): whether to use the flann based matcher [default=False]
      matching-distance (int): matching distance for the matcher [default=150]
      frame-update (int): after how many frame to recalibrate the features [default=50]
      output-video-name (str): file name of the video to produce, if None is passed, no video is produced
    """

    parser = subparsers.add_parser("kalman", help="Kalman feature tracker")
    parser.add_argument(
        "method",
        type=str,
        default="sift",
        choices=METHODS,
        help="Which feature detector to employ",
    )
    parser.add_argument(
        "--n-features", "-NF", type=int, default=100, help="Number of features to retain"
    )
    parser.add_argument(
        "--video", "-V", type=str, help="Video on which to run the Kalman filter"
    )
    parser.add_argument(
        "--flann", "-F", action="store_true", help="Use the FLANN matcher"
    )
    parser.add_argument(
        "--matching-distance", "-MD", type=int, default=150, help="Matching distance"
    )
    parser.add_argument(
        "--frame-update",
        "-FU",
        type=int,
        default=50,
        help="After how many frames to recalibrate",
    )
    parser.add_argument(
        "--output-video-name", "-O", type=str, help="Output video name"
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

    assert args.video is None or os.path.exists(
        args.video
    ), "Video passed does not exist"

    # call the kalman algorithm
    kalman(
        camera_index=args.camera,
        video=args.video,
        method=args.method,
        n_features=args.n_features,
        matching_distance=args.matching_distance,
        update_every_n_frame=args.frame_update,
        flann=args.flann,
        output_video_name=args.output_video_name
    )


def kalman(
    camera_index: int,
    video: str,
    method: str,
    n_features: int,
    matching_distance: int,
    update_every_n_frame: int,
    flann: bool,
    output_video_name: str
) -> None:
    """Apply the Kalman filter to track the object in the scene

    Args:
      camera_index (int): camera index
      video (str): video path
      methods (str): methods to employ in order to track the features
      n_features (int): number of features for the feature detector
      matching_distance (int): matching distance for the matcher
      update_every_n_frame (int): after how many frame to recalibrate the features
      flann (bool): whether to use the flann based matcher
      output-video-name (str): file name of the video to produce, if None is passed, no video is produced
    """
    # video capture input
    cap_input = video
    if video is None:
        cap_input = camera_index

    # open the capture
    cap = cv2.VideoCapture(cap_input)

    # set the parameters according to the method selected
    if method == "sift":
        # SIFT
        feature_extract = sift
        # the brute force matcher will employ the L2 NORM to find
        feature_matching_mode = cv2.NORM_L2
        # mode for the FLANN matcher
        FLANN_INDEX_KDTREE = 1
        # feature parameters for FLANN
        feature_index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    else:
        # ORB
        feature_extract = orb
        # the brute force matcher will employ the HAMMING NORM to find
        feature_matching_mode = cv2.NORM_HAMMING
        # FLANN index LSH
        FLANN_INDEX_LSH = 6
        # feature parameters for the FLANN matcher
        feature_index_params = dict(
            algorithm=FLANN_INDEX_LSH,
            table_number=6,  # 12
            key_size=12,  # 20
            multi_probe_level=1,
        )  # 2

    if not flann:
        # Brute force point matcher, thanks to the crossCheck only the consistent pairs are returned
        matcher = cv2.BFMatcher(feature_matching_mode, crossCheck=True)
    else:
        search_params = dict(checks=100)
        # Flann based matcher
        matcher = cv2.FlannBasedMatcher(feature_index_params, search_params)

    # capture the first frame
    ret, ref_frame = cap.read()
    # assert an exception whenever the return code is zero
    assert ret, "Error in reading the first frame from Video Capture"

    # resize the frame
    ref_frame = imutils.resize(ref_frame, width=550)

    # whether it is a dry run or not
    dry = output_video_name is None
    if not dry:
        # get the capture data
        fps = cap.get(cv2.CAP_PROP_FPS)
        height, width, _ = ref_frame.shape
        output_video = cv2.VideoWriter(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), f"../../output/{output_video_name}.avi"),
            cv2.VideoWriter_fourcc(*'XVID'),
            fps,
            (height, width)
        )

    # extract descriptors and keypoints of the current frame
    _, descriptors_to_match = feature_extract(ref_frame, n_features)

    # keypoints extracted using the kalman filter
    kalman_keypoints = []

    # Instantiate the Kalman Filter
    kalman = KalmanTracker(
        dynamic_params=current_conf["dynamic_params"],
        measure_params=current_conf["measure_params"],
        control_params=current_conf["control_params"],
        A=current_conf["A"],
        w=current_conf["w"],
        H=current_conf["H"],
        v=current_conf["v"],
        B=current_conf["B"],
    )

    # frame_counter
    frame_counter = 0

    # processing the video
    while cap.isOpened():

        # read the frame
        ret, frame = cap.read()
        # assert an exception whenever the return code is zero
        if not ret:
            break

        # Since it may be too big, but keeping the aspect ratio the same
        frame = imutils.resize(frame, width=550)

        # whether to update the feature matched
        if frame_counter % update_every_n_frame == 0:
            # extract descriptors and keypoints of the current frame
            _, descriptors_to_match = feature_extract(frame, n_features)

        # extract descriptors and keypoints of the current frame
        keypoints, descriptors = feature_extract(frame, n_features)

        # matching points
        matching_points, _ = match_features(
            matcher=matcher,
            first_descriptors=descriptors_to_match,
            second_descriptors=descriptors,
            second_keypoints=keypoints,
            max_matching_distance=matching_distance,
        )

        # update the kalman filter based on the measurements
        for x, y in matching_points:
            # the matching is only used in order to update the kalman matrices
            new_x, new_y = kalman.predict_next_position(x, y)
            # new predicted points
            kalman_keypoints.append(cv2.KeyPoint(new_x, new_y, 1))

        # show the matches frame
        updated_frame = draw_features_keypoints(image=frame, keypoints=kalman_keypoints)
        cv2.imshow(
            f"Kalman + {method} feature tracking",
            updated_frame
        )

        if not dry:
            output_video.write(updated_frame)

        # exit when q is pressed
        if cv2.waitKey(1) == ord("q"):
            break

        # reset the kalman keypoints
        kalman_keypoints.clear()

        # increase the frame counter
        frame_counter += 1

    # Close windows
    if not dry:
        output_video.release()
    cap.release()
    cv2.destroyAllWindows()
