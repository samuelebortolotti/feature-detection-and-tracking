""" Lucas-Kanade Optical Flow feature tracking module
"""
from argparse import _SubParsersAction as Subparser
from argparse import Namespace
import cv2
from fdt.detection import METHODS
from fdt.detection.blob import blob
from fdt.detection.harris import harris
from fdt.detection.orb import orb
from fdt.detection.sift import sift
from fdt.detection.utils import draw_lk_keypoints
import imutils
import numpy as np
import os


def configure_subparsers(subparsers: Subparser) -> None:
    """Configure a new subparser for running the Lucas Kanade Optical Flow

    Args:
      subparser (Subparser): argument parser
    """

    """
    Subparser parameters
    Args:
      method (str): feature detector, default [default: sift]
      nfeatures (int): number of features for the feature detector [default: 100]
      video (str): video file path if you are willing to run the algorithm on a video
      frame-update (int): after how many frame to recalibrate the features [default=50]
      output_video_name (str): output video name[if this is specified, the video will be saved in the `output folder`]
    """

    parser = subparsers.add_parser("lucas-kanade", help="Lucas Kanade feature tracker")
    parser.add_argument(
        "method",
        type=str,
        default="sift",
        choices=METHODS,
        help="Which feature detector to employ",
    )
    parser.add_argument(
        "--n-features",
        "-NF",
        type=int,
        default=100,
        help="Number of features to retain",
    )
    parser.add_argument(
        "--video",
        "-V",
        type=str,
        help="Video on which to run the Lucas Kanade optical flow tracker",
    )
    parser.add_argument(
        "--frame-update",
        "-FU",
        type=int,
        default=50,
        help="After how many frames to recalibrate",
    )
    parser.add_argument("--output-video-name", "-O", type=str, help="Output video name")
    # set the main function to run when Kalman is called from the command line
    parser.set_defaults(func=main)


def main(args: Namespace) -> None:
    """Checks the command line arguments and then runs Lucas-Kanade Optical Flow

    Args:
      args (Namespace): command line arguments
    """
    print("\n### Lucas Kanade optical flow feature tracker ###")
    print("> Parameters:")
    for p, v in zip(args.__dict__.keys(), args.__dict__.values()):
        print("\t{}: {}".format(p, v))
    print("\n")

    assert args.video is None or os.path.exists(
        args.video
    ), "Video passed does not exist"

    # call the lucas kanade optical flow
    lucas_kanade(
        camera_index=args.camera,
        video=args.video,
        method=args.method,
        n_features=args.n_features,
        update_every_n_frame=args.frame_update,
        output_video_name=args.output_video_name,
    )


def lucas_kanade(
    camera_index: int,
    video: str,
    method: str,
    n_features: int,
    update_every_n_frame: int,
    output_video_name: str,
) -> None:
    """Apply the Lucas Kanade Optical flow to track the features in the scene

    Args:
      camera_index (int): camera index
      video (str): video path
      methods (str): methods to employ in order to track the features
      n_features (int): number of features for the feature detector
      update_every_n_frame (int): after how many frame to recalibrate the features
      output_video_name (str): file name of the video to produce, if None is passed, no video is produced
    """

    # video capture input
    cap_input = video
    if video is None:
        cap_input = camera_index

    # open the capture
    cap = cv2.VideoCapture(cap_input)

    # set the parameters according to the method selected
    # and load the parameters accordingly
    if method == "sift":
        # SIFT
        feature_extract = sift
        feature_extract_conf = {"n_features": n_features}
    elif method == "orb":
        # ORB
        feature_extract = orb
        feature_extract_conf = {"n_features": n_features}
    elif method == "harris":
        # Harris corner detector
        feature_extract = harris
        # load config from `config` file
        feature_extract_conf = {
            "block_size": None,
            "k_size": None,
            "k": None,
            "tresh": None,
            "config_file": True,
        }
    else:
        # Blob detector
        feature_extract = blob
        # load config from `config` file
        feature_extract_conf = {
            "blob_param_dict": {
                "filterByColor": None,
                "blobColor": None,
                "filterByArea": None,
                "minArea": None,
                "maxArea": None,
                "filterByCircularity": None,
                "minCircularity": None,
                "maxCircularity": None,
                "filterByConvexity": None,
                "minConvexity": None,
                "maxConvexity": None,
                "filterByInertia": None,
                "minInertiaRatio": None,
                "maxInertiaRatio": None,
                "minThreshold": None,
                "maxThreshold": None,
                "thresholdStep": None,
                "minDistBetweenBlobs": None,
                "minRepeatability": None,
            },
            "config_file": True,
        }

    # capture the first frame
    ret, ref_frame = cap.read()
    # assert an exception whenever the return code is zero
    assert ret, "Error in reading the first frame from Video Capture"

    # resize the frame
    ref_frame = imutils.resize(ref_frame, width=550)

    # whether it is a dry run or not
    dry = output_video_name is None
    # with dry I mean that no output video is produced
    if not dry:
        # get the capture data (fps, height, width)
        fps = cap.get(cv2.CAP_PROP_FPS)
        height, width, _ = ref_frame.shape
        output_video = cv2.VideoWriter(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                f"../../output/{output_video_name}.avi",  # video is in AVI format
            ),
            cv2.VideoWriter_fourcc(*"XVID"),  # the XVID codec are used.
            fps,
            (width, height),
        )

    # extract descriptors and keypoints of the current frame
    keypoints_to_track, _ = feature_extract(ref_frame, **feature_extract_conf)

    # frame_counter
    frame_counter = 0

    # previous frame
    prev_frame = None

    # processing the video
    while cap.isOpened():

        # read the frame
        ret, frame = cap.read()
        # exit the loop whenever the video ends
        if not ret:
            break

        # Since it may be too big, but keeping the aspect ratio the same
        frame = imutils.resize(frame, width=550)

        # whether to update the feature matched
        if frame_counter % update_every_n_frame == 0:
            # extract descriptors and keypoints of the current frame
            keypoints_to_track, _ = feature_extract(frame, **feature_extract_conf)
            # conver the keypoints in numpy.ndarray for the Lucas-Kanade optical flow
            keypoints_to_track = np.float32(
                [key_point.pt for key_point in keypoints_to_track]
            ).reshape(-1, 1, 2)
        else:
            # run Lucas-Kanade optical flow
            keypoints_to_track, _status, _err = cv2.calcOpticalFlowPyrLK(
                prev_frame, frame, keypoints_to_track, None
            )

        updated_frame = draw_lk_keypoints(
            frame=frame.copy(),
            lk_features=keypoints_to_track,
            radius=5,
            color=(0, 0, 255),
            thickness=2,
            line_type=cv2.FILLED,
        )

        # draw the Lucas-Kanade Optical Flow feature tracking
        cv2.imshow(
            f"Lucas-Kanade optical flow + {method} feature tracking", updated_frame
        )

        # write the frame to the video
        if not dry:
            output_video.write(updated_frame)

        # exit when q is pressed
        if cv2.waitKey(1) == ord("q"):
            break

        # save the previous frame
        prev_frame = frame.copy()

        # increase the frame counter
        frame_counter += 1

    # Close windows
    if not dry:
        output_video.release()
    cap.release()
    cv2.destroyAllWindows()
