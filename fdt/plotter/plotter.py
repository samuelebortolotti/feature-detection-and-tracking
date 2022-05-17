""" Plotter utils
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt


def plot_image(image: np.ndarray, title: str) -> None:
    """Plot an image putting a title.

    Args:
      image (np.ndarray): image to display
      title (str): title of the image
    """
    # > Note: cv2 threats the images as BGR by default, however, matplotlib assumes they are in RBG
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title(title)

    # Handles the case in which there are some issues with the Matplotlib interactive backend
    # it saves the image in a picture
    try:
        plt.show()
    except:
        print(
            "======= There were some problems rendering the image with the Matplotlib backend ======="
        )
        print("Saving the plot in a picture...")
        plt.savefig(f"{title}.png")
        print("Plot recovered")
