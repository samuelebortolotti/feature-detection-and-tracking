""" Plotter utils
"""
import numpy as np
import matplotlib.pyplot as plt


def plot_image(image: np.ndarray, title: str) -> None:
    """Plot an image putting a title.

    Args:
      image (np.ndarray): image to display
      title (str): title of the image
    """
    plt.imshow(image)
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
