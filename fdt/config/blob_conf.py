"""
Recall:
      filterByColor (bool): wether to consider a specific colour as feature [default = False], since it is set to "false", then it finds bright and dark blobs, both.
      blobColor (int): blob colour, basically images are converted to many binary b/w layers. Then 0 searches for dark blobs, 255 searches for bright blobs.
      filterByArea (bool): Extracted blobs have an area between minArea (inclusive) and maxArea (exclusive)
      minArea (float): min area of the blob to look for. Notice, this is highly depending on image resolution and dice size
      maxArea (float): maximum area of the blob to look for. Highly depending on image resolution.
      filterByCircularity (bool): whether to filter by the circluarity shape of the objects in the scene
      minCircularity (float): 0 is rectangular, 1 is round. Not set because the dots are not always round when they are damaged, for example
      maxCircularity (float): max circularity default, in this case it has been set to infinity somehow.
      filterByConvexity (bool): whether to filter by convexity
      minConvexity (float): min convexity
      maxConvexity (float): max convexity, once again this is basically infinity
      filterByInertia (bool): basically a second way to find round blobs
      minInertiaRatio (float): minimum ineria ratio, where 1 is round and 0 is basically everything
      maxInertiaRatio (float): maximum inertia ratio, basically it is infinity once again
      minThreshold (float): from where to start filtering the image
      maxThreshold (float): where to end filtering the image
      thresholdStep (int): steps to perform
      minDistBetweenBlobs (float): a distance used in order to avoid overlapping blobs, must be bigger than 0 for obvious reasons
      minRepeatability (float): if the same blob center is found at different threshold values (within a minDistBetweenBlobs), then it increases a counter for that blob.
      if the counter for each blob is >: minRepeatability, then it's a stable blob, and produces a KeyPoint, otherwise the blob is discarded
"""

# Current Simple Blob extractor configuration
current_conf = {
    "filterByColor": False,
    "blobColor": 0,
    "filterByArea": True,
    "minArea": 3.0,
    "maxArea": 500.0,
    "filterByCircularity": True,
    "minCircularity": 0.8,
    "maxCircularity": 3.4028234663852886e38,
    "filterByConvexity": True,
    "minConvexity": 1.0,
    "maxConvexity": 3.4028234663852886e38,
    "filterByInertia": True,
    "minInertiaRatio": 0.7,
    "maxInertiaRatio": 3.4028234663852886e38,
    "minThreshold": 0,
    "maxThreshold": 255.0,
    "thresholdStep": 5,
    "minDistBetweenBlobs": 8.0,
    "minRepeatability": 2,
}
