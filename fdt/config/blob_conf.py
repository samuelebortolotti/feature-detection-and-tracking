# Current Simple Blob extractor configuration
current_conf = {
    # images are converted to many binary b/w layers. Then 0 searches for dark blobs, 255 searches for bright blobs. Or you set the filter to "false", then it finds bright and dark blobs, both.
    "filterByColor": False,
    "blobColor": 0,
    # Extracted blobs have an area between minArea (inclusive) and maxArea (exclusive).
    "filterByArea": True,
    "minArea": 3.0,  # Highly depending on image resolution and dice size
    "maxArea": 400.0,  # float! Highly depending on image resolution.
    "filterByCircularity": True,
    "minCircularity": 0.0,  # 0.7 could be rectangular, too. 1 is round. Not set because the dots are not always round when they are damaged, for example.
    "maxCircularity": 3.4028234663852886e38,  # infinity.
    "filterByConvexity": False,
    "minConvexity": 0.0,
    "maxConvexity": 3.4028234663852886e38,
    "filterByInertia": True,  # a second way to find round blobs.
    "minInertiaRatio": 0.55,  # 1 is round, 0 is anywhat
    "maxInertiaRatio": 3.4028234663852886e38,  # infinity again
    "minThreshold": 0,  # from where to start filtering the image
    "maxThreshold": 255.0,  # where to end filtering the image
    "thresholdStep": 5,  # steps to go through
    "minDistBetweenBlobs": 3.0,  # avoid overlapping blobs. must be bigger than 0. Highly depending on image resolution!
    "minRepeatability": 2,  # if the same blob center is found at different threshold values (within a minDistBetweenBlobs), then it (basically) increases a counter for that blob. if the counter for each blob is >: minRepeatability, then it's a stable blob, and produces a KeyPoint, otherwise the blob is discarded.
}
