import numpy as np

# Configuration which I am modeling right now
# Trying to complexify the transition matrix A since
# the tracking is hard in the video I have shown.
new_custom_conf = {
    "dynamic_params": 6,
    "measure_params": 2,
    "control_params": 0,
    "A": np.array(
        [
            [1, 0, 1, 0, 0.5, 0],
            [0, 1, 0, 1, 0, 0.5],
            [0, 0, 1, 0, 1, 0],
            [0, 0, 0, 1, 0, 1],
            [0, 0, 1, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ],
        np.float32,
    ),
    "w": np.eye(6, dtype=np.float32) * 0.2,
    "H": np.eye(2, 6, dtype=np.float32),
    "v": np.eye(2, dtype=np.float32),
    "B": None,
}

# Coniguration we have used in class
custom_conf = {
    "dynamic_params": 4,
    "measure_params": 2,
    "control_params": 0,
    "A": np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32),
    "w": np.eye(4, dtype=np.float32) * 0.003,
    "H": np.eye(2, 4, dtype=np.float32),
    "v": np.eye(2, dtype=np.float32),
    "B": None,
}

# Configuration with all the matrix set to the identity matrix
base_conf = {
    "dynamic_params": 6,
    "measure_params": 2,
    "control_params": 0,
    "A": np.eye(6, dtype=np.float32),
    "w": np.eye(6, dtype=np.float32) * 0.003,
    "H": np.eye(2, 6, dtype=np.float32),
    "v": np.eye(2, dtype=np.float32),
    "B": None,
}
