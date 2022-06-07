"""
Recall:
      block_size (int): the size of neighbourhood considered for corner detection
      k_size (int): aperture parameter of the Sobel derivative used
      k (float): Harris detector free parameter in the equation
      thresh (float): Harris detector good point threshold (the selected points are harris*thres)
      config_file (bool): use the automatic configuration, provided in the `config` folder for the non-specified arguments
"""
# Current Harris corner detector configuration
current_conf = {
    "block_size": 2,
    "k_size": 7,
    "k": 0.04,
    "thresh": 0.2,
}
