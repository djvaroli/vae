import numpy as np


def image_grid(images: np.ndarray, v_pad: int = 1, h_pad: int = 1,) -> np.ndarray:
    """Given an 4d array of images, returns a 3D grid of stacked images.

    Args:
        images (np.ndarray): [description]
        n_rows (int, optional): [description]. Defaults to 1.

    Returns:
        np.ndarray: [description]
    """
    if images.ndim != 4:
        raise Exception("Images must be a 4-dimensional array.")

    n_rows = 1
    n_images, img_h, img_w, n_channels = images.shape
    n_images_per_row = n_images

    n_cols = n_images // n_rows
    grid_shape = (
        n_rows * img_h + (n_rows > 1) * v_pad,
        n_cols * img_w + n_images_per_row * h_pad,
        n_channels,
    )
    grid = np.ones(grid_shape)

    y0 = 0
    y1 = y0 + img_h
    x0 = 0
    x1 = x0 + img_w
    for i_img in range(n_images):
        img = images[i_img, :, :, :]
        grid[y0:y1, x0:x1, :] = img
        x0 += img_w + h_pad
        x1 = x0 + img_w

    return grid
