from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from ..data import FloatOrInt


def image_grid(
    images: np.ndarray,
    n_rows: int = 1,
    v_pad: int = 1,
    h_pad: int = 1,
    clip_range: Optional[Tuple[FloatOrInt, FloatOrInt]] = None,
    dtype=None,
) -> np.ndarray:
    """Given an 4d array of images, returns a 3D grid of stacked images.

    Args:
        images (np.ndarray): 4D array of images of shape (batch, h, w, c)
        n_rows (int, optional): Number of rows in the output grid. Defaults to 1.
        clip_range: Clip values in output grid to be within a given range
        dtype: Datatype to cast grid into

    Returns:
        np.ndarray: A 3D array grid of stacked images with shape
        (n_rows, batch // n_rows, c)
    """

    if images.ndim != 4:
        raise Exception("Images must be a 4-dimensional array.")

    n_images, img_h, img_w, n_channels = images.shape
    n_cols = n_images // n_rows

    grid_shape = (
        n_rows * img_h + (n_rows + 1) * v_pad,
        n_cols * img_w + (n_cols + 1) * h_pad,
        n_channels,
    )
    grid = np.ones(grid_shape)

    for i_row in range(n_rows):
        x0 = h_pad
        x1 = x0 + img_w

        y0 = i_row * img_h + (i_row + 1) * v_pad
        y1 = y0 + img_h
        for i_col in range(n_cols):
            img_index = i_row * n_cols + i_col
            img = images[img_index, :, :, :]
            grid[y0:y1, x0:x1, :] = img
            x0 += img_w + h_pad
            x1 = x0 + img_w

    if clip_range is not None:
        grid = np.clip(grid, clip_range[0], clip_range[1])

    if dtype is not None:
        grid = grid.astype(dtype)

    return grid


def image_grid_plot(
    images: np.ndarray,
    n_rows: int = 1,
    v_pad: int = 1,
    h_pad: int = 1,
    clip_range: Optional[Tuple[FloatOrInt, FloatOrInt]] = None,
    dtype=None,
    figsize: Tuple[int, int] = (20, 16),
    show_axes: bool = False
):
    """Given an 4d array of images, plots them as a grid.

    Args:
        images (np.ndarray): 4D array of images of shape (batch, h, w, c)
        n_rows (int, optional): Number of rows in the output grid. Defaults to 1.
        clip_range: Clip values in output grid to be within a given range
        dtype: Datatype to cast grid into

    Returns:
        np.ndarray: A plot of the grid of stacked images with shape
        (n_rows, batch // n_rows, c)
    """

    grid = image_grid(images, n_rows, v_pad, h_pad, clip_range, dtype)
    if grid.shape[-1] == 1:  # if only one channel
        grid = grid[:, :, 0]

    figure = plt.figure(figsize=figsize)
    if show_axes is False:
        ax = plt.Axes(figure, [0., 0., 1., 1.])
        ax.set_axis_off()
        figure.add_axes(ax)

    return figure
