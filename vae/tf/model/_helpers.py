from typing import Any, Callable, List


def make_list_if_not(item_or_list: Any) -> List[Any]:
    """
    Given an input returns a list. If input is a list returns the unmodified list, otherwise
    returns a single-item list containing the input element.

    Args:
        item_or_list:

    Returns:
        List
    """

    if not isinstance(item_or_list, list):
        item_or_list = [item_or_list]

    return item_or_list


def compute_conv_output_shape(
    filters: int, kernel_size: int, stride: int, padding: int = 0, n_layers: int = 1
) -> int:
    """
    Computes the output dimensions of a 2D convolutional layer
    Args:
        filters:
        kernel_size:
        stride:
        padding:
        n_layers:

    Returns:
        An integer representing the output dimension of a convolutional layer
    """
    single_layer_output_shape = lambda f, k, s, p: (f - k + 2 * p) / s

    for layer in range(n_layers):
        filters = single_layer_output_shape(filters, kernel_size, stride, padding)

    return int(filters)


def compute_conv_transpose_output_shape(
    filters: int, kernel_size: int, stride: int, padding: int = 0, n_layers: int = 1
) -> int:
    """
    Computes the output dimensions of a 2D convolutional layer
    Args:
        filters:
        kernel_size:
        stride:
        padding:
        n_layers:

    Returns:
        An integer representing the output dimension of a convolutional transpose layer
    """
    single_layer_output_shape = lambda f, k, s, p: s * f + k - 2 * p

    for layer in range(n_layers):
        filters = single_layer_output_shape(filters, kernel_size, stride, padding)

    return int(filters)
