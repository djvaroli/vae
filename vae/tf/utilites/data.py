from typing import Callable
from typing import Tuple, Optional 


from tensorflow.keras.preprocessing.image import ImageDataGenerator, DirectoryIterator


_0to255_to_0to1 = lambda x: x / 255.
_0to1_to_0to255 = lambda x: x * 255  # still need to convert to an int after that


def get_directory_iterator(
    directory: str,
    target_shape: Tuple[int, int],
    image_generator: ImageDataGenerator = None,
    batch_size: int = 32,
    class_mode: Optional[str] = None,
    color_mode: str = "rgb",
    preprocessing_function: Callable = _0to255_to_0to1,
    **kwargs
) -> DirectoryIterator:
    """
    Creates a directory iterator from images in a specified directory
    Args:
        directory:
        target_shape:
        image_generator:
        batch_size:
        class_mode:
        color_mode:

    Returns:
        DirectoryIterator yielding a batch of images
    """
    if image_generator is None:
        image_generator = ImageDataGenerator(preprocessing_function=preprocessing_function)

    return DirectoryIterator(
        directory,
        image_generator,
        target_size=target_shape,
        color_mode=color_mode,
        class_mode=class_mode,
        batch_size=batch_size,
        **kwargs
    )
