from typing import Callable, List, Tuple, Union

import numpy as np
from tensorflow import Tensor

TensorOrNDArray = Union[Tensor, np.ndarray]
TupleOrInt = Union[Tuple, int]
ListOrInt = Union[List, int]
ThreeTensors = Tuple[Tensor, Tensor, Tensor]
ImageShape = Tuple[int, int, int]
StringOrCallable = Union[str, Callable]
FloatOrInt = Union[float, int]
