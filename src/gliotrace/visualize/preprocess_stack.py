import numpy as np
from pathlib import Path
from typing import Tuple


def prepare_gbm_vasc_arrays(gbm_array: np.ndarray, vasc_array: np.ndarray, stack_path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Normalize GBM/VASC arrays to (H, W, T), validate uint8-castability, cast to uint8,
    and crop spatial dims to be divisible by 8.
    @ Author: André Lasses Armatowski
    """
    stack_path = Path(stack_path)

    # Make sure stack is of dimensions width x height x time (works also for 500 x 501 x time)
    t = int(np.argmax(np.abs(np.array(gbm_array.shape) - np.median(gbm_array.shape))))
    gbm_array = np.moveaxis(gbm_array, t, -1)
    vasc_array = np.moveaxis(vasc_array, t, -1)

    if not (np.all(np.isfinite(gbm_array)) and
            np.all((gbm_array >= 0) & (gbm_array <= 255)) and
            np.all(np.equal(gbm_array, np.round(gbm_array)))):
        raise ValueError(
            f"In {stack_path}: gbm images must be safely castable to uint8")

    gbm = gbm_array.astype(np.uint8)

    if not (np.all(np.isfinite(vasc_array)) and
            np.all((vasc_array >= 0) & (vasc_array <= 255)) and
            np.all(np.equal(vasc_array, np.round(vasc_array)))):
        raise ValueError(
            f"In {stack_path}: vasc images must be safely castable to uint8")

    vasc = vasc_array.astype(np.uint8)

    # Enforce divisibility by 8 in dimensions to avoid problems in vascular segmentation
    new_pixel_size = int(np.floor(gbm.shape[0] / 8) * 8)
    gbm = gbm[0:new_pixel_size, 0:new_pixel_size, :]
    vasc = vasc[0:new_pixel_size, 0:new_pixel_size, :]

    return gbm, vasc
