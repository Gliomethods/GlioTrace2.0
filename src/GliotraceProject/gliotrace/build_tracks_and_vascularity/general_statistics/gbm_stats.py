import numpy as np


def stackscore_naive_sad(gbm):
    """
    Input: gbm: stack images (single channel) (H, W, T)
    Return: sad: a scalar

    @OG-Authors: Sven Nelander
    @date: 14082024

    @Python-Authors: André
    """

    # Work on copy since we are modifying in place
    g = gbm.astype(np.float32, copy=True)

    # Remove noise
    g -= 20
    g[g < 0] = 0

    # Absolute difference between neighbouring slices
    abs_diff = np.abs(g[:, :, 1:] - g[:, :, :-1])

    # SAD = mean over all slices and pixels
    sad = np.mean(abs_diff)
    return sad


def stackscore_naive_admad(gbm):
    """
    Input: gbm: stack (single channel) (H, W, T)
    Output: admad

    @OG-Authors: Sven Nelander
    @date: 14082024

    @Python-Authors: André
    """

    # Absolute differences between neighbouring slices
    abs_diff = np.abs(gbm[:, :, 1:] - gbm[:, :, :-1])

    n_slices = abs_diff.shape[2]
    c = np.zeros(n_slices, dtype=np.float32)

    for j in range(n_slices):
        ad = abs_diff[:, :, j].copy()
        ad[ad < 20] = 0  # threshold small changes

        # Denominator uses ORIGINAL data (no noise removal)
        denom = np.mean(gbm[:, :, j: j + 2])

        c[j] = 0.0 if denom == 0 else np.mean(ad) / denom

    return c


def stackscore_naive_growth_rate(gbm, dt):
    """
    Input: gbm: stack of images (single channel) (H, W, T)
           dt: a scalar describing the change in  time
           between frames

    Output: growth_rate: a scalar describing increase in gbm.

    @OG-Authors: Sven Nelander
    @date: 14082024

    @Python-Authors: André
    """
    y = np.mean(gbm, axis=(0, 1))

    T = gbm.shape[2]
    t = np.arange(T) * dt

    slope, intercept = np.polyfit(t, y, 1)
    return slope


def compute_gbm_stats(gbm, dt):
    """
    Input: gbm: stack of images (single channel) (H, W, T)
           dt: a scalar describing the change in  time
           between frames

    Output: Summary statiscs of change in gbm
    """

    # Convert to float to avoid doing modular arithmetic
    gbm_float = gbm.astype(np.float32, copy=True)

    # Total green/GBM signal per frame
    sum_green = gbm_float.sum(axis=(0, 1))

    # Growth rate
    growth_rate = stackscore_naive_growth_rate(gbm_float, dt)

    # Simplified SAD
    sad_val = stackscore_naive_sad(gbm_float)

    # Admad
    c = stackscore_naive_admad(gbm_float)

    return growth_rate, sad_val, c, sum_green
