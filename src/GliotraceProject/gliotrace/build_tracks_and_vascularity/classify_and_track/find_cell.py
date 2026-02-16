from scipy.ndimage import gaussian_laplace
from skimage.filters import threshold_otsu, gaussian
from skimage.morphology import disk, dilation, erosion
from skimage.measure import label, regionprops_table
from skimage.util import img_as_float

import matplotlib.pyplot as plt
import numpy as np


def extract_block_features_pair(im_a, im_b, coords, blocksize):
    """
    Python replacement of matlab function for extracting snapshots
    Python-Authors: André
    """
    h, w = im_a.shape
    half = blocksize // 2

    patches_a = []
    patches_b = []
    valid_coords = []

    for x, y in coords:
        x = int(x)
        y = int(y)

        x_min = x - half
        x_max = x + half + 1
        y_min = y - half
        y_max = y + half + 1

        if x_min < 0 or y_min < 0 or x_max > w or y_max > h:
            continue

        patch_a = im_a[y_min:y_max, x_min:x_max]
        patch_b = im_b[y_min:y_max, x_min:x_max]

        patches_a.append(patch_a.ravel())
        patches_b.append(patch_b.ravel())
        valid_coords.append((x, y))

    if len(patches_a) == 0:
        return (
            np.empty((0, blocksize * blocksize)),
            np.empty((0, blocksize * blocksize)),
            np.empty((0, 2)),
        )

    return (
        np.stack(patches_a, axis=0),
        np.stack(patches_b, axis=0),
        np.array(valid_coords),
    )


def macro_track2(
    mystack,
    vasc,
    sigmah,
    cutoff,
    blocksize,
    mode="normal",
    debug=True,
):
    """
    Detect cell bodies in the green channel of a brain-slice image stack and
    extract per-cell coordinates, intensities, and feature snapshots.

    Parameters
    ----------
    mystack : numpy.ndarray
        Green-channel image stack (height × width x frames).
    vasc : numpy.ndarray
        Red-channel image stack corresponding to the same frames.
    sigmah : float
        Standard deviation of the LoG (Laplacian of Gaussian) filter used for
        cell detection.
    cutoff : float
        Intensity threshold above which pixels are considered potential cells.
    blocksize : int
        Size of the square patch extracted around each detected cell for
        classification.
    mode : str
        Either 'normal' or 'sparse'. When set to 'sparse', alternative
        detection parameters are used for very faint or sparse cell populations.

    Returns
    -------
    cellsx : list of array-like
        Frame-wise x-coordinates of detected cells.
    cellsy : list of array-like
        Frame-wise y-coordinates of detected cells.
    intensity : list of array-like
        Pixel intensities at each detected (x, y) coordinate.
    FEAT : list of numpy.ndarray
        List of matrices where each entry corresponds to one frame and each row
        contains the compressed feature snapshot extracted from the green
        channel around a detected cell.
    VASC : list of numpy.ndarray
        Same structure as FEAT, but snapshots are extracted from the red channel.

    Notes
    -----
    In debug mode, each processed frame is visualized both as:
    - green channel alone
    - green + red combined

    Detected cells are annotated with white circles for inspection.

    @OG-Authors: Madeleine Skeppås, Sven Nelander
    @date: 10012025

    @Python-Authors: André
    """

    mystack_float = img_as_float(mystack)
    vasc_float = img_as_float(vasc)

    num_frames = mystack.shape[2]

    cellsx = [np.empty(0, dtype=int) for _ in range(num_frames)]
    cellsy = [np.empty(0, dtype=int) for _ in range(num_frames)]
    intensity = [np.empty(0, dtype=float) for _ in range(num_frames)]
    FEAT = [np.empty((0, blocksize * blocksize), dtype=float)
            for _ in range(num_frames)]
    VASC = [np.empty((0, blocksize * blocksize), dtype=float)
            for _ in range(num_frames)]

    # 3x3 neighborhood footprint without center pixel (for local max test)
    footprint = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=bool)

    for i in range(num_frames):

        im = mystack[:, :, i]
        im_vasc = vasc[:, :, i]

        im_f = mystack_float[:, :, i]
        im_vasc_f = vasc_float[:, :, i]

        xs = []
        ys = []

        # --- LoG-like filter step --------------------------------------------
        blob_im_1 = -gaussian_laplace(im_f, sigma=sigmah)

        # --- Local maxima for blob detection ---------------------------------
        dilated = dilation(blob_im_1, footprint=footprint)
        bw = blob_im_1 >= dilated

        # --- Cell body masks --------------------------------------------------
        # Note, cutoff needs to be retuned
        cellbodypixels_1 = dilation(
            erosion(blob_im_1 > cutoff, disk(3)), disk(3))
        thr_im = threshold_otsu(im_f)
        cellbodypixels_2 = im_f > thr_im
        cellbodypixels_3 = dilation(cellbodypixels_2, disk(3))

        # If cells are sparse: use cellbodypixels_1. Else: cellbodypixels_3
        if mode == "sparse":
            cellbodypixels = cellbodypixels_1
        else:
            cellbodypixels = cellbodypixels_3

        # --- Label connected components as cell bodies -----------------------
        L = label(cellbodypixels, connectivity=2)

        if L.max() > 0:
            regs = regionprops_table(L, properties=("label", "area"))
            areas = np.array(regs["area"])
            labels = np.array(regs["label"])

            # Ignore regions with area > 3000
            bad_labels = labels[areas > 3000]

            if bad_labels.size > 0:
                mask_bad = np.isin(L, bad_labels)
                L[mask_bad] = 0

        cellbodypixels = L

        # --- Bulk region mask via Gaussian smoothing -------------------------
        imsmooth = gaussian(im_f, sigma=30, preserve_range=True)

        thr_bulk = threshold_otsu(imsmooth)
        mask = imsmooth > thr_bulk

        # bulk present if max intensity > 200 in MATLAB (Here range is [0,1])
        bulk_present = imsmooth.max() > (200 / 255)

        # --- Iterate over each cell body object ------------------------------
        xs_list = []
        ys_list = []

        for k in range(1, cellbodypixels.max() + 1):
            px = cellbodypixels == k

            if not px.any():
                continue

            if bulk_present:
                coords_y, coords_x = np.where(
                    bw & (blob_im_1 > cutoff) & px & cellbodypixels_2 & (~mask)
                )
            else:
                coords_y, coords_x = np.where(
                    bw & (blob_im_1 > cutoff) & px & cellbodypixels_2
                )

            if coords_x.size > 1:
                xmean = int(np.round(coords_x.mean()))
                ymean = int(np.round(coords_y.mean()))
                area2 = px.sum()

                if cellbodypixels_3[ymean, xmean] and area2 > 200 and area2 < 1000:
                    xs_list.append(xmean)
                    ys_list.append(ymean)
                else:
                    xs_list.extend(coords_x.tolist())
                    ys_list.extend(coords_y.tolist())
            elif coords_x.size == 1:
                xs_list.append(int(coords_x[0]))
                ys_list.append(int(coords_y[0]))
            # else: no coordinates for this cell body

        xs = np.array(xs_list, dtype=int)
        ys = np.array(ys_list, dtype=int)

        cellsy[i] = ys
        cellsx[i] = xs

        # --- Extract features if any cells present ---------------------------
        if xs.size > 0:
            coords = np.stack([xs, ys], axis=1)  # (x, y)

            featA, featB, valid_coords = extract_block_features_pair(
                im, im_vasc, coords, blocksize  # gbm channel  # vascular channel
            )

            # If all points were too close to edges, valid_coords can be empty
            if valid_coords.shape[0] > 0:
                # Update coordinates to those actually used in feature extraction
                cellsx[i] = valid_coords[:, 0]
                cellsy[i] = valid_coords[:, 1]

                FEAT[i] = featA
                VASC[i] = featB

                iy = valid_coords[:, 1]
                ix = valid_coords[:, 0]
                intensity[i] = im[iy, ix]
            else:
                cellsx[i] = np.zeros(0, dtype=int)
                cellsy[i] = np.zeros(0, dtype=int)
                intensity[i] = np.zeros(0)
                FEAT[i] = np.zeros((0, blocksize * blocksize))
                VASC[i] = np.zeros((0, blocksize * blocksize))

        # --- Debug visualization ---------------------------------------------
        if debug and cellsx[i].size > 0:
            fig, axes = plt.subplots(1, 2, figsize=(8, 4))

            # Left: composite of vasc (R) and green (G)
            rgb = np.zeros((im.shape[0], im.shape[1], 3), dtype=float)
            rgb[..., 0] = img_as_float(im_vasc)
            rgb[..., 1] = img_as_float(im)
            axes[0].imshow(rgb)
            axes[0].scatter(
                cellsx[i], cellsy[i], marker="o", facecolors="none", edgecolors="w"
            )
            axes[0].set_title(f"Frame {i} - Composite")
            axes[0].axis("off")

            # Right: green channel only
            green_rgb = np.zeros_like(rgb)
            green_rgb[..., 1] = img_as_float(im)
            axes[1].imshow(green_rgb)
            axes[1].scatter(
                cellsx[i], cellsy[i], marker="o", facecolors="none", edgecolors="w"
            )
            axes[1].set_title(f"Frame {i} - Green only")
            axes[1].axis("off")

            plt.tight_layout()
            plt.show()

    # --- Check if all frames are empty --------------------------------------
    all_empty = True
    for x in cellsx:
        if x.size != 0:
            all_empty = False
            break

    return cellsx, cellsy, intensity, FEAT, VASC, all_empty
