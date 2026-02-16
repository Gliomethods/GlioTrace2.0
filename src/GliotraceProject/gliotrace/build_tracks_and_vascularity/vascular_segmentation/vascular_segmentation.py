import numpy as np
from pathlib import Path

import torch
import torch.nn.functional as F

from scipy.ndimage import median_filter
from skimage.morphology import closing, disk, skeletonize, dilation
import imageio.v2 as imageio

from skimage.measure import label, regionprops


def bwskel_min_branch_length(mask, min_length=15):
    """
    MATLAB-like bwskel(mask,'MinBranchLength',min_length)

    1. Skeletonize the mask
    2. Remove skeleton components shorter than min_length
    """
    # Step 1: skeletonize
    skel = skeletonize(mask > 0)

    # Step 2: prune small skeleton components
    lab = label(skel)
    out = np.zeros_like(skel, dtype=bool)

    for region in regionprops(lab):
        # number of skeleton pixels = branch length in pixels
        if region.area >= min_length:
            out[lab == region.label] = True

    return out


def segment_quantify_vasculature(Vstack, net):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    H, W, T = Vstack.shape
    vasc_length_stack = []
    segstack = np.zeros((H, W, 3, T), dtype=np.uint8)
    binarystack = np.zeros((H, W, T), dtype=bool)

    for i in range(T):
        # Median filtering (MATLAB medfilt2)
        im = Vstack[:, :, i]
        im = median_filter(im, size=3, mode="mirror")

        # ---- segmentation with PyTorch U-net ----
        # MATLAB: semanticseg(uint8(im), net_trained);
        im_uint8 = im.astype(np.uint8)

        # Build tensor [1, 1, H, W] – keep same scaling as MATLAB (0–255)
        x = (
            torch.from_numpy(im_uint8)
            .unsqueeze(0)
            .unsqueeze(0)
            .to(device, dtype=torch.float32)
        )

        with torch.no_grad():
            logits = net(x)  # [1, 2, H', W']
            probs = F.softmax(logits, dim=1)
            pred = probs.argmax(dim=1)[0].cpu(
            ).numpy().astype(np.uint8)  # [H', W']

        # Map to vessel mask
        # Assume class 0 = 'C1' (background), class 1 = vessel
        mask = pred != 0  # True where vessel

        # ---- morphological closing ----
        mask_closed = closing(mask, footprint=disk(3))

        # ---- skeletonization ----
        # We approximate with skimage.skeletonize; no branch-length filter here.
        skel = bwskel_min_branch_length(mask_closed, min_length=15)

        # Length of vasculature = number of skeleton pixels
        vasc_length_stack.append(int(skel.sum()))

        # ---- overlay construction ----
        imgray = im.astype(np.float32) / 100.0
        imgray = np.clip(imgray, 0.0, 1.0)
        imgray = 0.5 * (imgray - 0.5) + 0.5  # simple contrast shift
        imgray = np.clip(imgray, 0.0, 1.0)

        # Dilate skeleton for thicker overlay
        skel_dil = dilation(skel, footprint=disk(2))

        # Build RGB overlay
        rgbImage = np.zeros((H, W, 3), dtype=np.float32)

        # Colors: same as MATLAB (65,105,225) / 255 ~ blueish
        color = np.array([65, 105, 225], dtype=np.float32) / 255.0

        # Background: imgray, Vessel: color
        # rgb = imgray * (~skel_dil) + color * skel_dil
        bg_mask = ~skel_dil
        fg_mask = skel_dil

        for c in range(3):
            rgbImage[:, :, c] = imgray * bg_mask + color[c] * fg_mask

        rgb_uint8 = (np.clip(rgbImage, 0.0, 1.0) * 255).astype(np.uint8)
        segstack[:, :, :, i] = rgb_uint8
        binarystack[:, :, i] = fg_mask

    return vasc_length_stack, segstack, binarystack


def compute_vascular_segementation(vasc, net):
    """Compute vasculature-only statistics for a stack."""
    vasc_length, segstack, binarystack = segment_quantify_vasculature(
        vasc, net
    )
    return vasc_length, segstack, binarystack
