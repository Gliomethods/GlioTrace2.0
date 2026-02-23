from matplotlib.widgets import Button
from pathlib import Path
from scipy import ndimage
from skimage.registration import phase_cross_correlation
from matplotlib.patches import Rectangle

import numpy as np
import tifffile
import pandas as pd
import matplotlib.pyplot as plt
import re


def to_uint8(a):
    """
    Deterministic conversion to uint8 (most common formats).

    Accepted:
      - uint8: returned unchanged
      - uint16: a >> 8
      - uint32: a >> 24
      - uint64: a >> 56
      - floats in [0,1]: scaled by 255 then cast
      - floats in [0,255]: clipped then cast

    Rejected (ValueError):
      - Other dtypes than listed above

    @ Author: André

    """
    a = np.asarray(a)

    # Unsigned integer:
    if a.dtype == np.uint8:
        return a
    if a.dtype == np.uint16:
        return (a >> 8).astype(np.uint8)
    if a.dtype == np.uint32:
        return (a >> 24).astype(np.uint8)
    if a.dtype == np.uint64:
        return (a >> 56).astype(np.uint8)

    # Float case
    if np.issubdtype(a.dtype, np.floating):
        vals = a.astype(np.float32, copy=False)

        vmin = float(vals.min())
        vmax = float(vals.max())

        eps = 1e-6

        if vmin >= -eps and vmax <= 1.0 + eps:
            return (np.clip(vals, 0.0, 1.0) * 255.0).astype(np.uint8)

        if vmin >= -eps and vmax <= 255.0 + eps:
            return np.clip(vals, 0.0, 255.0).astype(np.uint8)

        raise ValueError(
            f"to_uint8: float image range not supported: min={vmin:g}, max={vmax:g}. "
            "Expected floats in [0,1] or [0,255], or provide uint8/uint16/uint32/uint64."
        )

    raise ValueError(f"to_uint8: dtype {a.dtype} not supported.")


def corr2(a, b):
    """
    Correlation coefficient between two 2D arrays.

    @ Author: André
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    a_flat = a.ravel()
    b_flat = b.ravel()

    # Subtract mean
    a_mean = a_flat - a_flat.mean()
    b_mean = b_flat - b_flat.mean()

    num = np.sum(a_mean * b_mean)
    den = np.sqrt(np.sum(a_mean**2) * np.sum(b_mean**2))

    if den == 0:
        return 0.0

    return num / den


def register_stack_DT(D1, D2, D3, upsample_factor=100):
    """
    Inputs:
        D1, D2, D3: 3D numpy arrays (H, W, T)
    Returns:
        D1_reg, D2_reg, D3_reg, delta
        where delta is an array of shape (T, 2) with [dx, dy] per frame

    @ Author: André
    """

    # Copy to avoid modifying inputs in-place
    D1 = np.array(D1, copy=True)
    D2 = np.array(D2, copy=True)
    D3 = np.array(D3, copy=True)

    if D1.shape != D2.shape or D1.shape != D3.shape:
        raise ValueError("D1, D2, D3 must have the same shape")

    H, W, T = D1.shape

    delta = np.zeros(
        (T, 2), dtype=float
    )  # [dx, dy] for each frame, frame 0 stays [0, 0]

    for i in range(1, T):
        correlation1 = corr2(D1[:, :, i - 1], D1[:, :, i])

        imgA = D1[:, :, i - 1]
        imgB = D1[:, :, i]
        imgX = D2[:, :, i - 1]
        imgY = D2[:, :, i]
        imgM = D3[:, :, i - 1]
        imgN = D3[:, :, i]

        # We approximate with phase_cross_correlation on the summed images.
        ref = imgA + imgX + imgM
        mov = imgB + imgY + imgN

        shift, error, diffphase = phase_cross_correlation(
            ref, mov, upsample_factor=upsample_factor
        )
        # phase_cross_correlation returns shift as (shift_y, shift_x)
        dy, dx = shift  # rows, cols

        delta[i, :] = [dx, dy]

        # scipy.ndimage.shift uses shift=(shift_y, shift_x) = (dy, dx)
        D1[:, :, i] = ndimage.shift(
            D1[:, :, i], shift=(dy, dx), order=3, mode="constant", cval=0.0
        )
        D2[:, :, i] = ndimage.shift(
            D2[:, :, i], shift=(dy, dx), order=3, mode="constant", cval=0.0
        )
        D3[:, :, i] = ndimage.shift(
            D3[:, :, i], shift=(dy, dx), order=3, mode="constant", cval=0.0
        )

        correlation2 = corr2(D1[:, :, i - 1], D1[:, :, i])

        if (i + 1) % 5 == 0:
            print(
                f"frame: {i+1} before: {correlation1:.6f} after: {correlation2:.6f}")

    return D1, D2, D3, delta


def _find_tif_groups(file_path: Path):
    tif_files = list(file_path.rglob("*.tif"))
    if not tif_files:
        raise FileNotFoundError(f"No .tif files found under {file_path}")

    in_subfolder = any(f.parent != file_path for f in tif_files)
    groups = {}
    if in_subfolder:
        for f in tif_files:
            groups.setdefault(f.parent, []).append(f)
    else:
        groups[file_path] = tif_files

    for k in groups:
        groups[k] = sorted(groups[k])

    return groups


class ROISelector:
    """
    Multiple fixed-size square ROIs (fast, blitted).

    - Click a ROI to make it active (highlight + status).
    - Left-click & drag inside the active ROI to move it.
    - "Add ROI" button adds a new ROI (numbered) and makes it active.
    - "Remove ROI" deletes the active ROI.
    - "Confirm" button finishes (closes window). Final ROIs are in self.rois.

    Keys:
      - a : add ROI
      - d / delete / backspace : remove active ROI
      - enter / q / esc : confirm

    @ Author: André
    """

    def __init__(self, image, region_size):
        self.image = to_uint8(image)
        self.region_size = int(region_size)

        h, w = self.image.shape[:2]
        self.h, self.w = h, w

        self.rois = []        # [{"X":x,"Y":y,"W":s,"H":s}, ...]
        self.rects = []       # Rectangle patches (animated)
        self.labels = []      # Number labels (animated)
        self.active_idx = None

        self.dragging = False
        self._drag_offset_x = 0.0
        self._drag_offset_y = 0.0

        self.background = None
        self._capturing_bg = False

        # --- Figure / axes ---
        self.fig, self.ax = plt.subplots()
        self.ax.imshow(self.image, origin="upper", interpolation="nearest")
        self.ax.set_xlim(0, w)
        self.ax.set_ylim(h, 0)
        self.ax.set_title(
            "Click ROI to activate, drag to move. Add / Remove / Confirm.")

        self.status = self.ax.text(
            0.01, 0.99, "Active ROI: none",
            transform=self.ax.transAxes,
            va="top", ha="left",
            fontsize=10, color="white",
            bbox=dict(boxstyle="round,pad=0.25",
                      fc="black", ec="none", alpha=0.6),
            animated=True,
            zorder=50,
        )

        # --- Buttons ---
        plt.subplots_adjust(bottom=0.15)
        ax_add = self.fig.add_axes([0.05, 0.03, 0.25, 0.08])
        ax_del = self.fig.add_axes([0.375, 0.03, 0.25, 0.08])
        ax_ok = self.fig.add_axes([0.70, 0.03, 0.25, 0.08])

        self.btn_add = Button(ax_add, "Add ROI")
        self.btn_del = Button(ax_del, "Remove ROI")
        self.btn_ok = Button(ax_ok, "Confirm")

        self.btn_add.on_clicked(self._on_add_button)
        self.btn_del.on_clicked(self._on_remove_button)
        self.btn_ok.on_clicked(self._on_confirm_button)

        # --- Events ---
        self.fig.canvas.mpl_connect("draw_event", self._on_draw)
        self.fig.canvas.mpl_connect("button_press_event", self._on_press)
        self.fig.canvas.mpl_connect("button_release_event", self._on_release)
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

        # Start with one ROI
        self._add_new_roi(refresh=False)

        self._init_blit()
        plt.show(block=True)

    # ---------- toolbar helper ----------

    def _toolbar_mode(self):
        canvas = self.fig.canvas
        toolbar = getattr(canvas, "toolbar", None)
        if toolbar is None:
            manager = getattr(canvas, "manager", None)
            if manager is not None:
                toolbar = getattr(manager, "toolbar", None)
        return getattr(toolbar, "mode", "") if toolbar is not None else ""

    # ---------- ROI helpers ----------

    def _animated_artists(self):
        return [*self.rects, *self.labels, self.status]

    def _roi_center_of_view(self):
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        return (xlim[0] + xlim[1]) / 2.0, (ylim[0] + ylim[1]) / 2.0

    def _clamp_top_left(self, x1, y1):
        s = self.region_size
        x1 = max(0.0, min(float(x1), float(self.w - s)))
        y1 = max(0.0, min(float(y1), float(self.h - s)))
        return x1, y1

    def _update_label_pos(self, idx):
        r = self.rects[idx]
        self.labels[idx].set_position((r.get_x() + 3, r.get_y() + 12))

    def _sync_roi_dict_from_rect(self, idx):
        r = self.rects[idx]
        x = int(round(r.get_x()))
        y = int(round(r.get_y()))
        s = int(round(r.get_width()))
        self.rois[idx] = {"X": x, "Y": y, "W": s, "H": s}

    def _set_active(self, idx):
        self.active_idx = idx

        for i, r in enumerate(self.rects):
            if idx is not None and i == idx:
                r.set_linewidth(3.5)
                r.set_edgecolor("yellow")
            else:
                r.set_linewidth(2.0)
                r.set_edgecolor("red")

        self.status.set_text(
            "Active ROI: none" if idx is None else f"Active ROI: #{idx + 1}")

        if self.background is not None:
            self._blit_update()

    def _add_new_roi(self, refresh=True):
        cx, cy = self._roi_center_of_view()
        x1, y1 = cx - self.region_size / 2.0, cy - self.region_size / 2.0
        x1, y1 = self._clamp_top_left(x1, y1)

        rect = Rectangle(
            (x1, y1),
            self.region_size, self.region_size,
            fill=False, linewidth=2.0, edgecolor="red",
            animated=True, zorder=10,
        )
        self.ax.add_patch(rect)
        self.rects.append(rect)

        label = self.ax.text(
            x1 + 3, y1 + 12, f"{len(self.rects)}",
            color="white", fontsize=10, weight="bold",
            bbox=dict(boxstyle="round,pad=0.15",
                      fc="black", ec="none", alpha=0.6),
            animated=True, zorder=11,
        )
        self.labels.append(label)

        self.rois.append({"X": int(round(x1)), "Y": int(round(y1)),
                          "W": self.region_size, "H": self.region_size})

        self._set_active(len(self.rects) - 1)

        if refresh:
            self._init_blit()

    def _remove_active_roi(self):
        if self.active_idx is None:
            return
        i = self.active_idx

        # Stop any ongoing drag
        self.dragging = False

        # Remove artists from axes
        self.rects[i].remove()
        self.labels[i].remove()

        # Remove from lists
        del self.rects[i]
        del self.labels[i]
        del self.rois[i]

        # Renumber labels to match list order (1..N)
        for k, lab in enumerate(self.labels, start=1):
            lab.set_text(str(k))

        # Choose next active ROI
        if not self.rects:
            self._set_active(None)
        else:
            new_i = min(i, len(self.rects) - 1)
            self._set_active(new_i)

        # Artists changed -> rebuild background
        self._init_blit()

    # ---------- blitting ----------

    def _capture_clean_background(self):
        if self._capturing_bg:
            return
        self._capturing_bg = True
        try:
            canvas = self.fig.canvas
            artists = self._animated_artists()

            prev_vis = [a.get_visible() for a in artists]
            for a in artists:
                a.set_visible(False)

            canvas.draw()
            self.background = canvas.copy_from_bbox(self.ax.bbox)

            for a, v in zip(artists, prev_vis):
                a.set_visible(v)
        finally:
            self._capturing_bg = False

    def _init_blit(self):
        self._capture_clean_background()
        self._blit_update()

    def _on_draw(self, event):
        if self._capturing_bg:
            return
        canvas = self.fig.canvas
        self.background = canvas.copy_from_bbox(self.ax.bbox)
        self._blit_update()

    def _blit_update(self):
        if self.background is None:
            self._capture_clean_background()
            if self.background is None:
                return

        canvas = self.fig.canvas
        canvas.restore_region(self.background)
        for a in self._animated_artists():
            self.ax.draw_artist(a)
        canvas.blit(self.ax.bbox)

    # ---------- interaction ----------

    def _move_active_with_offset(self, x, y):
        if self.active_idx is None or x is None or y is None:
            return

        rect = self.rects[self.active_idx]
        x1 = x - self._drag_offset_x
        y1 = y - self._drag_offset_y
        x1, y1 = self._clamp_top_left(x1, y1)

        rect.set_xy((x1, y1))
        self._update_label_pos(self.active_idx)
        self._blit_update()

    def _on_press(self, event):
        if event.button != 1 or event.inaxes is not self.ax:
            return
        if self._toolbar_mode():
            return

        picked = None
        for i in range(len(self.rects) - 1, -1, -1):
            contains, _ = self.rects[i].contains(event)
            if contains:
                picked = i
                break
        if picked is None:
            return

        self._set_active(picked)

        self.dragging = True
        x0, y0 = self.rects[picked].get_xy()
        self._drag_offset_x = event.xdata - x0
        self._drag_offset_y = event.ydata - y0

    def _on_motion(self, event):
        if not self.dragging or event.inaxes is not self.ax:
            return
        if self._toolbar_mode():
            return
        self._move_active_with_offset(event.xdata, event.ydata)

    def _on_release(self, event):
        if event.button != 1:
            return
        self.dragging = False
        if self.active_idx is not None:
            self._sync_roi_dict_from_rect(self.active_idx)

    def _on_key(self, event):
        k = event.key
        if k == "a":
            self._add_new_roi(refresh=True)
        elif k in ("d", "delete", "backspace"):
            self._remove_active_roi()
        elif k in ("enter", "return", "q", "escape"):
            self._finish()

    # ---------- buttons ----------

    def _on_add_button(self, _event):
        self._add_new_roi(refresh=True)

    def _on_remove_button(self, _event):
        self._remove_active_roi()

    def _on_confirm_button(self, _event):
        self._finish()

    def _finish(self):
        if self.active_idx is not None:
            self._sync_roi_dict_from_rect(self.active_idx)
        print("Confirmed ROIs:")
        for i, r in enumerate(self.rois, 1):
            print(f"  #{i}: {r}")
        plt.close(self.fig)


def _select_rois_manual(image, region_size):
    selector = ROISelector(image, region_size)
    return selector.rois


def select_stabilize(
    file_path,
    output_path,
    mode="full",  # "manual", "coords", or "full"
    region_size=400,  # used for manual only
    coordinate_file=None  # used for coords only
):
    """
    Stabilize images, includes ROI selection

    Modes:
      - "manual": interactively select ROIs per experiment
      - "coords": use coordinate_file with columns: exp, X, Y, W, H
      - "full":   no ROI selection, use full frame per experiment

    Saves:
      - one .mat per (experiment, ROI or full-frame) with a "stack" struct:
          stack.Tstack, stack.Vstack, stack.Bstack

    @ Author: André
    """

    file_path = Path(file_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    groups = _find_tif_groups(file_path)

    if mode == "coords":
        if coordinate_file is None:
            raise ValueError(
                "coordinate_file must be provided for mode='coords'")
        coordinate_file = Path(coordinate_file)
        if coordinate_file.suffix.lower() in [".xls", ".xlsx"]:
            coord_df = pd.read_excel(coordinate_file)
        else:
            coord_df = pd.read_csv(coordinate_file)
    else:
        coord_df = None

    stacktable = []

    for folder, tif_paths in groups.items():
        tif_paths = sorted(tif_paths)
        foldername = folder.name
        experiment_of_interest = foldername.replace(" ", "_")

        print(f"\n=== Experiment: {experiment_of_interest} ===")

        # Decide ROIs
        if mode == "full":
            # One ROI covering the whole frame
            sample = tifffile.imread(str(tif_paths[0]))

            if sample.ndim != 3:
                raise ValueError(
                    f"Expected 3 dim e.g. (H, W, C) , got {im.shape} at {tif_paths[0]}"
                )

            # Sort color channel to last
            smallest_axis = np.argmin(sample.shape)
            sample = np.moveaxis(sample, smallest_axis, -1)

            H, W, ch = sample.shape
            if ch != 2 and ch != 3:
                raise ValueError("Number of channels must be 2 or 3")

            rois = [{"X": 0, "Y": 0, "W": W, "H": H}]

        elif mode == "manual":
            # manual selection of ROI
            # Choose representative image
            mid_idx = int(len(tif_paths) / 2)
            sample = tifffile.imread(str(tif_paths[mid_idx]))

            if region_size < 64:
                raise ValueError(
                    f"Please choose a region size above 64"
                )

            print(sample.shape)
            # Require 2 or 3 channels
            if sample.ndim != 3:
                raise ValueError(
                    f"Expected 3 dim e.g (H, W, C), got {sample.shape} at {tif_paths[mid_idx]}"
                )

            # Sort color channel to last
            smallest_axis = np.argmin(sample.shape)
            sample = np.moveaxis(sample, smallest_axis, -1)

            H, W, ch = sample.shape
            min_side = np.min([H, W])
            if min_side < region_size:
                raise ValueError(
                    f"{region_size} is to large for image with size {H} x {W}"
                )

            print(ch)

            if ch != 2 and ch != 3:
                raise ValueError("Number of channels must be 2 or 3")

            # Add a fake last axis if the image is only 2 channels
            if ch == 2:
                blue = np.zeros((H, W, 1), dtype=sample.dtype)
                sample = np.concatenate([sample, blue], axis=-1)  # (H, W, 3)

            print(sample.shape)

            rois = _select_rois_manual(sample, region_size)

        elif mode == "coords":
            # Extracts experiment number
            m = re.search(r"^exp_(\d+)", experiment_of_interest)
            if m:
                num = int(m.group(1))

            rois_sub = coord_df[coord_df["exp"] == num]
            if rois_sub.empty:
                print(
                    f"No ROIs for {experiment_of_interest} in coordinate_file, skipping."
                )
                continue
            rois = [
                {
                    "X": int(row["X"]),
                    "Y": int(row["Y"]),
                    "W": int(row["W"]),
                    "H": int(row["H"]),
                }
                for _, row in rois_sub.iterrows()
            ]
        else:
            raise ValueError("mode must be 'manual', 'coords', or 'full'")

        # For each ROI or full-frame, build stacks, register, save
        for k, roi in enumerate(rois, start=1):
            x, y, w, h = roi["X"], roi["Y"], roi["W"], roi["H"]
            print(
                f"  ROI {k}: X={x}, Y={y}, W={w}, H={h}, frames={len(tif_paths)}")

            T_frames, V_frames, B_frames = [], [], []

            for t_idx, tif_path in enumerate(tif_paths, start=1):
                im = tifffile.imread(str(tif_path))

                if im.ndim != 3:
                    raise ValueError(
                        f"Expected 3 dim e.g (H, W, C), got {im.shape} at {tif_path}"
                    )

                # Move color channel to last
                smallest_axis = np.argmin(im.shape)
                im = np.moveaxis(im, smallest_axis, -1)

                H, W, ch = im.shape

                if ch != 2 and ch != 3:
                    raise ValueError("Number of channels must be 2 or 3")

                if ch == 2:
                    blue = np.zeros((H, W, 1), dtype=im.dtype)
                    im = np.concatenate([im, blue], axis=-1)  # (H, W, 3)

                crop = im[y: y + h, x: x + w, :]
                if crop.shape[0] != h or crop.shape[1] != w:
                    raise ValueError(
                        f"Crop out of bounds for ROI {roi} on {tif_path}")

                V_frames.append(crop[:, :, 0])  # red
                T_frames.append(crop[:, :, 1])  # green
                B_frames.append(crop[:, :, 2])  # blue

                if t_idx % 10 == 0 or t_idx == len(tif_paths):
                    print(f"    frame {t_idx}/{len(tif_paths)}")

            Tstack = np.stack(T_frames, axis=-1)
            Vstack = np.stack(V_frames, axis=-1)
            Bstack = np.stack(B_frames, axis=-1)

            Tstack_stab, Vstack_stab, Bstack_stab, delta = register_stack_DT(
                Tstack, Vstack, Bstack
            )

            Tu8 = to_uint8(Tstack_stab)
            Vu8 = to_uint8(Vstack_stab)
            Bu8 = to_uint8(Bstack_stab)

            stack = {
                "Tstack": Tu8,
                "Vstack": Vu8,
                "Bstack": Bu8,
                "delta": delta,
            }

            stackname = output_path / \
                f"{experiment_of_interest}_roi_{k}_stack.npz"

            np.savez_compressed(
                stackname,
                **stack,
            )

            stacktable.append(str(stackname))

    return stacktable
