import numpy as np
import imageio.v2 as imageio
from scipy.ndimage import median_filter
from pathlib import Path
import cv2


# ---- Morphology definitions ----
LABEL_NAMES = [
    "branching",
    "diffuse",
    "crowded",
    "locomotion",
    "perivascular",
    "round",
]

COLORS_RGB = np.array(
    [
        [255,   0,   0],  # branching      (red)
        [190,   0, 255],  # diffuse        (purple)
        [255, 255, 255],  # junk           (white)
        [255, 140,   0],  # locomotion     (orange)
        [255,   0, 200],  # perivascular   (magenta)
        [70, 120, 255],  # round          (blue)
    ],
    dtype=np.uint8,
)


def make_legend_panel(height, counts, width=320, title="Morphology"):
    # Base design tuned for ~340x320; scale everything to fit requested size
    H0, W0 = 340, 320
    s = min(height / H0, width / W0)
    s = max(0.45, s)  # clamp so it doesn't get absurdly tiny/unreadable

    def S(x):  # scale pixels
        return int(round(x * s))

    # Dark background
    panel = np.full((height, width, 3), 18, dtype=np.uint8)

    total = int(counts.sum())
    total = max(total, 1)

    # Scaled fonts + thickness
    fs_title = 0.8 * s
    fs_meta = 0.6 * s
    fs_name = 0.58 * s
    fs_num = 0.52 * s

    th_title = max(1, S(2))
    th_text = max(1, S(1))

    # Title
    cv2.putText(
        panel, title, (S(16), S(34)),
        cv2.FONT_HERSHEY_SIMPLEX, fs_title,
        (240, 240, 240), th_title, cv2.LINE_AA
    )
    cv2.putText(
        panel, f"Total: {int(counts.sum())}", (S(16), S(60)),
        cv2.FONT_HERSHEY_SIMPLEX, fs_meta,
        (200, 200, 200), th_text, cv2.LINE_AA
    )

    # Layout (scaled + vertical auto-fit)
    y0 = S(95)
    top_margin = y0
    bottom_margin = S(20)

    n = len(LABEL_NAMES)
    avail = max(1, height - top_margin - bottom_margin)
    dy = min(S(40), max(1, avail // max(1, n)))  # ensure rows fit

    sw = max(8, S(22))  # swatch size (keep a sane minimum)

    # Bar geometry (scaled, but safe for small widths)
    bar_x0 = min(width - S(40), max(S(110), S(170)))
    bar_x1 = width - S(16)
    bar_w_max = max(10, bar_x1 - bar_x0)

    for i, name in enumerate(LABEL_NAMES):
        y = y0 + i * dy

        # Stop before clipping at the bottom
        if y + S(20) > height - S(6):
            break

        # Color swatch
        bgr = tuple(int(c) for c in COLORS_RGB[i][::-1])
        cv2.rectangle(panel, (S(16), y - S(14)),
                      (S(16) + sw, y + S(8)), bgr, -1)
        cv2.rectangle(panel, (S(16), y - S(14)),
                      (S(16) + sw, y + S(8)), (60, 60, 60), 1)

        c = int(counts[i])
        pct = c / total

        # Label + numbers
        cv2.putText(
            panel, name, (S(50), y),
            cv2.FONT_HERSHEY_SIMPLEX, fs_name,
            (235, 235, 235), th_text, cv2.LINE_AA
        )
        cv2.putText(
            panel, f"{c:4d}  {pct:>4.0%}", (S(50), y + S(20)),
            cv2.FONT_HERSHEY_SIMPLEX, fs_num,
            (190, 190, 190), th_text, cv2.LINE_AA
        )

        # Bar (background + filled)
        cv2.rectangle(panel, (bar_x0, y - S(10)),
                      (bar_x1, y + S(4)), (40, 40, 40), -1)
        fill = int(round(bar_w_max * pct))
        if fill > 0:
            cv2.rectangle(panel, (bar_x0, y - S(10)),
                          (bar_x0 + fill, y + S(4)), bgr, -1)

    return panel


def vis_tracking_morphology_from_rows(
    data,
    mystack,
    myvasc,
    out_path,
    fps=5,
    tail_len=5,
    min_track_len=2,
    max_out_width=3840,   # <- keeps ffmpeg happy (no 12k-wide frames)
    enforce_macroblock=True,
):
    """
    Side-by-side output:
      [ original (red+green) | green-only (no red) | legend ]
    with alpha-faded tails.

    Parameters
    ----------
    data : pandas.DataFrame with columns:
        ['time', 'trax', 'tray', 'exp', 'roi', 'cellID', 'state_label']
    mystack : ndarray (H, W, T) green channel stack (uint8 or uint16 ok)
    myvasc  : ndarray (H, W, T) red channel stack   (uint8 or uint16 ok)
    """

    def to_u8(frame):
        """Convert common microscopy dtypes to uint8 safely."""
        if frame.dtype == np.uint8:
            return frame
        if frame.dtype == np.uint16:
            return (frame >> 8).astype(np.uint8)  # 16-bit -> 8-bit
        return np.clip(frame, 0, 255).astype(np.uint8)

    def crop_to_16(img):
        h, w = img.shape[:2]
        return img[: h - (h % 16), : w - (w % 16)]

    def downscale_to_max_width(img, max_w):
        """Downscale only (never upscale) to keep ffmpeg/x264 from failing."""
        h, w = img.shape[:2]
        if max_w is None or w <= max_w:
            return img
        s = max_w / float(w)
        new_w = int(w * s)
        new_h = int(h * s)
        # make even dimensions (good for yuv420p)
        new_w = max(2, (new_w // 2) * 2)
        new_h = max(2, (new_h // 2) * 2)
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    out_path = Path(out_path)
    out_path.mkdir(parents=True, exist_ok=True)

    H, W, T = mystack.shape
    df = data.copy()

    df["track_key"] = df["cellID"].astype(str)
    exp_id = int(df["exp"].iloc[0])
    roi_id = int(df["roi"].iloc[0])

    # Drop short tracks (< min_track_len valid positions)
    valid_xy = df["trax"].notna() & df["tray"].notna()
    track_len = df[valid_xy].groupby("track_key").size()
    keep = track_len[track_len >= min_track_len].index
    df = df[df["track_key"].isin(keep)]

    by_t = {int(t): g for t, g in df.groupby("time")}
    by_track = {k: g.sort_values("time") for k, g in df.groupby("track_key")}

    filename = out_path / f"{exp_id}_roi_{roi_id}_morphology.mp4"

    writer = imageio.get_writer(
        str(filename),
        fps=5,
        codec="libx264",
        ffmpeg_params=["-crf", "18", "-pix_fmt", "yuv420p"],
    )

    for t in range(T):
        # ---- base images (BACK TO YOUR ORIGINAL LOOK, but dtype-safe) ----
        g_u8 = to_u8(mystack[:, :, t])
        r_u8 = to_u8(myvasc[:, :, t])

        g_med = median_filter(g_u8, size=3)

        # Original composite: R in red, G in green
        im_rgb = np.zeros((H, W, 3), dtype=np.uint8)
        im_rgb[:, :, 1] = g_med
        im_rgb[:, :, 0] = r_u8
        im_bgr_rgb = cv2.cvtColor(im_rgb, cv2.COLOR_RGB2BGR)

        # Green-only: grayscale version of green
        im_bgr_green = cv2.cvtColor(g_med, cv2.COLOR_GRAY2BGR)

        counts = np.zeros(6, dtype=int)

        # ---- draw alpha-faded tails onto BOTH panels ----
        for track_df in by_track.values():
            hist = track_df[
                (track_df["time"] >= t - tail_len) &
                (track_df["time"] <= t)
            ]
            if len(hist) < 2:
                continue

            xs = hist["trax"].to_numpy(dtype=float)
            ys = hist["tray"].to_numpy(dtype=float)
            ok = ~np.isnan(xs) & ~np.isnan(ys)
            xs, ys = xs[ok], ys[ok]
            if xs.size < 2:
                continue

            last = hist.dropna(subset=["state_label"]).tail(1)
            if len(last):
                li = int(np.clip(int(last["state_label"].iloc[0]) - 1, 0, 5))
                rcol, gcol, bcol = COLORS_RGB[li]
                col = (int(bcol), int(gcol), int(rcol))  # BGR
            else:
                col = (255, 0, 0)

            pts = np.stack([xs, ys], axis=1).astype(np.int32)

            for i in range(len(pts) - 1):
                age = (len(pts) - 2 - i)  # 0 = newest
                alpha = float(np.clip(1.0 - age / max(1, tail_len), 0.15, 1.0))
                thickness = max(1, int(round(2 * alpha)))

                for canvas in (im_bgr_rgb, im_bgr_green):
                    overlay = canvas.copy()
                    cv2.line(
                        overlay,
                        tuple(pts[i]),
                        tuple(pts[i + 1]),
                        col,
                        thickness,
                        cv2.LINE_AA,
                    )
                    cv2.addWeighted(overlay, alpha, canvas,
                                    1.0 - alpha, 0, dst=canvas)

        # ---- draw current markers onto BOTH panels ----
        if t in by_t:
            cur = by_t[t].dropna(subset=["state_label", "trax", "tray"])
            lab_idx = cur["state_label"].astype(int).to_numpy() - 1
            lab_idx = np.clip(lab_idx, 0, 5)
            counts[:] = np.bincount(lab_idx, minlength=6)

            xs = cur["trax"].to_numpy(np.int32)
            ys = cur["tray"].to_numpy(np.int32)

            for x, y, li in zip(xs, ys, lab_idx):
                rcol, gcol, bcol = COLORS_RGB[li]
                col = (int(bcol), int(gcol), int(rcol))
                for canvas in (im_bgr_rgb, im_bgr_green):
                    cv2.circle(canvas, (int(x), int(y)),
                               3, col, -1, cv2.LINE_AA)

        # ---- assemble output: [RGB | green-only | legend] ----
        im_rgb_out = cv2.cvtColor(im_bgr_rgb, cv2.COLOR_BGR2RGB)
        im_green_out = cv2.cvtColor(im_bgr_green, cv2.COLOR_BGR2RGB)

        legend = make_legend_panel(H, counts)
        legend_rgb = cv2.cvtColor(legend, cv2.COLOR_BGR2RGB)

        sep_w = 12
        # dark gray separator
        sep_rgb = np.full((H, sep_w, 3), 30, dtype=np.uint8)

        combined = np.hstack(
            [im_rgb_out, sep_rgb, im_green_out, sep_rgb, legend_rgb])

        combined = downscale_to_max_width(combined, max_out_width)

        if enforce_macroblock:
            combined = crop_to_16(combined)

        writer.append_data(combined)

    writer.close()
    return str(filename)
