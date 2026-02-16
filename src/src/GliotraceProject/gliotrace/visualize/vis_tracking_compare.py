import numpy as np
import imageio.v2 as imageio
from scipy.ndimage import median_filter
from pathlib import Path
import cv2

# ---- Morphology definitions ----
LABEL_NAMES = [
    "branching",
    "diffuse",
    "junk",
    "locomotion",
    "perivascular",
    "round",
]

COLORS_RGB = np.array(
    [
        [0, 200, 255],  # branching      (cyan-ish)
        [190,   0, 255],  # diffuse        (purple)
        [255, 255, 255],  # junk           (white)
        [255, 140,   0],  # locomotion     (orange)
        [255,   0, 200],  # perivascular   (magenta)
        [70, 120, 255],  # round          (blue)
    ],
    dtype=np.uint8,
)


def make_legend_panel(height, counts, width=320, title="Morphology"):
    # Dark background
    panel = np.full((height, width, 3), 18, dtype=np.uint8)

    total = int(counts.sum())
    total = max(total, 1)

    # Title
    cv2.putText(
        panel, title, (16, 34),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8,
        (240, 240, 240), 2, cv2.LINE_AA
    )
    cv2.putText(
        panel, f"Total: {int(counts.sum())}", (16, 60),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
        (200, 200, 200), 1, cv2.LINE_AA
    )

    # Layout
    y0 = 95
    dy = 40
    sw = 22  # swatch size
    bar_x0 = 170
    bar_x1 = width - 16
    bar_w_max = max(10, bar_x1 - bar_x0)

    for i, name in enumerate(LABEL_NAMES):
        y = y0 + i * dy

        # Color swatch
        bgr = tuple(int(c) for c in COLORS_RGB[i][::-1])
        cv2.rectangle(panel, (16, y - 14), (16 + sw, y + 8), bgr, -1)
        cv2.rectangle(panel, (16, y - 14), (16 + sw, y + 8), (60, 60, 60), 1)

        c = int(counts[i])
        pct = c / total

        # Label + numbers
        cv2.putText(
            panel, name, (50, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.58,
            (235, 235, 235), 1, cv2.LINE_AA
        )
        cv2.putText(
            panel, f"{c:4d}  {pct:>4.0%}", (50, y + 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.52,
            (190, 190, 190), 1, cv2.LINE_AA
        )

        # Bar (background + filled)
        cv2.rectangle(panel, (bar_x0, y - 10),
                      (bar_x1, y + 4), (40, 40, 40), -1)
        fill = int(bar_w_max * pct)
        if fill > 0:
            cv2.rectangle(panel, (bar_x0, y - 10),
                          (bar_x0 + fill, y + 4), bgr, -1)

    return panel


def vis_tracking_morphology_compare_viterbi(
    data,
    mystack,
    myvasc,
    out_path,
    left_label_col="state_label",
    right_label_col="viterbi_state",
    fps=5,
    tail_len=5,
    min_track_len=4,
    max_out_width=3840,
    enforce_macroblock=True,
    title_left="CNN",
    title_right="Viterbi",
):
    def to_u8(frame):
        if frame.dtype == np.uint8:
            return frame
        if frame.dtype == np.uint16:
            return (frame >> 8).astype(np.uint8)
        return np.clip(frame, 0, 255).astype(np.uint8)

    def crop_to_16(img):
        h, w = img.shape[:2]
        return img[: h - (h % 16), : w - (w % 16)]

    def downscale_to_max_width(img, max_w):
        h, w = img.shape[:2]
        if max_w is None or w <= max_w:
            return img
        s = max_w / float(w)
        new_w = max(2, (int(w * s) // 2) * 2)
        new_h = max(2, (int(h * s) // 2) * 2)
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    out_path = Path(out_path)
    out_path.mkdir(parents=True, exist_ok=True)

    H, W, T = mystack.shape
    df = data.copy()

    df["track_key"] = df["cellID"].astype(str)
    exp_id = df["exp"].iloc[0]
    roi_id = df["roi"].iloc[0]

    # Drop short tracks
    valid_xy = df["trax"].notna() & df["tray"].notna()
    track_len = df[valid_xy].groupby("track_key").size()
    keep = track_len[track_len >= min_track_len].index
    df = df[df["track_key"].isin(keep)]

    by_t = {int(t): g for t, g in df.groupby("time")}
    by_track = {k: g.sort_values("time") for k, g in df.groupby("track_key")}

    filename = out_path / f"{exp_id}_roi_{roi_id}_compare_viterbi.mp4"
    writer = imageio.get_writer(
        str(filename),
        fps=fps,
        codec="libx264",
        ffmpeg_params=["-crf", "18", "-pix_fmt", "yuv420p"],
    )

    for t in range(T):
        g_u8 = to_u8(mystack[:, :, t])
        r_u8 = to_u8(myvasc[:, :, t])
        g_med = median_filter(g_u8, size=3)

        # Base composite image (same background for both panels)
        base_rgb = np.zeros((H, W, 3), dtype=np.uint8)
        base_rgb[:, :, 1] = g_med
        base_rgb[:, :, 0] = r_u8
        base_bgr = cv2.cvtColor(base_rgb, cv2.COLOR_RGB2BGR)

        # Two canvases: left (old), right (viterbi)
        canvas_L = base_bgr.copy()
        canvas_R = base_bgr.copy()

        counts_L = np.zeros(6, dtype=int)
        counts_R = np.zeros(6, dtype=int)

        # ---- draw alpha-faded tails onto BOTH, but color by different label cols ----
        for track_df in by_track.values():
            hist = track_df[(track_df["time"] >= t - tail_len)
                            & (track_df["time"] <= t)]
            if len(hist) < 2:
                continue

            xs = hist["trax"].to_numpy(dtype=float)
            ys = hist["tray"].to_numpy(dtype=float)
            ok = ~np.isnan(xs) & ~np.isnan(ys)
            xs, ys = xs[ok], ys[ok]
            if xs.size < 2:
                continue

            pts = np.stack([xs, ys], axis=1).astype(np.int32)

            def color_for_last(label_col):
                last = hist.dropna(subset=[label_col]).tail(1)
                if len(last) == 0:
                    return (255, 0, 0)  # BGR fallback
                li = int(last[label_col].iloc[0]) - 1  # 1..6 -> 0..5
                li = int(np.clip(li, 0, 5))
                rcol, gcol, bcol = COLORS_RGB[li]
                return (int(bcol), int(gcol), int(rcol))

            col_L = color_for_last(left_label_col)
            col_R = color_for_last(right_label_col)

            for i in range(len(pts) - 1):
                age = (len(pts) - 2 - i)
                alpha = float(np.clip(1.0 - age / max(1, tail_len), 0.15, 1.0))
                thickness = max(1, int(round(2 * alpha)))

                for canvas, col in ((canvas_L, col_L), (canvas_R, col_R)):
                    overlay = canvas.copy()
                    cv2.line(overlay, tuple(pts[i]), tuple(
                        pts[i + 1]), col, thickness, cv2.LINE_AA)
                    cv2.addWeighted(overlay, alpha, canvas,
                                    1.0 - alpha, 0, dst=canvas)

        # ---- draw current markers for time t ----
        if t in by_t:
            cur = by_t[t].dropna(subset=["trax", "tray"])

            def draw_markers(canvas, label_col):
                c = cur.dropna(subset=[label_col])
                if len(c) == 0:
                    return np.zeros(6, dtype=int)
                lab_idx = c[label_col].astype(int).to_numpy() - 1
                lab_idx = np.clip(lab_idx, 0, 5)
                counts = np.bincount(lab_idx, minlength=6)

                xs = c["trax"].to_numpy(np.int32)
                ys = c["tray"].to_numpy(np.int32)

                for x, y, li in zip(xs, ys, lab_idx):
                    rcol, gcol, bcol = COLORS_RGB[li]
                    col = (int(bcol), int(gcol), int(rcol))
                    cv2.circle(canvas, (int(x), int(y)),
                               3, col, -1, cv2.LINE_AA)
                return counts

            counts_L = draw_markers(canvas_L, left_label_col)
            counts_R = draw_markers(canvas_R, right_label_col)

        # Titles on panels
        cv2.putText(canvas_L, title_left, (18, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (245, 245, 245), 2, cv2.LINE_AA)
        cv2.putText(canvas_R, title_right, (18, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (245, 245, 245), 2, cv2.LINE_AA)

        # Convert to RGB for stacking
        out_L = cv2.cvtColor(canvas_L, cv2.COLOR_BGR2RGB)
        out_R = cv2.cvtColor(canvas_R, cv2.COLOR_BGR2RGB)

        # Legends
        legend_L = make_legend_panel(H, counts_L, title=title_left)
        legend_R = make_legend_panel(H, counts_R, title=title_right)
        legend_L = cv2.cvtColor(legend_L, cv2.COLOR_BGR2RGB)
        legend_R = cv2.cvtColor(legend_R, cv2.COLOR_BGR2RGB)

        sep_w = 12
        sep_rgb = np.full((H, sep_w, 3), 30, dtype=np.uint8)

        combined = np.hstack(
            [out_L, sep_rgb, out_R, sep_rgb, legend_L, sep_rgb, legend_R])
        combined = downscale_to_max_width(combined, max_out_width)
        if enforce_macroblock:
            combined = crop_to_16(combined)

        writer.append_data(combined)

    writer.close()
    return str(filename)
