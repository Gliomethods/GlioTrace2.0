from gliotrace.initalize_class.defaults import SOFTMAX_COLUMNS

import json
import torch
import torch.nn.functional as F
from pathlib import Path
from scipy.io import loadmat
import numpy as np
import pandas as pd
from cv2 import imwrite


def classify_tumor_cells(
    feat,
    vasc,
    blocksize,
    morph_net,
    tme_net,
    m,
    debug=False,
    output_path="",
):
    """
    Classify tumor cells with morphology and TME networks.

    Returns
    -------
    properties_meta : list of pandas.DataFrame
        One DataFrame per frame with:
        - frame_index (int, 1-based)
        - cell_index (int, 0-based index within frame)
        - morphology class probabilities (columns = class names)
        - state_label (1..6, morphology class index)
        - tme_label (1, 2, or 3)

    properties_embed : list of pandas.DataFrame
        One DataFrame per frame with:
        - frame_index
        - cell_index
        - emb_0 .. emb_255 (256-dim embedding)
    """

    SCRIPT_DIR = Path(__file__).resolve().parent

    mean_image_tme_path = SCRIPT_DIR / "mean_image_tme.mat"
    mean_image_tme = loadmat(mean_image_tme_path)["mean_image"]

    stats_path = SCRIPT_DIR / "stats.json"
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    mean = stats["mean_rgb_0_1"]
    std = stats["std_rgb_0_1"]

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    device_morph = "cuda" if torch.cuda.is_available() else "cpu"
    device_tme = "cuda" if torch.cuda.is_available() else "cpu"

    # Pre-build tensors for normalization
    mean_t = torch.tensor(mean, device=device_morph,
                          dtype=torch.float32).view(1, 3, 1, 1)
    std_t = torch.tensor(std, device=device_morph,
                         dtype=torch.float32).view(1, 3, 1, 1)

    # TME class names
    classNames2 = ["Microglia colocalized", "Vessel associated"]

    properties_meta = []   # metadata: probs, labels, TME, indices

    morph_net.eval()
    tme_net.eval()

    with torch.no_grad():
        # Iterate through frames
        for i, (mat, mat_vasc) in enumerate(zip(feat, vasc), start=1):
            n_cells = mat.shape[0]

            frame_rows_meta = []
            frame_rows_embed = []

            for j in range(n_cells):
                # Create RGB image
                im = np.zeros((blocksize, blocksize, 3), dtype=np.float32)

                # Green channel: MATLAB(:,:,2) -> Python[:,:,1]
                im[:, :, 1] = mat[j, :].reshape(blocksize, blocksize)

                # Red channel: MATLAB(:,:,1) -> Python[:,:,0]
                im[:, :, 0] = mat_vasc[j, :].reshape(blocksize, blocksize)

                # Scale to [0,1]
                im_norm = im / 255.0

                # ---- TME uses mean image----
                im_predict_tme = im_norm - mean_image_tme

                # ---- Morphology preprocessing  ----
                im_morph_tensor = (
                    torch.from_numpy(im_norm)
                    .permute(2, 0, 1)   # HWC -> CHW
                    .unsqueeze(0)       # [1, 3, H, W]
                    .to(device_morph, dtype=torch.float32)
                )

                im_morph_tensor = (im_morph_tensor - mean_t) / \
                    std_t       # Normalize

                # ---- TME tensor ----
                im_tme_tensor = (
                    torch.from_numpy(im_predict_tme)
                    .permute(2, 0, 1)
                    .unsqueeze(0)
                    .to(device_tme, dtype=torch.float32)
                )

                # --- Morphology prediction ---
                logits_morph = morph_net(
                    im_morph_tensor, return_embedding=False
                )

                probs_morph = F.softmax(logits_morph, dim=1)
                probs_morph_np = probs_morph.cpu().numpy().squeeze()

                # Argmax + 1-based index
                predictedIndex_0 = int(np.argmax(probs_morph_np))
                predictedIndex = predictedIndex_0 + 1  # 1..6
                predictedLabel = SOFTMAX_COLUMNS[predictedIndex_0]

                max_prob_morph = float(np.max(probs_morph_np))

                # Debug image saving by morphology confidence
                if debug:
                    if max_prob_morph < 0.6:
                        folder = Path(predictedLabel) / "<60%"
                    else:
                        folder = Path(predictedLabel) / ">60%"

                    folder_path = output_path / folder
                    folder_path.mkdir(parents=True, exist_ok=True)

                    filename = (
                        f"score_{max_prob_morph}_stack_{m}_im_{i}_"
                        f"{np.random.randint(1, 1001)}_.tif"
                    )
                    imwrite(folder_path / filename,
                            (im_norm * 255).astype(np.uint8))

                # --------- BUILD ROWS (META + EMBEDDING) ---------

                # --- METADATA ROW (no string cell_id, just indices) ---
                row_meta = {
                    "frame_index": i,  # 1-based frame number
                    "cell_index": j,  # 0-based cell index within this frame
                }

                # Add morphology probabilities
                for cls, prob in zip(SOFTMAX_COLUMNS, probs_morph_np):
                    row_meta[cls] = float(prob)

                # Predicted morphology label (1..6)
                row_meta["state_label"] = predictedIndex

                # --- TME prediction (PyTorch) ---
                logits_tme = tme_net(im_tme_tensor)  # [1, 2] logits
                probs_tme = F.softmax(logits_tme, dim=1)  # [1, 2]
                probs_tme_np = probs_tme.cpu().numpy().squeeze()  # (2,)

                score = float(np.max(probs_tme_np))
                predictedIndex_tme_0 = int(np.argmax(probs_tme_np))
                predictedLabel_tme = classNames2[predictedIndex_tme_0]

                # MATLAB used 1-based indices for TME classes
                predictedIndex_tme = predictedIndex_tme_0 + 1  # 1 or 2 initially

                # Thresholding logic
                if predictedLabel_tme == "Microglia colocalized" and score < 0.975:
                    folder_name = "Non-associated"
                    predictedIndex_tme = 3
                elif predictedLabel_tme == "Vessel associated" and score < 0.8:
                    folder_name = "Non-associated"
                    predictedIndex_tme = 3
                else:
                    folder_name = predictedLabel_tme

                if debug:
                    folder_path = output_path / folder_name
                    folder_path.mkdir(parents=True, exist_ok=True)

                    filename = (
                        f"score_{score}_stack_{m}_im_{i}_"
                        f"{np.random.randint(1, 1001)}_.tif"
                    )
                    imwrite(folder_path / filename,
                            (im_norm * 255).astype(np.uint8))

                # Save TME status (1, 2, or 3 for Non-associated)
                row_meta["tme_label"] = predictedIndex_tme

                # Collect rows for this frame
                frame_rows_meta.append(row_meta)

            # Build DataFrames for this frame
            df_meta = pd.DataFrame(frame_rows_meta)

            properties_meta.append(df_meta)

    return properties_meta
