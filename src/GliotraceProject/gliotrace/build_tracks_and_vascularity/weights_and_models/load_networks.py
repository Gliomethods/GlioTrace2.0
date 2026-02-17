import torch
from pathlib import Path

from gliotrace.build_tracks_and_vascularity.weights_and_models import models

PACKAGE_ROOT = Path(__file__).resolve().parent


def load_trained_networks(device=None):
    """
    Load pretrained PyTorch networks (state, TME, vasculature segmentation) from packaged weights.

    Parameters
    ----------
    device : torch.device or None
        Target device for the models. If None, selects CUDA if available, otherwise CPU.

    Returns
    -------
    net1 : torch.nn.Module
        "State" network (StateNet) 
    net2 : torch.nn.Module
        Binary TME network (TMENet).
    net3 : torch.nn.Module
        Vasculature segmentation network (VascNET).

    @ Author: André Lasses Armatowski
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    weights_dir = PACKAGE_ROOT / "weights"

    # ---- STATE NETWORK ----
    net1_ckpt = torch.load(weights_dir / "gbm.pt", map_location=device)

    class_names = net1_ckpt["class_names"]
    num_classes = len(class_names)

    cfg = net1_ckpt.get("config", {})
    emb_dim = cfg.get("emb_dim", 256)

    net1 = models.StateNet(
        num_classes=num_classes,
        emb_dim=emb_dim,
    ).to(device)

    net1.load_state_dict(net1_ckpt["state_dict"], strict=True)

    # ---- TME NETWORK ----
    net2 = models.TMENet().to(device)
    net2.load_state_dict(torch.load(
        weights_dir / "tme.pth", map_location=device))

    # ---- VASC NETWORK ----
    net3 = models.VascNet().to(device)
    net3.load_state_dict(torch.load(
        weights_dir / "seg.pth", map_location=device))

    net1.eval()
    net2.eval()
    net3.eval()
    return net1, net2, net3
