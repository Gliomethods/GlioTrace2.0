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
        "State" network (MatlabNet) 
    net2 : torch.nn.Module
        Binary TME network (MatlabNet2Class).
    net3 : torch.nn.Module
        Vasculature segmentation network (MatlabSegNet).

    @ Author: André Lasses Armatowski
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    weights_dir = PACKAGE_ROOT / "weights"

    net1 = models.MatlabNet().to(device)
    net2 = models.MatlabNet2Class().to(device)
    net3 = models.MatlabSegNet().to(device)

    net1.load_state_dict(torch.load(
        weights_dir / "gbm.pth", map_location=device))
    net2.load_state_dict(torch.load(
        weights_dir / "tme.pth", map_location=device))
    net3.load_state_dict(torch.load(
        weights_dir / "seg.pth", map_location=device))

    net1.eval()
    net2.eval()
    net3.eval()
    return net1, net2, net3
