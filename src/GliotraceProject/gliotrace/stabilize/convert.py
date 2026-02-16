from pathlib import Path
import numpy as np
import pandas as pd
import re
from scipy.io import loadmat
import json
import h5py


def stack_conversion(inpath, outpath, metadata_path):
    """
    Stack conversion from .mat to .npz format.
    Takes input and output path and checks the overlap to
    avoid overwriting already converted stacks.
    Returns a list of all the stacks in outpath after
    conversion.

    Inputs:
    inpath - path of stacks to convert
    outpath - final destination for converted stacks
    metadata - information on

    Returns:
    stacktable - .json list of all stacks in outpath
    """

    # Read metadata from path
    new_path = read_inhouse_metadata(metadata_path)

    # List all stacks in paths
    input_path = Path(inpath)
    output_path = Path(outpath)

    new_files = list(input_path.rglob("*.mat"))
    old_files = list(output_path.rglob("*.npz"))

    # Compare new and old files
    old_names = {f.stem for f in old_files}
    to_process = [f for f in new_files if f.stem not in old_names]

    # Fetch unique experiments
    new_exps = np.unique(
        [int(re.search(r"exp_(\d+)", f.name).group(1)) for f in to_process])
    old_exps = np.unique(
        [int(re.search(r"exp_(\d+)", f).group(1)) for f in old_names])

    # Create folders if missing
    metadata = pd.read_csv(new_path)
    old_sets = np.unique([metadata.loc[metadata["experiment_id"]
                         == exp, "set"].values[0] for exp in old_exps]).astype(int)
    new_sets = np.unique([s for exp in new_exps if (
        s := metadata.loc[metadata["experiment_id"] == exp, "set"].iloc[0]) not in old_sets]).astype(int)

    for s in new_sets:
        folder_name = f"Set_{s}"
        (output_path / folder_name).mkdir(parents=True, exist_ok=True)

    # Load stacks, assure uint8, save as .npz
    for f in to_process:
        try:
            data = loadmat(f, struct_as_record=False, squeeze_me=True)
            stack = data['stack']
            tstack = np.clip(stack.Tstack, 0, 255).astype(np.uint8)
            vstack = np.clip(stack.Vstack, 0, 255).astype(np.uint8)
        except NotImplementedError:
            with h5py.File(f, 'r') as f_h5:
                tstack = np.clip(
                    np.array(f_h5['stack']['Tstack']), 0, 255).astype(np.uint8)
                vstack = np.clip(
                    np.array(f_h5['stack']['Vstack']), 0, 255).astype(np.uint8)

        bstack = np.zeros_like(tstack)

        exp = int(re.search(r"exp_(\d+)", f.name).group(1))
        set = metadata.loc[metadata["experiment_id"]
                           == exp, "set"].iloc[0].astype(int)
        delta = metadata.loc[metadata["experiment_id"]
                             == exp, "delta_t"].iloc[0]

        stack = {
            "Tstack": tstack,
            "Vstack": vstack,
            "Bstack": bstack,
            "delta": delta,
        }

        stackname = output_path / f"Set_{set}" / f.name.replace(".mat", "")

        np.savez_compressed(
            stackname,
            **stack,
        )

    stacktable = list(output_path.rglob("*.npz"))
    outfile = Path.cwd().parent / "stacktable.json"
    with open(outfile, "w") as f:
        json.dump(stacktable, f, indent=4, default=str)


def read_inhouse_metadata(metadata_path):
    """
    Converts Nelander-style metadata to gliotrace compatible format.
    Saves metadata file as .csv.

    Inputs:
    metadata_path

    Saves:
    metadata as .csv
    """

    # Open file
    data = pd.read_excel(metadata_path)

    # Select columns to use
    cols_ = ["experiment", "set", "HGCC", "perturbation",
             "dose", "unit", "delta_t", "frames", "t",
             "missing_frames"]

    cols_select = [c for c in data.columns if any(
        c.lower() == p.lower() or c.lower().startswith(f"{p.lower()}_") for p in cols_)]
    filtered_data = data[cols_select]

    # Replace experiment column name
    exp_idx = next(i for i, c in enumerate(
        filtered_data.columns) if c.startswith('exp'))
    filtered_data = filtered_data.rename(
        columns={filtered_data.columns[exp_idx]: "experiment_id"})
    filtered_data["experiment_id"] = filtered_data["experiment_id"].str.extract(
        r"(\d+)").astype(int)

    # Replace patient id column name if incorrect
    if "hgcc" in filtered_data.columns.str.lower():
        filtered_data = filtered_data.rename(columns={"hgcc": "patient_id",
                                                      "HGCC": "patient_id"})

    # Remove rows with undefined deltat
    filtered_data.dropna(subset=["delta_t"], inplace=True)

    # Save as .csv in upper dir
    filtered_data.to_csv(Path.cwd().parent / "metadata.csv", index=False)
    return Path.cwd().parent / "metadata.csv"
