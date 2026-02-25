from pathlib import Path
import json

from gliotrace.stabilize.process_image import select_stabilize

# Inputs
file_path = Path("Path/To/TiffFolder")
output_path = Path("Stack/outputPath")
json_path = Path("Path_stack_File.json")

# The process image program is runnable in three modes:

# Select ROI for stabilization (full image no selection)
stack_roi = select_stabilize(
    file_path=str(file_path),
    output_path=str(output_path),
    mode="full"
)

# Select ROI for stabilization (manual roi selection)
stack_roi = select_stabilize(
    file_path=str(file_path),
    output_path=str(output_path),
    mode="manual",
    region_size=400,
)

# Select ROI for stabilization (Coordinates selection):
# exp, X, Y, H, W
import pandas as pd

# Specify coordinates for each roi in each experiment:
df = pd.DataFrame([[12, 0, 0, 200, 200]], columns=["exp", "X", "Y", "W", "H"])
df.to_csv("coords.csv", index=0)

stack_roi = select_stabilize(
    file_path=str(file_path),
    output_path=str(output_path),
    mode="coords",
    coordinate_file="coords.csv"
)

# Save ROI settings/metadata to JSON
json_path.parent.mkdir(parents=True, exist_ok=True)
with json_path.open("w", encoding="utf-8") as f:
    json.dump(stack_roi, f, indent=2, ensure_ascii=False)