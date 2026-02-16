from gliotrace.stabilize.process_image import select_stabilize
import json

file_path = "mouse1"
output_path = "result"


stack_roi_man = select_stabilize(
    file_path=file_path, output_path=output_path, mode="manual", region_size=400
)

with open("stackable.json", "w") as f:
    json.dump(stack_roi_man, f, indent=2)
