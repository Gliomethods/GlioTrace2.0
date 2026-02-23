from gliotrace.stabilize.process_image import select_stabilize
import json

file_path = "C://Users//andrelas//Desktop//cns_code//GlioTrace2.0//src//GliotraceProject//mouse1"
output_path = "C://Users//andrelas//Desktop//cns_code//GlioTrace2.0//src//GliotraceProject//result"


stack_roi_man = select_stabilize(
    file_path=file_path, output_path=output_path, mode="manual", region_size=400
)

with open("C://Users//andrelas//Desktop//cns_code//GlioTrace2.0//src//GliotraceProject//stackable.json", "w") as f:
    json.dump(stack_roi_man, f, indent=2)
