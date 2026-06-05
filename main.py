import nbformat

notebook_path = "notebooks/training_ViT.ipynb"

# Load the broken notebook
with open(notebook_path, "r", encoding="utf-8") as f:
    nb = nbformat.read(f, as_version=4)

# Option A: Completely remove the widgets metadata (Easiest)
if "widgets" in nb.get("metadata", {}):
    del nb["metadata"]["widgets"]
    print("Removed 'widgets' from metadata.")

# Option B: Or, if you want to keep it, force an empty state
# if "widgets" in nb.get("metadata", {}):
#     for key in nb["metadata"]["widgets"]:
#         if "state" not in nb["metadata"]["widgets"][key]:
#             nb["metadata"]["widgets"][key]["state"] = {}

# Save the fixed notebook
with open(notebook_path, "w", encoding="utf-8") as f:
    nbformat.write(nb, f)

print("Notebook successfully updated!")