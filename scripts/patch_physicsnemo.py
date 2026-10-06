"""
Patch for nvidia-physicsnemo 2.2.2: VTKFileReader is missing the abstract
method `read_file_attributes`, so it can't be instantiated and STL inference
(predict_on_stl.py) fails with:

    TypeError: Can't instantiate abstract class VTKFileReader without an
    implementation for abstract method 'read_file_attributes'

Run it with the same Python environment that has physicsnemo installed:

    python scripts/patch_physicsnemo.py

Safe to run more than once. The original file is kept next to it as
cae_dataset.py.orig.
"""

import pathlib

import physicsnemo.datapipes.cae.cae_dataset as cae_dataset

ANCHOR = "            return dir_name / fname\n\n        def read_file(self, filename: pathlib.Path)"
PATCHED = (
    "            return dir_name / fname\n\n"
    "        def read_file_attributes(self, filename: pathlib.Path) -> dict[str, torch.Tensor]:\n"
    "            # Added by patch_physicsnemo.py: VTKFileReader is missing this\n"
    "            # abstract method in physicsnemo 2.2.2, so it can't be instantiated.\n"
    "            # Bare-STL inference has no file-level attributes to read.\n"
    "            return {}\n\n"
    "        def read_file(self, filename: pathlib.Path)"
)
MARKERS = ("Added by patch_physicsnemo.py", "Added by setup_local_inference.sh", "VTKFileReader never actually defined")


def main():
    path = pathlib.Path(cae_dataset.__file__)
    src = path.read_text()
    if any(m in src for m in MARKERS):
        print(f"Already patched: {path}")
        return
    if src.count(ANCHOR) != 1:
        raise SystemExit(f"Patch anchor not found in {path} -- is this physicsnemo 2.2.2?")
    path.with_suffix(".py.orig").write_text(src)
    path.write_text(src.replace(ANCHOR, PATCHED, 1))
    print(f"Patched: {path}")


if __name__ == "__main__":
    main()
