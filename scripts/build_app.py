import PyInstaller.__main__
from pathlib import Path
import shutil
import sys

# Define paths
ROOT = Path(__file__).resolve().parent.parent
DIST = ROOT / "dist"
BUILD = ROOT / "build"

def build():
    print(f"Building UCBFaculty from {ROOT}...")
    
    # Clean previous builds
    if DIST.exists():
        print("Cleaning dist...")
        shutil.rmtree(DIST)
    if BUILD.exists():
        print("Cleaning build...")
        shutil.rmtree(BUILD)

    # PyInstaller arguments
    args = [
        str(ROOT / "ucbfaculty" / "cli.py"),
        "--name=UCBFaculty",
        "--noconfirm",
        "--clean",
        # --onedir keeps startup fast (no temp-dir extraction on every launch).
        # Distribute the resulting UCBFaculty/ folder as a zip or installer.
        "--onedir",

        # Hidden imports that PyInstaller's analysis misses
        "--hidden-import=panel",
        "--hidden-import=bokeh",
        "--hidden-import=pandas",
        "--hidden-import=ligonlibrary",
        "--hidden-import=openpyxl",
        "--hidden-import=pyarrow",
        "--hidden-import=git",
        "--hidden-import=ucbfaculty.apps.lookup",
        "--hidden-import=ucbfaculty.apps.setup",

        # App source files Panel needs to serve
        f"--add-data={ROOT}/ucbfaculty/apps/*.py:ucbfaculty/apps",

        # Hidden imports for ucbfaculty submodules that static analysis may
        # miss (Panel loads apps by path, CLI lazily imports some modules).
        "--hidden-import=ucbfaculty",
        "--hidden-import=ucbfaculty.committee",
        "--hidden-import=ucbfaculty.faculty",
        "--hidden-import=ucbfaculty.feature_spec_loader",
        "--hidden-import=ucbfaculty.merge_features",
        "--hidden-import=ucbfaculty.new_feature",
        "--hidden-import=ucbfaculty.propose_spec",
        "--hidden-import=ucbfaculty.rawdata",
        "--hidden-import=ucbfaculty.security",
        "--hidden-import=ucbfaculty.spelling",
        "--hidden-import=ucbfaculty.transformations",
        "--hidden-import=ucbfaculty.utils",
        "--hidden-import=ucbfaculty.var_processing",

        # Collect full packages whose runtime assets PyInstaller can't trace.
        # --collect-all=ucbfaculty is intentionally avoided: it spawns a
        # subprocess that imports ucbfaculty.apps.*, pulling in panel/bokeh,
        # which crashes on Windows (TP_NUM_C_BUFS / ACCESS_VIOLATION).
        # We use --collect-data for data files (spelling_dictionary.json etc.)
        # and list submodule hidden-imports explicitly above.
        "--collect-all=panel",
        "--collect-all=bokeh",
        "--collect-data=ucbfaculty",
    ]
    
    print(f"Running PyInstaller with args: {args}")
    PyInstaller.__main__.run(args)
    print(f"Build complete. executable should be in {DIST}")

if __name__ == "__main__":
    build()
