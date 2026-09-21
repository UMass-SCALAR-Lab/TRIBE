import sys
from pathlib import Path

# Make the `tribe` package importable regardless of cwd/shell PYTHONPATH: this repo root (the parent of
# `tribe/`) is not itself an installed package, so pytest's rootdir insertion alone doesn't add it.
_repo_root = str(Path(__file__).resolve().parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)
