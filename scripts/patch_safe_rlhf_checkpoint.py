"""
Patch a Safe-RLHF-produced checkpoint so it's directly loadable with `from_pretrained`.

Safe-RLHF's own `save()` (safe_rlhf/trainers/base.py) writes `config.json` and the tokenizer files into
the TOP-LEVEL output directory, but under ZeRO-3 the actual weights get converted (via `zero_to_fp32.py`)
into a `pytorch_model.bin/` SUBDIRECTORY (sharded `.bin` files + `pytorch_model.bin.index.json`) that
doesn't have those files alongside it — `AutoModelForScore`/`LlamaForScore.from_pretrained` on that
subdirectory alone fails. Applies to every checkpoint this project loads out of the safe-rlhf repo (SFT,
reward model, cost model, ...), not just one specific handoff — call this before loading any of them.

Usage: python scripts/patch_safe_rlhf_checkpoint.py /path/to/checkpoint/sft
(patches /path/to/checkpoint/sft/pytorch_model.bin/ in place; safe to re-run, skips files already there)
"""

import shutil
import sys
from pathlib import Path


CONFIG_FILES = ["config.json", "tokenizer_config.json", "tokenizer.json", "special_tokens_map.json", "tokenizer.model"]


def patch_safe_rlhf_checkpoint(output_dir: str) -> str:
    """
    Args:
        output_dir (`str`):
            A Safe-RLHF training script's `--output_dir` (the top-level directory, not the
            `pytorch_model.bin` subfolder itself).

    Returns:
        `str`: the path to actually pass to `from_pretrained` — the `pytorch_model.bin` subfolder if one
        exists (patched in place), otherwise `output_dir` itself unchanged (nothing to patch, e.g.
        `--save_16bit True` was used, which saves directly into `output_dir` with no subfolder).
    """
    output_dir = Path(output_dir)
    weights_dir = output_dir / "pytorch_model.bin"
    if not weights_dir.is_dir():
        return str(output_dir)

    for name in CONFIG_FILES:
        src = output_dir / name
        dst = weights_dir / name
        if src.is_file() and not dst.exists():
            shutil.copy2(src, dst)

    return str(weights_dir)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} /path/to/checkpoint/output_dir", file=sys.stderr)
        sys.exit(1)
    patched_path = patch_safe_rlhf_checkpoint(sys.argv[1])
    print(patched_path)
