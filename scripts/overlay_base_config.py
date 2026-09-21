"""
Overwrites a checkpoint directory's non-weight artifacts (config.json, generation_config.json, tokenizer
files, chat_template.jinja) with the ones from the base model it was fine-tuned from, keeping only the
checkpoint's own weights untouched. See conversation: transformers 5.14.1's save_pretrained() silently
drops Llama-3.2's top-level rope_theta (merges it into a renamed rope_parameters dict vLLM doesn't
recognize), degrading every checkpoint's generation quality without changing a single weight. Weights were
independently verified byte-identical to base pre- and post-training (state_dict diff, 0 nonzero params) --
this only ever needs to touch non-weight files.

Prefer this over patching individual fields (tokenizer_class, then rope_scaling, then rope_theta -- three
rounds of whack-a-mole to find this one bug): loading the whole non-weight artifact set from a known-good
source makes the next schema-drift field the writer's transformers version introduces a non-issue, instead
of another round of field-level archaeology.

Usage:
python scripts/overlay_base_config.py \
    --checkpoint_dir /path/to/trained/checkpoint \
    --base_model meta-llama/Llama-3.2-3B-Instruct
"""

import argparse
import json
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download
from transformers import __version__ as reader_transformers_version

NON_WEIGHT_FILES = [
    "config.json",
    "generation_config.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "chat_template.jinja",
    "special_tokens_map.json",
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--base_model", required=True, help="The model this checkpoint was fine-tuned from.")
    args = parser.parse_args()

    ckpt_dir = Path(args.checkpoint_dir)
    config_path = ckpt_dir / "config.json"
    # Record the writer's version BEFORE it gets overwritten -- this is the actual variable behind the
    # whole bug class (a version mismatch between the env that trained/saved and the env that evals), and
    # nothing else in this project's artifacts currently records it.
    writer_transformers_version = None
    if config_path.exists():
        writer_transformers_version = json.loads(config_path.read_text()).get("transformers_version")

    base_dir = Path(snapshot_download(args.base_model))
    for filename in NON_WEIGHT_FILES:
        src = base_dir / filename
        if src.exists():
            shutil.copy(src, ckpt_dir / filename)

    # Assert, don't trust: the whole failure mode was a silently-dropped field. An overlay that could
    # silently drop a field too (wrong base_model, a stale snapshot, a future rope schema change again) is
    # the same bug in a new place -- fail loudly instead.
    overlaid_config = json.loads((ckpt_dir / "config.json").read_text())
    base_config = json.loads((base_dir / "config.json").read_text())
    assert overlaid_config.get("rope_theta") == base_config["rope_theta"], (
        f"rope_theta mismatch after overlay: {overlaid_config.get('rope_theta')} != {base_config['rope_theta']}"
    )
    assert overlaid_config.get("rope_scaling", {}).get("factor") == base_config["rope_scaling"]["factor"], (
        f"rope_scaling.factor mismatch after overlay: {overlaid_config.get('rope_scaling')} != {base_config['rope_scaling']}"
    )

    manifest = {
        "overlay_base_model": args.base_model,
        "writer_transformers_version": writer_transformers_version,
        "reader_transformers_version": reader_transformers_version,
    }
    (ckpt_dir / "overlay_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Overlay OK. writer={writer_transformers_version} reader={reader_transformers_version}")
