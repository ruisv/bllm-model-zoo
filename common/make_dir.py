#!/usr/bin/env python3
"""The ONLY thing in this repo that writes a model.json — a thin wrapper over
BLLM's own `bllm-make-model-dir` (shipped with the conda package). It exists so a
recipe never hand-writes model.json: the eos, chat markers and (for VLM) the four
vision settings that cannot be defaulted are resolved by the one authoritative
tool, and a recipe just names the parts.

    python common/make_dir.py --config models/qwen3.5-0.8b/config.yaml \\
        --hbm text.hbm --embed embed_tokens.bin --tokenizer tokenizer.json \\
        --out ~/models/qwen3.5-0.8b-ctx2k

Why not write model.json directly: a mis-set eos never stops the model, a mis-set
mrope turns images into noise, and none of it shows up in the .hbm — only
bllm-make-model-dir resolves eos by authority (generation_config -> specials ->
SP-vocab) and forces the four vision flags for a hybrid VLM. See BLLM docs/MODELS.
"""
import argparse
import subprocess
import sys

try:
    import yaml
except ImportError:
    yaml = None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="models/<name>/config.yaml")
    ap.add_argument("--out", required=True)
    ap.add_argument("--hbm", required=True)
    ap.add_argument("--embed", help="hybrid/omni host embedding table")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--prefill", help="optional seq_len=N prefill graph .hbm")
    ap.add_argument("--visual", help="vision tower .hbm (VLM)")
    a = ap.parse_args()

    if yaml is None:
        print("pip install pyyaml", file=sys.stderr); return 2
    cfg = yaml.safe_load(open(a.config))
    arch = cfg["emit"]["arch"]
    cl = cfg["compile"]["cache_len"]
    cores = cfg["compile"].get("bpu_cores", 1)

    cmd = ["bllm-make-model-dir", arch, a.out,
           "--hbm", a.hbm, "--tokenizer", a.tokenizer, "--cache-len", str(cl)]
    if a.embed:    cmd += ["--embed", a.embed]
    if a.prefill:  cmd += ["--hbm-prefill", a.prefill]
    if a.visual:   cmd += ["--visual", a.visual]
    if cores > 1:  cmd += ["--bpu-cores", str(cores)]
    # VLM: config.yaml carries the four un-defaultable vision settings
    v = cfg.get("vision")
    if v:
        cmd += ["--vision-patch", str(v["patch"]),
                "--vision-mean", *map(str, v["mean"]),
                "--vision-std", *map(str, v["std"]),
                "--mrope-section", *map(str, v["mrope_section"])]
        if v.get("mrope_interleaved"):
            cmd += ["--mrope-interleaved"]

    print("+", " ".join(cmd))
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
