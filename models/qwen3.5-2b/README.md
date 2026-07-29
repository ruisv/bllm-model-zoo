# Qwen3.5-2B — hybrid Gated-DeltaNet SSM, text, 100% BPU int8

The middle sibling of the Qwen3.5 hybrid-SSM family. Architecturally identical in
*shape* to [qwen3.5-0.8b](../qwen3.5-0.8b/) — same symmetric GDN head layout (16
heads, head_dim 128, key heads == value heads), same attention shape (8 query / 2
kv heads, head_dim 256) — only hidden size (2048 vs 1024), intermediate size (6144
vs 3584) and weight volume scale up. Contrast with
[qwen3.5-4b](../qwen3.5-4b/), whose GDN key and value heads are **decoupled**
(16 key / 32 value) — that is a genuine architecture change, not just a bigger
version of this one.

Upstream: [Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B), Apache-2.0 (check
the model card). This is the **text** package — Qwen3.5 is natively an
image-text model; the vision half is the [qwen3.5-vlm](../qwen3.5-vlm/) recipe,
which packages this exact decoder `.hbm` alongside a vision tower.

| variant | cache_len | on-board (S100P) |
|---|---|---|
| `qwen3.5-2b-ctx4k-int8-s100` | 4096 | 13.20 tok/s |
| `qwen3.5-2b-ctx512-int8-s100` | 512 | 15.29 tok/s |

Measured 2026-07-29 against this release's GQA batched-matmul + mask-dedup
attention rewrite — essentially unchanged from the pre-rewrite baseline (13.22
tok/s), consistent with board output verified bit-identical to that baseline.

## What goes wrong

Same architecture family as `qwen3.5-0.8b`, so the same failure modes apply —
see that recipe's `expected.json.rejected_builds` for the full write-up
(dynamic-quant decode is 130× slower if the static `leap.linear` path is not
used; the all-BPU prefill's triangular inverse must be forward-substitution or
block-recursion, not Newton doubling, or it goes numerically wrong on deep
layers under int16). Nothing new is introduced at this size — the symmetric GDN
heads mean the same math, just bigger tensors.

## Decode scales with weight volume, not just parameter count

`qwen3.5-0.8b` decodes at 21.5 tok/s (ctx2k) / 18.65 tok/s (ctx4k); this model at
13.22 tok/s (ctx4k) — decode here is weight-bandwidth-bound (see
`qwen3.5-0.8b`'s measured breakdown: ~95% of the token is the BPU graph
streaming int8 weights), so the ratio tracks weight volume closely, not FLOPs.
Sizing a deployment on tok/s alone without checking the weight-bandwidth story
will mis-predict how a fine-tune with a different `intermediate` size scales.

## Running alongside another model (ION is per-package, not per-board)

This decoder `.hbm` was verified to load and run correctly **while a different
model was resident on the same board** — this size class is a realistic
multi-package deployment, not just the smallest demo model. `bllm-serve` (BLLM's
serving layer) supports exactly one loaded model at a time per process; to swap
models, stop the running instance before loading a new one rather than trying to
hold two BPU graphs' ION allocations simultaneously, which is unsupported and can
take the board down without raising an exception.

## Acceptance

Run `common/eval_parity.py` (prefill↔decode identity, prompt > chunk) and
`common/eval_ppl.py`, then the coherence anchors (EN/ZH factual, arithmetic,
multi-turn, thinking on/off). Cosine vs fp32 is a smoke test only — the parity +
task numbers are the gate. Numbers in `expected.json`.

## Build

The conversion is the native leap-graph toolchain (not published — see repo
README). Given a compiled text `.hbm` + `embed_tokens.bin` + `tokenizer.json`,
`common/make_dir.py` (which wraps `bllm-make-model-dir`) writes the `model.json`
and produces the `bllm.load()`-ready directory. `config.yaml` records the knobs.
