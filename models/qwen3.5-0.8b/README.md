# Qwen3.5-0.8B — hybrid Gated-DeltaNet SSM, text, 100% BPU int8

The mother template of this repo, and the model that best shows what BLLM's native
runtime does that the official OE-LLM stack cannot: a **hybrid linear-attention
(Gated-DeltaNet / SSM) + full-attention** model, quantised to int8, running
**strictly 100% on the BPU** (`CPU_inference_time_cost == 0`), at ~21.5 tok/s on
S100P. libxlm cannot run this architecture at all (attention-only, head_dim
hardwired).

Upstream: [Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B), Apache-2.0
(check the model card). This is the **text** package — Qwen3.5 is natively an
image-text model; the vision half is a separate recipe (`qwen3.5-vlm`).

| variant | cache_len | on-board (S100P) |
|---|---|---|
| `qwen3.5-0.8b-ctx2k-int8-s100` | 2048 | 21.5 tok/s |
| `qwen3.5-0.8b-ctx4k-int8-s100` | 4096 | 18.65 tok/s |

`cache_len` is a **performance** knob, not only capacity — a larger KV window is
more per-token traffic, so ctx4k decodes slower. Emit the cache_len that matches
your typical context.

## What goes wrong (the builds that are wrong for you)

An LLM build compiles cleanly, loads, and runs at full speed even when it is
wrong — the failure is silent degradation, not a crash. The ones recorded in
`expected.json.rejected_builds`:

- **Dynamic-quant decode** (`block_quantized_matmul`) — 6 s/token. That path is a
  *prefill* op (51 ms/matmul at M=1); decode must use the static int8-weight /
  int16-activation `leap.linear` path (0.33 ms/matmul). Same weights, 130× slower
  if you pick the wrong matmul.
- **Newton triangular inverse in the all-BPU prefill** — cos −0.34 on deep layers.
  The inverse is numerically the hard part; Newton doubling's `2I−AX` transiently
  overshoots and int16 clips it. Forward-substitution / block-recursion are the
  stable inverses.
- **Flat int16 activation ranges on 4B** (not this model) — 4B clips and loops on
  the prompt; it needs per-linear activation calibration. 0.8B/2B fit the flat
  data-free ranges. Recorded here because it is the first thing to check when a
  scaled-up sibling degenerates.

## The prefill story (why the shipped one keeps [C,C] on the CPU)

The chunked delta-rule prefill needs a per-sub-chunk triangular solve. Its `[C,C]`
intermediates are tiny and input-dependent, so a fixed int16 range starves them —
the shipped build runs those matmuls in **float on the CPU** (correct, 2.30 s
image TTFT via the VLM recipe). An all-BPU int16 version is numerically solved
(block-recursive inverse, cos 0.9998) but hbcompile schedules the single fused
18-layer segment with a **serial** pass unless `max_time_per_fc` splits it into
funccalls. This is documented for the record, not because 100% BPU there is a
speed win — prefill is compute-bound and the CPU/BPU split is already good.

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
