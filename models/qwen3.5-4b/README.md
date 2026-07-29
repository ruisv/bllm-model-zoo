# Qwen3.5-4B — hybrid Gated-DeltaNet SSM, text, 100% BPU int8, decoupled GDN heads

The largest of the Qwen3.5 hybrid-SSM family currently converted, and
architecturally distinct from its smaller siblings, not just bigger: this size
**decouples the GDN key and value head counts** (16 key heads, 32 value heads)
where [qwen3.5-0.8b](../qwen3.5-0.8b/) and [qwen3.5-2b](../qwen3.5-2b/) are
symmetric (16/16). That is a real shape change in the recurrence, not a scaling
factor — a recipe that copies the 0.8b/2b constants and only widens `hidden`
will fail during weight loading (a tensor reshape that assumes symmetric heads
does not match the checkpoint's actual tensor shapes) or, worse, silently
mis-shapes the recurrence if the mismatch happens to reshape without error.

Upstream: [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B), Apache-2.0
(check the model card). This is the **text** package — the image+text package
built from this same decoder is the [qwen3.5-vlm](../qwen3.5-vlm/) recipe.

| variant | cache_len | on-board (S100P) |
|---|---|---|
| `qwen3.5-4b-ctx4k-int8-s100` | 4096 | 6.14 tok/s |
| `qwen3.5-4b-ctx512-int8-s100` | 512 | 7.54 tok/s |

Measured 2026-07-29 against this release's GQA batched-matmul + mask-dedup
attention rewrite — nearly 2x the pre-rewrite baseline (3.3 tok/s). Unlike
0.8b/2b, this rewrite is not a pure no-op on 4B (it adds int16 quantisation the
old `expand_win` path never had); see `expected.json` for the full caveat and
why the formal acceptance run, not just the speed number, is what clears it.

## The decoupled GDN heads

Every architecture detail below is read directly off the checkpoint's
`text_config` (`linear_num_key_heads: 16`, `linear_num_value_heads: 32`, vs
0.8b/2b's `16`/`16`) — do not assume a scaled-up sibling is symmetric just
because the smaller ones are; check the config every time.

The consequence for the graph: Q and K are produced at the **key** head count
(16 heads), L2-normalized, then **broadcast up to the value head count** (32)
by repeating each key head twice (`repeat_interleave`, not a learned
projection) before entering the delta-rule recurrence — V is already produced
at 32 heads directly. The recurrence itself, and its state tensor shape
(`[value_heads, head_dim, head_dim]` = `[32,128,128]`, not `[16,128,128]`),
therefore run at the **value** head count uniformly; the asymmetry only matters
at the point Q/K are expanded, upstream of everything else. This is exactly
GQA-style broadcasting, applied to the linear-attention recurrence instead of
softmax attention — same idea D-Robotics' own `expand_win` uses for grouped
K/V in the attention layers, one level up in the same model.

**Porting a chunked-prefill recipe from a symmetric sibling**: the batched
per-chunk delta-rule math (the `[C,C]`/`[C,Dg]` intermediates, the
forward-substitution inverse, the int16 ranges on those intermediates) is
**unchanged** — those ranges were derived to be O(1)..O(20) regardless of
hidden size (see `qwen3.5-0.8b`'s prefill note), and that argument does not
depend on symmetric heads either. What has to change is everything *before*
the recurrence: the tensor split boundaries (key-head-count-wide for Q/K,
value-head-count-wide for V — not one shared boundary), and inserting the
broadcast-Q/K step. Get the reference for the correct shapes from an
already-converted, already board-verified decode graph at the same size before
writing a prefill graph for it — do not re-derive the head-splitting logic
from the paper or from a symmetric sibling's code and assume it transfers.

## Per-linear activation calibration (flat ranges clip this size)

`qwen3.5-0.8b`/`qwen3.5-2b` both fit the data-free flat `±32`/`±64` int16
activation ranges with no calibration set. At this size, the deep linears'
real `|input|` reaches roughly 200 — well past the flat range — and a clipped
build **still compiles and runs**, it just degenerates (echoes the prompt in
a loop rather than answering). The fix is per-linear calibration: run a real
fp32 forward pass over a small prompt set, record `max|input| * 1.3` margin
per linear, and use that instead of the flat constant. Concretely (measured on
this model, on-board `hrt_model_exec`-adjacent debug hooks, real not synthetic
inputs):

| linear | flat range | measured `max|input|` | why the flat range fails |
|---|---|---|---|
| `down` (MLP) | ±64 | ~198 | 3× over |
| `q`/`k`/`v` | ±32 | ~84 | 2.6× over |
| `qkv`/`a`/`b`/`z` (GDN gates) | ±32 | ~65 | 2× over |

This is the same class of failure the vision tower's massive-activation
channels hit (see `qwen3.5-vlm`'s calibration section) — a fixed range is a bet
that the real distribution stays inside it, and a bigger model is exactly where
that bet stops paying off silently.

## What goes wrong

- **Flat activation ranges** (see above) — the model degenerates into a loop,
  not a crash. This is the size-specific trap; everything below is inherited
  from the smaller siblings and still applies unchanged.
- **`block_quantized_matmul` (dynamic-quant decode)** — same failure as
  `qwen3.5-0.8b`: that path is a prefill op, ~130× slower per matmul in decode.
- **Newton triangular inverse in an all-BPU prefill** — same numerical failure
  as `qwen3.5-0.8b`; not re-derived at this size, the mechanism is head-count
  independent.
- **S600 (`nash-p`) prefill compile crash at default optimisation** — this
  size specifically; see the `qwen3.5-vlm` recipe's S600 section (the crash and
  its fix are prefill-graph-compile concerns, so they are recorded there
  alongside the rest of the prefill toolchain notes, not duplicated here).

## Acceptance

Run `common/eval_parity.py` (prefill↔decode identity, prompt > chunk) and
`common/eval_ppl.py`, then the coherence anchors. Cosine vs fp32 is a smoke
test only. Numbers in `expected.json`.

## Build

The conversion is the native leap-graph toolchain (not published — see repo
README). Given a compiled text `.hbm` + `embed_tokens.bin` + `tokenizer.json`,
`common/make_dir.py` (which wraps `bllm-make-model-dir`) writes the `model.json`
and produces the `bllm.load()`-ready directory. `config.yaml` records the knobs
— note its `gdn.key_heads` / `gdn.value_heads` split, absent from the smaller
siblings' single symmetric `heads` field.
