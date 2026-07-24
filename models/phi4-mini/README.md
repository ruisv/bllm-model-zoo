# Phi-4-mini — dense 3.8B, int8, the `phi` chat format

A dense model the official OE-LLM toolchain does not ship, brought onto the BPU by
**remapping onto the qwen2.5-7b leap recipe** (both dense, both head_dim 128) with
a **partial-rotary patch**. int8, ~8.7 tok/s on S100P. It is the reference for two
things this repo cares about: the **remap route** for a non-official dense arch,
and the **`phi` chat format** (BLLM's third format after chatml and none).

Upstream: [Phi-4-mini-instruct](https://huggingface.co/microsoft/Phi-4-mini-instruct),
MIT (check the model card).

| variant | cache_len | on-board (S100P) |
|---|---|---|
| `phi4-mini-ctx1k-int8-s100` | 1024 | 8.7 tok/s |

## The `phi` chat format (why it is not chatml)

Phi frames a turn as `<role>content<|end|>` — a bare role marker, no role text, no
newline — where the assistant's previous turn is closed by re-emitting `<|end|>`
on the next turn (it is an eos, so it is never left sitting in the KV cache). This
is a genuinely different template from ChatML, and BLLM has a dedicated `phi`
branch for it. `bllm-make-model-dir` **auto-detects** the markers
(`<|user|>`200021 / `<|assistant|>`200019 / `<|system|>`200022 / `<|end|>`200020)
and emits `format:"phi"` — do not hand-write it, and do not force chatml.

## What goes wrong

Recorded in `expected.json.rejected_builds`, all of them silent (the model loads,
runs, generates plausible text, then misbehaves):

- **chatml on Phi** — wrong turn framing; drifts and does not stop cleanly.
- **wrong eos** — if `<|end|>` is not the eos, the model never stops or leaks the
  marker into the reply. The single most common LLM-packaging failure. `eos` is
  resolved by `bllm-make-model-dir`, never hand-set.
- **missing partial-rotary patch** — Phi rotates only part of head_dim; remapping
  onto a full-rotary recipe without the patch mis-applies rope to the non-rotated
  dimensions. Loads and runs, degenerates.

## The remap route (head_dim == 128 is the gate)

Phi-4-mini reaches the BPU by borrowing the **qwen2.5-7b** dense leap recipe — the
two are close enough (dense, head_dim 128) that the recipe transfers with a
partial-rotary patch. **head_dim == 128 is the hard gate**: an architecture with a
different head_dim does not go through this remap. GLM-Edge and the Qwen3 dense
family reach the BPU the same way (the Qwen3 family also needs a QK-norm patch).
The graph source is private (see repo README); this recipe records the decisions.

## Acceptance

Coherence anchors: EN factual (Paris), multi-turn name recall — this specifically
proves the `phi` turn open/close is correct, since a broken template loses the
earlier turn — and a clean stop on `<|end|>` with no marker leak. Then
`common/eval_ppl.py`. Numbers in `expected.json`.

## Build

Given the compiled `.hbm` + `tokenizer.json`, `common/make_dir.py` (wrapping
`bllm-make-model-dir`) writes `model.json`, auto-detecting the phi markers, and
produces the `bllm.load()`-ready directory. `config.yaml` records the knobs.
