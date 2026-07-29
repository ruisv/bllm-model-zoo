# bllm-model-zoo

Conversion recipes and acceptance records for the on-board LLM / VLM models that
[BLLM](https://github.com/ruisv/bllm) can run on the D-Robotics RDK
**S100 / S100P / S600** BPU.

BLLM is the on-board runtime: it loads a compiled `.hbm` + tokenizer + `model.json`
and runs decode, sampling, chat, multimodal — all on the BPU, no extra LLM SDK.
**It does not convert models.** This repo is the other half — it runs on an x86
host with a GPU and the D-Robotics OE-LLM toolchain, and records how each model
was turned into a package BLLM consumes.

It is the LLM sibling of [bcdl-model-zoo](https://github.com/ruisv/bcdl-model-zoo)
(computer-vision conversion recipes), and follows the same shape.

## What this repo ships

**Recipes and acceptance, not binaries.** Each model records the toolchain
version, the quantisation, the acceptance thresholds, and the **on-board measured
numbers** — enough to know what a correct build is and to attribute a future
regression. The compiled `.hbm` files are GB-scale and not committed
(`.gitignore` excludes `*.hbm`).

The full offline conversion source (the native leap-graph construction) is the
proprietary part of the runtime and is **not** published here — the same
boundary [BLLM's own repo](https://github.com/ruisv/bllm) draws (its
`host_toolchain/` is private). What this repo publishes is the **methodology**
(`CONVERSION.md`), the **per-model config + acceptance** (`models/<name>/`), and
the **verification harness** (`common/`, reusing BLLM's public test scripts).

**The model weights and the `.hbm` compiled from them follow their own upstream
licences.** A compiled `.hbm` is a derivative of upstream weights.

| model | upstream | licence |
|---|---|---|
| Qwen2.5, Qwen3.5, Qwen2.5-Omni | Alibaba (Qwen) | Apache-2.0 (check per model card) |
| DeepSeek-R1-Distill-Qwen | DeepSeek | derived — check terms |
| InternLM2 | Shanghai AI Lab | Apache-2.0 (check) |
| GLM-Edge | Zhipu | check terms |
| Phi-4-mini | Microsoft | MIT (check) |

## Catalogue

| model | arch | origin | recipe | on-board (S100P) |
|---|---|---|---|---|
| **Qwen3.5-0.8B** | hybrid (Gated-DeltaNet SSM) | native (BLLM-only) | complete | 21.5 tok/s · 100% BPU int8 |
| **Qwen3.5-2B** | hybrid | native | complete | 13.20 tok/s (ctx4k) · 15.29 tok/s (ctx512) |
| **Qwen3.5-4B** | hybrid, decoupled GDN heads | native | complete | 6.14 tok/s (ctx4k) · 7.54 tok/s (ctx512) |
| **Phi-4-mini** | dense | native | complete | 8.7 tok/s · remap+phi format |
| GLM-Edge | dense | native | planned | — |
| Qwen2.5-1.5B / 7B | dense | oellm-official | recipe | 24.5 / 6.4 tok/s |
| Qwen2.5-Omni-3B | omni (text+image+audio+video) | oellm-official | recipe | 14 tok/s |
| **Qwen3.5-VLM (0.8B/2B/4B, 320/448px)** | hybrid + vision | native | complete (S100P); recipe (S600, load-verified) | TTFT 1.23 s (0.8B/320px) |

- **origin** — `native` = self-converted (an architecture the official OE-LLM
  toolchain doesn't cover; the value of this repo). `oellm-official` = the
  official pre-compiled `.hbm` runs on BLLM's native runtime unchanged.
- **recipe status** — `complete` = the on-board three-part acceptance (below)
  passed. `recipe` = the conversion is recorded but the full acceptance has not
  been re-run for this release (e.g. compiled and load-verified on hardware, but
  end-to-end `chat()` not yet re-run — see the specific model's `expected.json`
  for exactly what is and is not verified). `planned` = not yet converted.

Only models BLLM's native runtime **loads and has verified on the board** are
listed. A model that merely compiles but was never run on-board is not a recipe —
it is an unverified claim.

## Getting the compiled models

Two ways to get a `bllm.load()`-ready package:

1. **Build from the recipe** — needs the x86 convert host + OE-LLM toolchain.
   See `CONVERSION.md` and each `models/<name>/`.
2. **Download an A-tier package** — permissively-licensed models are published on
   Hugging Face as `bllm.load()`-ready directories (`model.hbm` + `tokenizer.json` +
   `model.json`, hybrid also `embed_tokens.bin`, VLM also `visual.hbm`). Copyleft /
   non-commercial weights are never redistributed as binaries — only the recipe.

   | model | Hugging Face |
   |---|---|
   | Qwen3.5-2B (VLM, S100P, ctx4k/ctx512) | [ruisv/bllm-qwen3.5-2b](https://huggingface.co/ruisv/bllm-qwen3.5-2b) |
   | Qwen3.5-4B (VLM, S100P, ctx4k/ctx512) | [ruisv/bllm-qwen3.5-4b](https://huggingface.co/ruisv/bllm-qwen3.5-4b) |

   Each repo's `s100p/ctx4k/` or `s100p/ctx512/` subfolder is a self-contained,
   ready-to-load package — download the one you want and point `bllm.load()` at it.
   0.8B is not yet published this way; build it from the recipe in the meantime.

Then on the board:

```bash
conda install -c https://mirrors.ruis.ai/conda -c conda-forge bllm
python -c "import bllm; print(bllm.load('<dir>').chat('你好'))"
```

## The acceptance contract (why it is not cosine)

An LLM is not accepted on a cosine number. The gate is, in decreasing strictness:

1. **Prefill↔decode parity** — the chunked-prefill graph and one-token-at-a-time
   decode must produce the same first token and an identical greedy continuation
   (`common/eval_parity.py`, reusing BLLM's `check_prefill_parity.py`). A vacuous
   gate — prompt shorter than the chunk — is asserted against.
2. **Perplexity** — teacher-forced, on held-out text (`common/eval_ppl.py`).
3. **Task coherence anchors** — EN/ZH factual, arithmetic, multi-turn memory,
   thinking on/off. Recorded in `expected.json.task_metric`.

Cosine vs the fp32 reference is a *smoke test* of "is the graph still intact",
not a delivery gate — a model with cosine 0.982 can be perfectly usable and a
model with cosine 0.9998 can loop; the retrieval/PPL/task number is what decides.

## Layout

```
common/
  eval_parity.py    # prefill↔decode parity (wraps BLLM's check_prefill_parity.py)
  eval_ppl.py       # teacher-forced perplexity via NativeLlm.perplexity
  make_dir.py       # the ONLY way a model.json is written (calls bllm-make-model-dir)
models/<name>/
  README.md         # what goes wrong on this model, and which build is correct
  config.yaml       # compile knobs: march / cache_len / prefill chunk / quant / bpu_cores
  expected.json     # acceptance thresholds + on-board measured + rejected_builds
```

## The rule that exists because it was learned the hard way

**A wrong LLM build compiles cleanly, loads, and runs at full speed — then
degrades silently.** A mis-set eos never stops; a mis-set `mrope` turns images
into noise while text stays fine; an under-ranged activation clips a deep layer
and the model loops on the prompt. So every `models/<name>/README.md` records the
**wrong** builds (`expected.json.rejected_builds`), not just the right one — the
name of a build (`_int8` / `_ctx4k` / `_calib`) is which version it is, the only
anchor that separates two graphs that look identical.

## Relationship to the official RDK Model Zoo

Complementary. The [official rdk_model_zoo](https://github.com/D-Robotics/rdk_model_zoo)
ships pre-compiled vision models and (via OE-LLM) a set of official LLMs; those
run on BLLM's native runtime unchanged (`origin=oellm-official` above). This repo
covers the architectures the official toolchain does **not** — the hybrid-SSM
Qwen3.5 line, GLM-Edge, Phi-4-mini — plus the acceptance methodology for driving
any `.hbm` on the native runtime.
