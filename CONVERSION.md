# Conversion methodology

How an LLM / VLM checkpoint becomes a `bllm.load()`-ready package. This is the
**method** — the quantisation decisions, the acceptance, the traps — not the graph
source (that stays private, see PUBLISHING.md). Two tracks, by origin.

Everything here runs on an x86 host with a GPU and the D-Robotics OE-LLM toolchain;
BLLM (the runtime) only consumes the result on the board.

---

## Track A — official architectures (`origin=oellm-official`)

Qwen2.5, DeepSeek-R1-Distill-Qwen, InternLM2, Qwen2.5-Omni. The official
`oellm_build` toolchain produces the `.hbm`; BLLM's native runtime drives it
unchanged. Here the "recipe" is small: the official pre-compiled `.hbm` is
downloaded, and `common/make_dir.py` wraps it into a package.

- The `.hbm` holds two dnn sub-models (prefill + decode) + a tokenizer config.
- The only per-model work is `model.json`: eos, chat format, cache_len — resolved
  by `bllm-make-model-dir`, never hand-written.
- **Omni trap**: `embed_tokens.bin` is a separate ~1.2 GB download whose filename
  carries no "Omni" — the BPU graph consumes an embedding, not a token id, so the
  package is broken without it even though the three `*_Omni_*.hbm` look complete.

## Track B — native (`origin=native`)

Architectures the official toolchain does not cover: the hybrid Gated-DeltaNet
(SSM) Qwen3.5 line, GLM-Edge, Phi-4-mini. These are assembled as a **direct leap
graph** (not the KV-cache-centric `oellm_build` framework) and driven by BLLM's
native hbDNN runtime. The graph construction is private; the reproducible decisions:

### Quantisation

- **Data-free int8** where it works: per-channel int8 weight (`const_fake_quant`)
  + per-token/​per-tensor int16 activation on the **static** `leap.linear` path.
  0.8B/2B fit flat activation ranges with no calibration set.
- **The static path is not optional for decode.** The dynamic `block_quantized_matmul`
  is a prefill op — 51 ms/matmul at M=1 — so decode on it is ~6 s/token. Same
  weights, 130× slower if the wrong matmul is used. (In `expected.json` as a
  rejected build.)
- **Per-linear activation calibration** when flat ranges clip: 4B's deep linears
  reach |input|~200 and the flat ±32/64 ranges make it echo the prompt in a loop.
  Record max|input|×1.3 per linear from a real fp32 decode forward. 0.8B/2B do not
  need this; 4B does. The trap is that a clipped model still compiles and runs —
  it just degenerates.
- **Keep in fp32**: SSM recurrence/state, depthwise conv, RMSNorms, the GDN
  gated-norm. Variance-through-int16 is where a RMSNorm silently dies (a deep
  massive-activation row rounds to 0 → `rsqrt(0+eps)`), the same failure the
  vision tower hits — see the qwen3.5-vlm recipe.
- **100% BPU**: gating ops (sigmoid/silu/softplus/exp) become int16 LUTs (`b30.lut`),
  and the attention softmax is hand-built (reduce_max→qexp→reduce_sum→div), so
  `remaining CPU native ops == {}`. Verify with `hrt_model_exec perf`
  (`CPU_inference_time_cost == 0`).

### The chunked prefill, and its two walls

A prefill graph turns per-token ingestion (weight-bandwidth-bound, ~46 ms/token)
into "one weight stream per N-token chunk". The GDN part needs the parallel
delta-rule (WY/UT transform) with a per-sub-chunk triangular inverse. Two things
were learned the hard way:

- **Numerical wall**: the inverse must be **forward-substitution** or
  **block-recursion**, not Newton doubling — Newton's `2I−AX` transiently
  overshoots and int16 clips it to garbage on deep layers (cos −0.34). The stable
  inverses keep every intermediate O(1) (cos 0.9998). Quantisation unit tests must
  use **real** q/k/v: random keys are near-orthogonal and hide the ill-conditioning
  that real correlated keys expose.
- **Compile wall**: putting the tiny `[C,C]` matmuls on the BPU makes hbcompile's
  global scheduling pass over the single fused segment go **serial** (`--jobs`
  idle) and super-linear in op count — hours. `max_time_per_fc` splits the segment
  into funccalls so the scheduler runs on small graphs and parallelises. The
  shipped default keeps `[C,C]` in float on the CPU (correct, image TTFT 2.30 s);
  all-BPU is a purity option, not a speed win (prefill is compute-bound).

### Acceptance (Track B)

Never cosine as a gate. `common/eval_parity.py` (prefill↔decode identity, prompt >
chunk) + `common/eval_ppl.py` + coherence anchors. Cosine vs fp32 is a smoke test
of "graph still intact" only.

---

## Traps that bite both tracks

- **GPU index ≠ CUDA index** on multi-GPU convert hosts — the calibration forward
  may land on the wrong card.
- **Host toolchain and board HBRT must match.** An `.hbm` compiled with a newer
  host HBDK can fail to load on an older board runtime; record both in
  `expected.json.toolchain`.
- **Performance mode is not sticky** on the board (`0x2b047000` drifts back) —
  re-write `0x99` and read it back before every benchmark, or you measure a
  throttled number.
- **A number without a board is a guess.** `expected.json.measured` is `null`
  until someone runs it on hardware.
