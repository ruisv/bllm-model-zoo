# Qwen3.5 image+text — a self-built vision tower on three decoder sizes

Qwen3.5 is natively **image-text-to-text**: the text-only packages in the
[qwen3.5-0.8b](../qwen3.5-0.8b/), [qwen3.5-2b](../qwen3.5-2b/) and
[qwen3.5-4b](../qwen3.5-4b/) recipes are a **build choice** (the vision half is
dropped at compile time), not a limitation of the checkpoint. The vision
weights ship in the same upstream checkpoint; this recipe is the vision tower
that BLLM's official Model Zoo counterpart does not cover, converted and
attached to each of the three decoder sizes.

Upstream: [Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) /
[-2B](https://huggingface.co/Qwen/Qwen3.5-2B) /
[-4B](https://huggingface.co/Qwen/Qwen3.5-4B), Apache-2.0 (check the model
cards). Each image+text package pairs this vision tower + a chunked-prefill
graph with the corresponding sibling recipe's text decoder `.hbm` — the
decoder itself is unchanged by adding vision.

| variant | decoder | bucket (vision tokens) | cache_len | board | TTFT | decode |
|---|---|---|---|---|---|---|
| `qwen3.5-0.8b-vlm320-ctx2k-int8-s100` | 0.8B | 320px (100 tok) | 2048 | S100P | 1.23 s | — |
| `qwen3.5-0.8b-vlm448-ctx2k-int8-s100` | 0.8B | 448px (196 tok) | 2048 | S100P | 2.30 s | — |
| `qwen3.5-2b-vlm320-ctx4k-int8-s100` | 2B | 320px (100 tok) | 4096 | S100P | 1.65 s | 13.20 tok/s |
| `qwen3.5-2b-vlm320-ctx512-int8-s100` | 2B | 320px (100 tok) | 512 | S100P | 1.51 s | 15.29 tok/s |
| `qwen3.5-4b-vlm320-ctx4k-int8-s100` | 4B | 320px (100 tok) | 4096 | S100P | 3.29 s | 6.14 tok/s |
| `qwen3.5-4b-vlm320-ctx512-int8-s100` | 4B | 320px (100 tok) | 512 | S100P | 2.93 s | 7.54 tok/s |
| `qwen3.5-2b-vlm320-ctx4k-int8-s600` | 2B | 320px (100 tok) | 4096 | S600 (`nash-p`) | — (see S600 section) | — |
| `qwen3.5-4b-vlm320-ctx4k-int8-s600` | 4B | 320px (100 tok) | 4096 | S600 (`nash-p`) | — (see S600 section) | — |

**2B/4B decode is measured against this release's GQA batched-matmul + mask-dedup
rewrite** (2026-07-29) — 4B is nearly 2x faster than the pre-rewrite baseline
(3.29 tok/s), the GQA path drops most of `expand_win`'s redundant DMA. 2B is
essentially unchanged (13.22 → 13.20), consistent with its board output being
verified bit-identical to the pre-rewrite build. See `../qwen3.5-2b/expected.json`
and `../qwen3.5-4b/expected.json` for the run-by-run numbers.

**320px is the recommended default bucket** across all three sizes — see
"Choosing a bucket" below. 224/448 are also legal; 336 is not (see
`config.yaml`'s note).

## What goes wrong

An image+text build compiles, loads, and answers *fluently* even when it is
wrong — a wrong build does not refuse to run, it produces a confident answer
about pixels it never actually received, or received distorted. Every one of
these was hit for real during conversion:

- **Copying Omni's vision settings onto a Qwen3.5 tower** — patch14 vs
  Qwen3.5's patch16, CLIP mean/std vs Qwen3.5's flat 0.5/0.5, contiguous mrope
  segments `[16,24,24]` vs Qwen3.5's interleaved `[11,11,10]`. All four are
  silently wrong: the compiled `.hbm` has no way to store or check them, so a
  mismatch is shape-correct garbage discovered only once an image is fed —
  never at compile time, never on a text-only smoke test.
- **The interleaved-mrope trap that only shows up with real (non-square)
  positions**: when `t == h == w` (true for a single image with a square grid
  at the token level the rope actually indexes), an interleaved layout and a
  naively-assumed contiguous layout produce **numerically identical** cos/sin
  tables — so a build with the wrong mrope layout can pass a single-image
  smoke test and still be wrong. It is only a multi-image or video-adjacent
  position pattern that diverges. Verify the mrope construction against the
  reference implementation directly (see "The vision tower" below), not
  against a single easy test case.
- **Flat/synthetic calibration for the vision tower** — see "Calibration"
  below; clips deep-MLP massive-activation channels into garbage, and a
  synthetic reference image under-represents how bad the real-photo case is.
- **336px bucket** — not a legal patch16×merge2 side length (336/16=21 is
  odd); compiles for other side lengths but 336 specifically must be rejected
  before compile, not discovered after.
- **4B's decoupled GDN heads carried into the prefill graph** — see the
  qwen3.5-4b recipe; this is a decoder-side trap that also affects the
  image+text package's prefill graph, since it shares the same recurrence.
- **S600 (`nash-p`) 4B prefill compile crash at default optimisation** — see
  "S600" below.

## The vision tower

Patch16 → 2×2 merge → a ViT-style transformer (12 blocks for the 0.8B/2B
tower; **4B's tower is deeper — 24 blocks**, since the merger output width
must match each decoder's own hidden size, not because a bigger decoder needs
a proportionally bigger tower for accuracy reasons alone). Image rows splice
into the decoder's token stream at `<|vision_start|>` / `<|vision_end|>`
exactly like the host-fed-embedding design already used for the official Omni
package — no `<|IMAGE|>` placeholder expansion is needed, since the host
already owns the embedding stream. Image tokens carry **3-D mrope positions**
`(t, h, w)`: the first image token is at `(t0, t0, t0)`, the last at
`(t0, t0+grid-1, t0+grid-1)`, and text resumes at `t0 + grid` — the same
scheme the official Omni package uses, unaffected by the interleaved-vs-
contiguous mrope-*section* question above (that is about how the rope
*frequency dimensions* are split among t/h/w, not about the position values
themselves).

Positional/embedding constants that depend only on the fixed bucket side
length (the learned position embedding's bilinear interpolation, and the 2-D
rope cos/sin table) are **folded into the graph offline** rather than
computed on-device — once a bucket is fixed, they are constants, and folding
them means the compiled graph has exactly **one runtime input**: the
patchified pixel buffer.

## Calibration (massive activation, not RMSNorm)

The a priori suspicion for a vision-tower quantisation failure is usually
RMSNorm's variance channel (a documented failure mode in adjacent embedding-
model work: a variance stored per-tensor in int16 rounds a massive-activation
row to zero, and `rsqrt(0 + eps)` degenerates the norm). That was **not** the
mechanism here — it was ruled out first, deliberately, before looking
elsewhere. The actual failure is **massive activation in the deep MLP
blocks**: a flat calibration range assumed across all blocks clips the last
few blocks' `fc1`/`fc2` activations, which run 2-4× larger than the earlier
blocks'. Per-linear calibration (COCO images, `max|input| * 1.3` margin) fixes
it.

Two calibration traps worth recording precisely:

1. **Recalibrate per bucket, not just per model size.** The massive-activation
   channel's real magnitude shifts with input resolution (measured: one
   specific channel read ~176 at the 224px bucket, ~325 at 448px, ~263 at
   320px — monotonic in neither direction the naive guess would predict).
   Reusing a smaller or larger bucket's calibration under-ranges and clips.
2. **Recalibrate per decoder size too** — see the qwen3.5-4b recipe's
   activation table; the 4B tower's own deep-MLP magnitudes are on a
   different scale than 0.8B/2B's, independent of the bucket question above.
3. **Gate on real photos, not synthetic/rendered reference images.** A
   synthetic gate image under-represents the true activation distribution: a
   build measured at cosine ~0.987 against a synthetic 224px reference (looks
   like a borderline fail) measured ~0.997 on a held-out real photo — the
   *build* was fine, the *gate image* was the misleading part. Once this was
   understood, board acceptance switched to held-out real photos exclusively.

## Choosing a bucket (320px is the recommended default)

Legal buckets are governed by `side % 32 == 0` (patch16 × merge2); the
practical choice is a tradeoff between TTFT (grows with vision-token count)
and description quality (a free-form "describe this image" answer at 224px
can be measurably less detailed than at 448px on the same photo, in ways a
narrow factual question like "what colour is the ground" does not surface).
320px (100 tokens, `20×20 -> merge -> 10×10`) sits at TTFT close to 224px
while matching 448px's answer quality on the coarser questions that most
usage is — measured on 0.8B: TTFT 1.23 s at 320px vs 2.30 s at 448px (1.86×),
with output identical on both narrow-fact questions and equivalent in detail
on a free-form description. **336px is not reachable** (see `config.yaml`).

## The chunked prefill's role here

Without a prefill graph, ingestion is weight-bandwidth-bound per token — a
single 448px image (196 vision rows) costs on the order of 10+ seconds before
the chunked-prefill graph existed, purely from streaming the decoder's
weights once per token. The chunked-prefill graph (same recipe as the
sibling text-only decoders — `N=32`, `C=16`, `[C,C]` intermediates kept float
on CPU) amortises that weight stream across a whole chunk instead of once per
token, which is what makes an interactive image TTFT possible at all. See the
sibling text recipes for the numerical/compile-wall story behind that design;
it is identical here, image tokens are just more rows through the same
pipeline as text tokens.

## S600 (`nash-p`)

The vision tower and the prefill graph both need a **separate recompile**
per `march` — an `.hbm` compiled for `nash-m` (S100/S100P) fails to load on
`nash-p` (S600) with a clean, non-destructive error
(`Model march incompatible! model march: nash-e, platform march: nash-p`) —
confirmed on real S600 hardware, not a guess. This is expected and matches
the same board-family boundary as everything else in this repo's `march`
knob; it is not specific to vision.

**The one S600-specific trap**: compiling the **4B prefill graph** for
`nash-p` at the toolchain's **default** hbcompile optimisation level crashes
natively — not a catchable exception, a hard abort from the compiler backend,
with the message `error: B30 VPU ops do not support circular buffer`. This is
NOT a blanket "nash-p doesn't support this op" fact: the identical prefill
graph code compiles for `nash-p` fine at 2B scale, and 4B compiles fine on
`nash-m` (S100P) at the default level too. It is specifically the combination
of {nash-p backend, this graph's shape at 4B scale, the default optimisation
level} that trips an unsupported code-generation path — `--probe`
(export+convert only, skipping the actual code generation step) passes clean
both before and after, confirming the graph's structure and quantisation are
not the problem. **Fix: recompile at a lower hbcompile optimisation level**
(the toolchain's `opt=0` setting) — same correctness (the same count of
CPU-fallback matmuls as the working `nash-m` build), compiles in around 46
minutes instead of crashing. 2B needed no such workaround at either march.

Board-level verification on S600 (real hardware, not emulated): both the
vision tower and the prefill graph, at both 2B and 4B, load cleanly via the
board's own model-inspection tool with the correct internal model name and no
march error. The **decoder graph's own raw speed** was measured directly on
real S600 hardware (`hrt_model_exec perf`, 4-core, 100% BPU, no CPU fallback):
33.5 ms/token (2B) and 97.3 ms/token (4B) — roughly 2.2x and 3.1x faster than
the same decoders' single-core S100P numbers (see the qwen3.5-2b/qwen3.5-4b
recipes), tracking the 4-core binding rather than anything vision-specific.

Full end-to-end `chat()` verification was blocked for a while by a genuinely
broken dependency — the `hobot-dnn-s600` conda package's `libhbtl.so` was
truncated (ELF section header table pointing past the end of the file;
confirmed with `readelf`, and confirmed **not** a local cache problem by
clearing the package cache and re-downloading fresh, which reproduced the
identical truncation byte-for-byte). That has since been fixed upstream (a
corrected `hobot-dnn-s600` package), which unblocked building a current
`bllm` with VLM support on S600 for the first time.

That in turn surfaced a real BLLM-side bug: `NativeVlm`'s constructor never
bound the text decoder to explicit BPU cores the way `NativeLlm` (text-only)
already did, so every S600 image `chat()` call crashed immediately with
`hbUCPSubmitTask` / "the backend must be specified to specific cores". Fixed.

**With that crash fixed, a second, still-unresolved problem showed up**:
S600 image `chat()` is intermittently wrong — roughly 40-50% of calls (temp=0,
fresh process and fresh `bllm.load()` each time) return empty/whitespace
output instead of a real reply, rather than the correct with-image answer.
The isolated vision tower alone is 100% deterministic (13/13 runs across two
batches, even when forced onto the decoder's multi-core BPU mask), and
text-only `chat()` (no image) is 100% deterministic (8/8) — only the combined
vision+text pipeline is flaky. An initial hypothesis (the vision tower
shouldn't share the decoder's explicit multi-core mask) looked like a fix in
an 8/8 test, but a larger, clean follow-up re-test (10 runs, no concurrent
sandbox load) showed the same ~40-50% failure rate regardless — that was
noise, not a fix. Root cause is unknown; suspected an SDK- or hardware-level
race switching BPU graphs across cores on this specific board/SDK
combination, but this is not confirmed.

**Do not treat S600 VLM `chat()` as reliable yet.** The decoder-only numbers
below are real hardware evidence of raw decode viability; they are not a
substitute for a trustworthy image-in, TTFT-out measurement, and none is
recorded here until the intermittent failure is understood.

## Deploying alongside another resident model

A vision-tower + prefill-graph package's total ION footprint (text decoder +
prefill graph + vision tower + embedding table, several GB combined at 2B/4B
scale) was verified to load and answer correctly on a board that also had a
**different model's serving process** running — stop the other process before
loading, the runtime does not support two BPU graphs' ION allocations
resident at once (this can take the board down without raising an exception,
so treat it as a hard constraint, not a soft one). See the qwen3.5-2b and
qwen3.5-4b recipes' equivalent note for the text-only case; the mechanism and
the caution are identical for image+text.

## Acceptance

The **strongest cheap correctness check for "did the pixels actually reach
the decoder"** is not cosine on the tower output — it is a same-question,
with-image vs without-image comparison at `temp=0`: an unconnected image tower
(a splice that never actually happens, despite the graph looking correct)
still produces a fluent, confident, *wrong* answer, and cosine on the tower's
own output cannot catch a splice failure downstream of the tower. A build
that changes its answer specifically when an image is fed, and answers a
narrow factual question about that image's actual content correctly, is
strong evidence the whole pipeline works end to end. Combine with:
`common/eval_parity.py` (prefill↔decode identity, same as the text-only
recipes, on a real image-bearing prompt so the image tokens are actually
inside the parity-checked window) and the tower's own cosine-vs-fp32 (a smoke
test that the graph is structurally intact, gated on **real photos**, never
synthetic ones — see "Calibration"). Numbers in `expected.json`.

## Build

The vision-tower and prefill-graph construction are the native leap-graph
toolchain (not published — see repo README). Given a compiled `visual.hbm` +
`hbm_prefill` (the chunked-prefill graph) alongside the paired sibling
recipe's `hbm` / `embed_tokens.bin` / `tokenizer.json`, `common/make_dir.py`
(wrapping `bllm-make-model-dir`) writes the `model.json` — including the four
un-defaultable vision settings from `config.yaml`'s `vision:` block — and
produces the `bllm.load()`-ready directory. Do not hand-write these four
settings even if they seem obvious; they are exactly the ones that fail
silently (see "What goes wrong").
