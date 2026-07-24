#!/usr/bin/env python3
"""Teacher-forced perplexity — the acceptance number that a cosine cannot give.
A quantised LLM can hold cosine 0.999 and still drift in generation, or hold a
mediocre cosine and be perfectly usable; PPL on held-out text is the honest scalar.
Uses BLLM's own NativeLlm.perplexity (dense + hybrid). Run on the board.

    python common/eval_ppl.py --model DIR [--text FILE]

Records the value into expected.json.variants[].measured.ppl by hand after a run
(the repo does not auto-edit expected.json — a measured number is entered once,
with the board and date it came from).
"""
import argparse

# A small fixed English+code sample; swap --text for wikitext2 raw to match the
# public PPL convention when comparing across quantisations.
DEFAULT = (
    "The transformer architecture replaced recurrence with self-attention, letting "
    "every position attend to every other in one step. Linear-attention variants such "
    "as Gated-DeltaNet trade that quadratic mixing for a recurrent state, which is what "
    "lets them run in constant memory per token on a fixed KV budget. def add(a, b):\n"
    "    return a + b\n"
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--text", default=None, help="path to a text file; default is a builtin sample")
    a = ap.parse_args()
    import bllm
    text = open(a.text, encoding="utf-8").read() if a.text else DEFAULT
    s = bllm.load(a.model)
    ppl = s.perplexity(text)
    ntok = len(s.encode(text)) if hasattr(s, "encode") else 0
    print(f"model   {getattr(s, 'name', a.model)}")
    print(f"tokens  {ntok}")
    print(f"ppl     {ppl:.4f}   (lower is better; compare same text across quantisations)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
