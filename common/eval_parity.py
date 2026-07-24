#!/usr/bin/env python3
"""Prefill↔decode parity — the strictest acceptance gate for a chunked-prefill
package. Runs the model with its prefill graph and again without, and requires the
first token and the greedy continuation to be identical. Reuses the exact logic of
BLLM's own scripts/check_prefill_parity.py; this wrapper just makes it a zoo gate
and asserts the prompt is longer than the chunk (a shorter prompt never exercises
the prefill graph — a green result that proves nothing).

    python common/eval_parity.py --with-prefill DIR_A --without DIR_B

DIR_A carries a prefill graph (model.json hbm_prefill); DIR_B is the same model
without it. Build DIR_B by dropping hbm_prefill from a copy of DIR_A's model.json.
Run on the board.
"""
import argparse

PROMPT = ("The history of computing spans many decades and countless people. " * 24
          + " In a single word, name the field just described.")
CONTROL = "The quick brown fox jumps over the lazy dog. " * 4


def run(path, max_new, topk):
    import bllm
    s = bllm.load(path)
    chunk = getattr(s, "prefill_chunk", None)
    s.set_sampling(temp=0.0, logprobs=topk)
    s.reset()
    text = s.chat(PROMPT, max_new=max_new)
    ntok = len(s.encode(PROMPT)) if hasattr(s, "encode") else 0
    return {"chunk": chunk, "text": text, "lps": s.last_logprobs(),
            "ppl": s.perplexity(CONTROL), "ntok": ntok}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-prefill", required=True)
    ap.add_argument("--without", required=True)
    ap.add_argument("--max-new", type=int, default=48)
    ap.add_argument("--topk", type=int, default=10)
    a = ap.parse_args()
    A = run(a.with_prefill, a.max_new, a.topk)
    B = run(a.without, a.max_new, a.topk)
    if not A["chunk"]:
        print("!! the with-prefill package reports no prefill graph"); return 2
    if A["ntok"] <= A["chunk"]:
        print(f"!! prompt {A['ntok']} tok <= chunk {A['chunk']}: VACUOUS"); return 2
    print(f"prompt {A['ntok']} tok > chunk {A['chunk']}: prefill chunk ran")
    ok = True
    if A["lps"] and B["lps"]:
        same = A["lps"][0]["id"] == B["lps"][0]["id"]
        print(f"[1] first token  {A['lps'][0]['id']} vs {B['lps'][0]['id']}  {'MATCH' if same else 'DIFFER'}")
        ok &= same
    print(f"[2] greedy {a.max_new} tok  {'IDENTICAL' if A['text']==B['text'] else 'DIVERGES'}")
    ok &= A["text"] == B["text"]
    print(f"[3] control ppl  {A['ppl']:.4f} vs {B['ppl']:.4f}  "
          f"{'MATCH' if abs(A['ppl']-B['ppl'])<1e-4 else 'DIFFER'}")
    print("=>", "PARITY OK" if ok else "PARITY FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
