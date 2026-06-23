"""
data.py — data for the smoke test.

Two sources, two helpers:
  * addition  : on-the-fly integer-addition batches (depth-scaling carry task). This is
                the from-scratch task that actually exercises the loop.
  * transcripts: char-level next-token batches built from a transcripts.jsonl produced by
                ../experiments/gen_math_transcripts.py (the SFT pass).
"""

import json
import torch

# --------------------------------------------------------------------------- #
# Addition task:  a(nd) '+' b(nd) '=' c(nd+1, reversed)   — score answer tokens only
# --------------------------------------------------------------------------- #
ADD_VOCAB = {c: i for i, c in enumerate("0123456789+=")}
ADD_V = len(ADD_VOCAB)


def make_addition_batch(bsz, nd, device="cpu"):
    """Return (inp, tgt, ans_mask) for bsz nd-digit addition problems."""
    a = torch.randint(0, 10 ** nd, (bsz,))
    b = torch.randint(0, 10 ** nd, (bsz,))
    s = a + b
    seqs = []
    for ai, bi, si in zip(a.tolist(), b.tolist(), s.tolist()):
        astr, bstr = str(ai).zfill(nd), str(bi).zfill(nd)
        cstr = str(si).zfill(nd + 1)[::-1]
        seqs.append([ADD_VOCAB[ch] for ch in (astr + "+" + bstr + "=" + cstr)])
    full = torch.tensor(seqs, dtype=torch.long)
    inp, tgt = full[:, :-1].to(device), full[:, 1:].to(device)
    ans_mask = torch.zeros_like(tgt, dtype=torch.float)
    ans_mask[:, -(nd + 1):] = 1.0                     # last nd+1 positions = the answer
    return inp, tgt, ans_mask


# --------------------------------------------------------------------------- #
# Transcript SFT: char-level LM over "question \n solution"
# --------------------------------------------------------------------------- #
def load_transcript_texts(path):
    texts = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            texts.append(d["question"].strip() + "\n" + d["solution"].strip() + "\n")
    return texts


def build_char_vocab(texts):
    chars = sorted(set("".join(texts)))
    stoi = {c: i for i, c in enumerate(chars)}
    return stoi, len(stoi)


class CharLM:
    """Holds a concatenated char stream and serves random fixed-length next-token batches."""
    def __init__(self, texts, stoi, block_size=64, device="cpu"):
        self.block_size = block_size
        self.device = device
        stream = []
        for t in texts:
            stream.extend(stoi[c] for c in t)
        self.data = torch.tensor(stream, dtype=torch.long)

    def batch(self, bsz):
        n = self.data.numel()
        ix = torch.randint(0, n - self.block_size - 1, (bsz,))
        x = torch.stack([self.data[i:i + self.block_size] for i in ix])
        y = torch.stack([self.data[i + 1:i + 1 + self.block_size] for i in ix])
        return x.to(self.device), y.to(self.device)
