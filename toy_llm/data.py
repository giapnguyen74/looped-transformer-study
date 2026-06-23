"""
toy_llm/data.py — char/byte-level corpora for the comparison.

  shakespeare : Tiny Shakespeare (~1MB) — fast, but small enough that big models overfit.
  enwik8      : first 100MB of Wikipedia (byte-level, vocab 256), standard 90M/5M/5M split —
                large enough that the deep baseline is data-bound, not memorizing (fixes the
                overfit confound in the iso-depth comparison).

Data is stored as uint8 (ids 0..255) to keep enwik8 ~100MB in RAM; batches convert to long.
Stdlib + torch. Pass --data to point at a local file (skips download).
"""

import os
import urllib.request
import zipfile

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
SHAKES_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
ENWIK8_URL = "http://mattmahoney.net/dc/enwik8.zip"


def _get_shakespeare(path):
    path = path or os.path.join(HERE, "input.txt")
    if not os.path.exists(path):
        print(f"[data] downloading tiny shakespeare -> {path}")
        urllib.request.urlretrieve(SHAKES_URL, path)
    text = open(path, encoding="utf-8").read()
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.uint8)
    return data, len(chars), stoi


def _get_enwik8(path):
    """Return first 100MB of enwik8 as a uint8 tensor (byte-level, vocab=256)."""
    if path and os.path.exists(path):
        raw = open(path, "rb").read()
    else:
        zpath = os.path.join(HERE, "enwik8.zip")
        if not os.path.exists(zpath):
            print(f"[data] downloading enwik8 (~36MB zip) -> {zpath}")
            urllib.request.urlretrieve(ENWIK8_URL, zpath)
        with zipfile.ZipFile(zpath) as z:
            raw = z.read("enwik8")
    raw = raw[:100_000_000]
    data = torch.frombuffer(bytearray(raw), dtype=torch.uint8).clone()
    return data, 256, None


class CharData:
    def __init__(self, dataset, block, device, path=None, val_frac=0.1):
        self.dataset, self.block, self.device = dataset, block, device
        if dataset == "shakespeare":
            data, self.vocab, self.stoi = _get_shakespeare(path)
            n = int(len(data) * (1 - val_frac))
            self.train, self.val = data[:n], data[n:]
        elif dataset == "enwik8":
            data, self.vocab, self.stoi = _get_enwik8(path)
            self.train, self.val = data[:90_000_000], data[90_000_000:95_000_000]
        else:
            raise ValueError(dataset)
        print(f"[data] {dataset}: vocab={self.vocab} train={self.train.numel():,} "
              f"val={self.val.numel():,} chars")

    def batch(self, split, bsz, generator=None):
        d = self.train if split == "train" else self.val
        ix = torch.randint(0, d.numel() - self.block - 1, (bsz,), generator=generator)
        x = torch.stack([d[i:i + self.block] for i in ix]).long()
        y = torch.stack([d[i + 1:i + 1 + self.block] for i in ix]).long()
        return x.to(self.device), y.to(self.device)
