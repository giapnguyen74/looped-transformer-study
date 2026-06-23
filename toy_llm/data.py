"""
toy_llm/data.py — char-level Tiny Shakespeare (Karpathy), the canonical toy-LM corpus.
Downloads once to toy_llm/input.txt if absent; or pass --data <textfile>. Stdlib + torch.
"""

import os
import urllib.request

import torch

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
HERE = os.path.dirname(os.path.abspath(__file__))


def get_text(path=None):
    path = path or os.path.join(HERE, "input.txt")
    if not os.path.exists(path):
        print(f"[data] downloading tiny shakespeare -> {path}")
        urllib.request.urlretrieve(URL, path)
    return open(path, encoding="utf-8").read()


class CharData:
    def __init__(self, block, device, path=None, val_frac=0.1):
        text = get_text(path)
        chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.itos = {i: c for c, i in self.stoi.items()}
        self.vocab = len(chars)
        self.block = block
        self.device = device
        data = torch.tensor([self.stoi[c] for c in text], dtype=torch.long)
        n = int(len(data) * (1 - val_frac))
        self.train, self.val = data[:n], data[n:]

    def batch(self, split, bsz, generator=None):
        d = self.train if split == "train" else self.val
        ix = torch.randint(0, d.numel() - self.block - 1, (bsz,), generator=generator)
        x = torch.stack([d[i:i + self.block] for i in ix])
        y = torch.stack([d[i + 1:i + 1 + self.block] for i in ix])
        return x.to(self.device), y.to(self.device)
