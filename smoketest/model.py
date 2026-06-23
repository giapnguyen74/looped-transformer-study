"""
model.py — a compact char-level LOOPED transformer for the smoke test.

Same skeleton as experiments/loop_scaling_lab.py (RMSNorm pre-norm block, input
injection, loop-index embedding, Parcae negative-diagonal injection so rho(A) < 1),
but generic over an arbitrary vocab so it can train on BOTH:
  * the synthetic addition task (train_addition.py), and
  * raw transcript text for SFT (train_sft.py).

Kept tiny on purpose — this is a does-it-train smoke check, not a real model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.g = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * self.g


class Block(nn.Module):
    """Pre-norm attention + FFN (this is the residual R-bar applied each loop step)."""
    def __init__(self, dim, heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))
        self.n1, self.n2 = RMSNorm(dim), RMSNorm(dim)

    def forward(self, x, attn_mask):
        a = self.n1(x)
        x = x + self.attn(a, a, a, attn_mask=attn_mask, need_weights=False)[0]
        x = x + self.ffn(self.n2(x))
        return x


class Injection(nn.Module):
    """State-carry rule between loop steps. 'parcae' guarantees a contractive diagonal
    A-bar in (0,1) -> rho < 1 -> the loop is stable by construction."""
    def __init__(self, dim, mode="parcae"):
        super().__init__()
        self.mode = mode
        self.proj_e = nn.Linear(dim, dim, bias=False)
        if mode == "parcae":
            self.log_A = nn.Parameter(torch.zeros(dim))
            self.dt = nn.Parameter(torch.zeros(dim))

    def a_bar(self):
        return torch.exp(F.softplus(self.dt) * (-torch.exp(self.log_A)))   # in (0,1)

    def forward(self, h, e, block_out):
        if self.mode == "free":
            return h + self.proj_e(e) + block_out
        return self.a_bar() * h + self.proj_e(e) + block_out

    def rho(self):
        return 1.0 if self.mode == "free" else float(self.a_bar().max().item())


def loop_index_embedding(h, t, loop_dim, theta=10000.0):
    """Sinusoidal 'where am I in the loop' signal on the first loop_dim channels."""
    dev, dt = h.device, h.dtype
    freqs = 1.0 / (theta ** (torch.arange(0, loop_dim, 2, device=dev, dtype=dt) / loop_dim))
    ang = t * freqs
    emb = torch.cat([ang.sin(), ang.cos()], dim=-1)[:loop_dim]
    full = torch.zeros(h.shape[-1], device=dev, dtype=dt)
    full[:loop_dim] = emb
    return h + full.view(1, 1, -1)


class LoopedLM(nn.Module):
    """n_unique shared blocks looped n_loops times; effective depth = n_unique * n_loops."""
    def __init__(self, vocab, dim=64, heads=2, n_unique=1, n_loops=3, max_len=128, inj="parcae"):
        super().__init__()
        self.dim, self.n_loops = dim, n_loops
        self.tok = nn.Embedding(vocab, dim)
        self.pos = nn.Embedding(max_len, dim)
        self.blocks = nn.ModuleList([Block(dim, heads) for _ in range(n_unique)])
        self.inj = Injection(dim, inj)
        self.norm = RMSNorm(dim)
        self.head = nn.Linear(dim, vocab)
        self.loop_dim = max(2, dim // 8)
        self._mask = {}

    def causal_mask(self, T, device):
        if T not in self._mask:
            m = torch.full((T, T), float("-inf"), device=device)
            self._mask[T] = torch.triu(m, diagonal=1)
        return self._mask[T]

    def forward(self, ids, n_loops=None, return_resid=False):
        n_loops = n_loops or self.n_loops
        B, T = ids.shape
        mask = self.causal_mask(T, ids.device)
        e = self.tok(ids) + self.pos(torch.arange(T, device=ids.device))[None]
        h = e
        for t in range(n_loops):
            x = loop_index_embedding(self.norm(h + e), t, self.loop_dim)
            for blk in self.blocks:
                x = blk(x, mask)
            h = self.inj(h, e, x)
        logits = self.head(self.norm(h))
        if return_resid:
            return logits, float(h.norm(dim=-1).mean().item())
        return logits


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def pick_device(prefer="auto"):
    """Return the best available device: cuda > mps (Apple Silicon) > cpu.
    Pass an explicit name ('cuda'/'mps'/'cpu') to override auto-detection."""
    if prefer != "auto":
        return prefer
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"
