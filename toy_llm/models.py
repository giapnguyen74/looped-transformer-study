"""
toy_llm/models.py — two causal char-LMs for a parameter-matched comparison:

  * VanillaLM   — N distinct transformer layers (the baseline).
  * LoopedLM    — one k-layer block looped L times (effective depth = k*L, params = k layers).
                  inject=True → Parcae contractive injection (rho<1) + loop-index;
                  inject=False → bare looped (prior-RDM style, norm+residual only).

Same dim/heads/vocab so comparisons are apples-to-apples (reasoning_with_latent_thoughts's
iso-param vs iso-FLOP framing). Both are standard autoregressive next-token LMs (causal mask).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def pick_device(prefer="auto"):
    if prefer != "auto":
        return prefer
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


class RMSNorm(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.g = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * self.g


class CausalBlock(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))
        self.n1, self.n2 = RMSNorm(dim), RMSNorm(dim)

    def forward(self, x, mask):
        a = self.n1(x)
        x = x + self.attn(a, a, a, attn_mask=mask, need_weights=False)[0]
        x = x + self.ffn(self.n2(x))
        return x


class Injection(nn.Module):
    """Parcae contractive carry: diagonal A-bar in (0,1) -> rho<1."""
    def __init__(self, dim):
        super().__init__()
        self.proj_e = nn.Linear(dim, dim, bias=False)
        self.log_A = nn.Parameter(torch.zeros(dim))
        self.dt = nn.Parameter(torch.zeros(dim))

    def a_bar(self):
        return torch.exp(F.softplus(self.dt) * (-torch.exp(self.log_A)))

    def forward(self, h, e, block_out):
        return self.a_bar() * h + self.proj_e(e) + block_out

    def rho(self):
        return float(self.a_bar().max().item())


def loop_index_embedding(h, t, loop_dim, theta=10000.0):
    dev, dt = h.device, h.dtype
    freqs = 1.0 / (theta ** (torch.arange(0, loop_dim, 2, device=dev, dtype=dt) / loop_dim))
    ang = t * freqs
    emb = torch.cat([ang.sin(), ang.cos()], dim=-1)[:loop_dim]
    full = torch.zeros(h.shape[-1], device=dev, dtype=dt)
    full[:loop_dim] = emb
    return h + full.view(1, 1, -1)


def _causal_mask(T, device, cache={}):
    key = (T, device)
    if key not in cache:
        m = torch.triu(torch.full((T, T), float("-inf"), device=device), diagonal=1)
        cache[key] = m
    return cache[key]


class VanillaLM(nn.Module):
    def __init__(self, vocab, dim=256, heads=4, n_layers=8, max_len=256):
        super().__init__()
        self.tok = nn.Embedding(vocab, dim)
        self.pos = nn.Embedding(max_len, dim)
        self.blocks = nn.ModuleList([CausalBlock(dim, heads) for _ in range(n_layers)])
        self.norm = RMSNorm(dim)
        self.head = nn.Linear(dim, vocab)

    def forward(self, ids):
        B, T = ids.shape
        m = _causal_mask(T, ids.device)
        x = self.tok(ids) + self.pos(torch.arange(T, device=ids.device))[None]
        for blk in self.blocks:
            x = blk(x, m)
        return self.head(self.norm(x))


class LoopedLM(nn.Module):
    def __init__(self, vocab, dim=256, heads=4, n_unique=2, n_loops=4, max_len=256, inject=True):
        super().__init__()
        self.n_loops, self.inject = n_loops, inject
        self.tok = nn.Embedding(vocab, dim)
        self.pos = nn.Embedding(max_len, dim)
        self.blocks = nn.ModuleList([CausalBlock(dim, heads) for _ in range(n_unique)])
        self.inj = Injection(dim) if inject else None
        self.norm = RMSNorm(dim)
        self.head = nn.Linear(dim, vocab)
        self.loop_dim = max(2, dim // 8)

    def forward(self, ids, n_loops=None):
        n_loops = n_loops or self.n_loops
        B, T = ids.shape
        m = _causal_mask(T, ids.device)
        e = self.tok(ids) + self.pos(torch.arange(T, device=ids.device))[None]
        h = e
        for t in range(n_loops):
            if self.inject:
                x = loop_index_embedding(self.norm(h + e), t, self.loop_dim)
                for blk in self.blocks:
                    x = blk(x, m)
                h = self.inj(h, e, x)
            else:
                x = loop_index_embedding(self.norm(h), t, self.loop_dim)
                for blk in self.blocks:
                    x = blk(x, m)
                h = x
        return self.head(self.norm(h))


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def build(kind, vocab, dim, heads, max_len, layers=8, k=2, loops=4):
    """kind: vanilla | parcae | bare. Returns (model, label)."""
    if kind == "vanilla":
        return VanillaLM(vocab, dim, heads, n_layers=layers, max_len=max_len), f"vanilla-{layers}L"
    if kind == "parcae":
        return LoopedLM(vocab, dim, heads, k, loops, max_len, inject=True), f"parcae-{k}x{loops}"
    if kind == "bare":
        return LoopedLM(vocab, dim, heads, k, loops, max_len, inject=False), f"bare-{k}x{loops}"
    raise ValueError(kind)
