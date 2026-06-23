"""
phase_b/model.py — looped transformer for the multi-hop KG game (Phase B).

The task is read-prompt → predict-one-answer (NOT autoregressive generation): the model reads
the fixed-length query `[BOS, head, rel_1..rel_K, QUERY]`, reasons over its PARAMETRIC knowledge
by iterating one shared block T times, and reads out the answer entity at the QUERY position.

Design (PROBLEM.md §6, summary/*):
  * full (bidirectional) self-attention — the prompt is read at once; depth comes from the loop,
    not from autoregression, so no causal mask.
  * one shared block looped T times = weight-tied depth; depth is a test-time dial.
  * Parcae-style contractive injection (rho<1) re-anchors to the input each loop (stability).
  * zero-init block output projections (Loop-Think) → identity map at init → stable under many
    unrollings.
  * loop-index embedding tells the block "which iteration am I on."
  * per-iteration readout (`per_iter=True`) exposes the answer decoded after each loop, which the
    trainer supervises toward the t-th hop (deep supervision: one loop = one hop).

Length generalization is NOT a concern here: the prompt length is fixed by `K_max`. The axis we
test is the loop count.
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


class Block(nn.Module):
    """Full self-attention + FFN, pre-norm. Output projections zero-init -> identity at start."""
    def __init__(self, dim, heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))
        self.n1, self.n2 = RMSNorm(dim), RMSNorm(dim)
        nn.init.zeros_(self.attn.out_proj.weight); nn.init.zeros_(self.attn.out_proj.bias)
        nn.init.zeros_(self.ffn[-1].weight); nn.init.zeros_(self.ffn[-1].bias)

    def forward(self, x):
        a = self.n1(x)
        x = x + self.attn(a, a, a, need_weights=False)[0]
        x = x + self.ffn(self.n2(x))
        return x


class Injection(nn.Module):
    """Parcae contractive carry: diagonal A-bar in (0,1) -> rho < 1, stable by construction."""
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


class LoopedKGReasoner(nn.Module):
    def __init__(self, vocab, max_len, dim=128, heads=4, n_unique=1, n_loops=6):
        super().__init__()
        self.n_loops = n_loops
        self.tok = nn.Embedding(vocab, dim)
        self.pos = nn.Embedding(max_len, dim)            # prompt length is fixed -> learned is fine
        self.blocks = nn.ModuleList([Block(dim, heads) for _ in range(n_unique)])
        self.inj = Injection(dim)
        self.norm = RMSNorm(dim)
        self.head = nn.Linear(dim, vocab)
        self.loop_dim = max(2, dim // 8)

    def _readout(self, h):
        """Decode the answer entity from the QUERY (last) position."""
        return self.head(self.norm(h[:, -1]))

    def forward(self, ids, n_loops=None, per_iter=False):
        n_loops = n_loops or self.n_loops
        T = ids.shape[1]
        e = self.tok(ids) + self.pos(torch.arange(T, device=ids.device))[None]
        h = e
        per = []
        for t in range(n_loops):
            x = loop_index_embedding(self.norm(h + e), t, self.loop_dim)
            for blk in self.blocks:
                x = blk(x)
            h = self.inj(h, e, x)
            if per_iter:
                per.append(self._readout(h))             # (B, vocab) after loop t+1
        if per_iter:
            return per                                   # list length n_loops
        return self._readout(h)


def n_params(m):
    return sum(p.numel() for p in m.parameters())
