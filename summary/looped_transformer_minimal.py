"""
Looped Recurrent-Depth Transformer — the algorithm in ~130 readable lines.

This is a teaching abstraction (Universal Transformer / OpenMythos style), not a
performance implementation. Every design choice maps to an intuition we built up:

  * ONE shared block, applied T times .......... "a single fold, reused"
  * residual + LayerNorm around the block ...... "fold GENTLY (Jacobian ~ I)
                                                  => forward stays bounded AND
                                                     gradients don't explode/vanish
                                                  => stable + trainable"
  * input injection every step (contractive) ... "re-anchor to the input;
                                                  forget your own transient scratch-work"
  * timestep embedding added every step ........ "tell the block WHERE it is in the loop,
                                                  so the same weights can play many roles"
  * per-token ACT halting ...................... "easy tokens stop early,
                                                  hard tokens keep folding"
  * MoE-FFN inside the block ................... "WHERE FACTS LIVE: many expert memories,
                                                  a router that branches per token"

THE WHOLE IDEA IN ONE SENTENCE:
    depth = how many times we fold;  one block = the fold;
    stability = folding without tearing;  halting = stop folding when done;
    MoE = the knowledge the folding operates on.

THREE INDEPENDENT DIALS:
    n_experts  -> how much it KNOWS      (parameters / stored knowledge)
    top_k      -> compute per token      (how many branches fire)
    max_steps  -> how deep it REASONS    (loop iterations)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# WHERE FACTS LIVE.                                                            #
# One Expert = one small FFN = one associative key-value memory:               #
#   gate/up matrices act as KEYS that detect patterns in a token,              #
#   the down matrix holds the VALUE written back into the residual stream.     #
# --------------------------------------------------------------------------- #
class Expert(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.gate = nn.Linear(dim, hidden, bias=False)   # keys (pattern detectors)
        self.up   = nn.Linear(dim, hidden, bias=False)
        self.down = nn.Linear(hidden, dim, bias=False)   # values (what to write back)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class MoEFFN(nn.Module):
    """Mixture-of-Experts (DeepSeekMoE style). Many memories + a router that,
    per token, BRANCHES to the top-k relevant ones. Sparse: lots of stored
    parameters (knowledge) but only k experts run per token (cheap compute).

      * routed experts -> specialize    (domain-specific knowledge, pick top-k)
      * shared experts -> always on     (common knowledge, so routed ones can specialize)
    """
    def __init__(self, dim, hidden, n_experts=8, top_k=2, n_shared=1):
        super().__init__()
        self.top_k  = top_k
        self.router = nn.Linear(dim, n_experts, bias=False)            # the BRANCH selector
        self.routed = nn.ModuleList([Expert(dim, hidden) for _ in range(n_experts)])
        self.shared = nn.ModuleList([Expert(dim, hidden) for _ in range(n_shared)])

    def forward(self, x):
        B, T, D = x.shape
        flat = x.reshape(B * T, D)

        # 1) router scores every expert, we keep only the top-k branches per token
        gates = F.softmax(self.router(flat), dim=-1)                   # (N, n_experts)
        gate, idx = gates.topk(self.top_k, dim=-1)                     # which experts, how much
        gate = gate / gate.sum(-1, keepdim=True)                       # renormalize the k weights

        # 2) run ONLY the chosen experts, blend by gate weight (sparse compute)
        out = torch.zeros_like(flat)
        for k in range(self.top_k):
            for e in range(len(self.routed)):
                m = idx[:, k] == e
                if m.any():
                    out[m] += gate[m, k:k + 1] * self.routed[e](flat[m])

        # 3) shared experts always fire -> common knowledge for every token
        for s in self.shared:
            out = out + s(flat)

        return out.reshape(B, T, D)


# --------------------------------------------------------------------------- #
# The single "fold": mix info across tokens (attention), then transform each   #
# token (MoE-FFN). Residual + norm keep each application near identity, which   #
# is what makes it safe to apply the SAME block many times.                    #
# --------------------------------------------------------------------------- #
class Block(nn.Module):
    def __init__(self, dim, heads, n_experts=8, top_k=2, n_shared=1):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ffn  = MoEFFN(dim, 4 * dim, n_experts, top_k, n_shared)  # transform step = the knowledge store
        self.n1   = nn.LayerNorm(dim)
        self.n2   = nn.LayerNorm(dim)

    def forward(self, h, mask=None):
        a = self.n1(h)
        h = h + self.attn(a, a, a, attn_mask=mask)[0]                 # fold 1: tokens talk to each other
        h = h + self.ffn(self.n2(h))                                  # fold 2: transform + fetch knowledge
        return h


# --------------------------------------------------------------------------- #
# The looped model: embed -> RECUR over depth (with halting) -> read out.       #
# --------------------------------------------------------------------------- #
class LoopedTransformer(nn.Module):
    def __init__(self, vocab, dim=256, heads=8, max_steps=8,
                 n_experts=8, top_k=2, n_shared=1):
        super().__init__()
        self.embed    = nn.Embedding(vocab, dim)
        self.block    = Block(dim, heads, n_experts, top_k, n_shared)  # <-- ONE block, shared across ALL steps
        self.step_emb = nn.Embedding(max_steps, dim)                   # "where am I in the loop"
        self.inject   = nn.Linear(dim, dim)                           # re-injects the frozen input each step
        self.halt     = nn.Linear(dim, 1)                            # per-token halting score
        self.norm     = nn.LayerNorm(dim)
        self.head     = nn.Linear(dim, vocab)
        self.max_steps = max_steps

    def stable_injection(self, e):
        # The injection is the term that gets iterated, so its "gain" must stay < 1
        # or the loop tears (residual explosion). Real code enforces this with a
        # spectral / negative-diagonal parameterization (LTI injection, Parcae).
        # tanh just caps the magnitude here as a stand-in for "keep it contractive".
        return torch.tanh(self.inject(e))

    def forward(self, ids, threshold=0.99):
        B, T = ids.shape
        e = self.embed(ids)        # encoded input -- kept FROZEN, re-injected every step
        h = e.clone()              # the state we iteratively refine

        # ---- ACT (Adaptive Computation Time) bookkeeping, PER TOKEN ----
        halting   = ids.new_zeros(B, T, dtype=torch.float)  # accumulated halt prob
        remainder = ids.new_zeros(B, T, dtype=torch.float)
        out       = torch.zeros_like(h)                     # committed (weighted) output

        for t in range(self.max_steps):
            still = (halting < 1.0).float()                 # 1 for tokens still looping

            # 1) refine: re-anchor to input + say which step this is, then apply THE fold
            h = h + self.stable_injection(e) + self.step_emb.weight[t]
            h = self.block(h)

            # 2) per-token halting decision (Graves-style ACT)
            p = torch.sigmoid(self.halt(h)).squeeze(-1)
            new_halted = ((halting + p * still) > threshold).float() * still
            keep_going = still * (1.0 - new_halted)

            halting   = halting + p * keep_going
            remainder = remainder + new_halted * (1.0 - halting)
            halting   = halting + new_halted * remainder

            # blend weight: p if still going, remainder if halting now, 0 if already halted (frozen)
            w   = (p * keep_going + new_halted * remainder).unsqueeze(-1)
            out = out + w * h

            if (halting >= 1.0).all():                      # everyone halted -> stop early
                break

        return self.head(self.norm(out))                    # logits: (B, T, vocab)


# --------------------------------------------------------------------------- #
# Sanity demo (won't run without torch installed, but shows the shapes).        #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    model  = LoopedTransformer(vocab=1000, dim=128, heads=4,
                               max_steps=8, n_experts=8, top_k=2, n_shared=1)
    ids    = torch.randint(0, 1000, (2, 16))    # (batch=2, seq_len=16)
    print("logits:", model(ids).shape)          # -> torch.Size([2, 16, 1000])

    # "Think longer" at inference is just: increase max_steps. Same weights.
    model.max_steps = 20
    print("logits (deeper):", model(ids).shape)
