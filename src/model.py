"""Decoder-only Transformer with optional µP parameterization.

Architecture: nanoGPT-style — pre-LayerNorm, GELU MLP, learned positional embeddings,
weight-tied input/output embedding (SP only; µP keeps them untied so the readout
can use its µP-specific 1/d_model output multiplier).

µP details (following Yang et al. 2022, Tensor Programs V):
  - attention logits scaled by 1/d_head  (instead of 1/sqrt(d_head))
  - hidden weight init: N(0, 1/d_model)  (width-independent stddev under µP rule
    because per-step update size is normalized by the µP optimizer multipliers)
  - readout (lm_head) init: N(0, 1/d_model^2) and forward output multiplied by
    1/d_model (the "output multiplier")
  - per-tensor learning rate multipliers:
        embeddings, biases, LayerNorm: lr
        hidden weights:                lr / d_model_ratio
        readout weights:               lr / d_model_ratio
    where d_model_ratio = d_model / base_d_model (base = the proxy width
    on which the LR was tuned).
We expose param_groups(base_d_model, lr) so train.py can pass groups directly to
torch.optim.AdamW; this keeps the dependency on the `mup` package optional.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int
    seq_len: int = 1024
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    d_ff: int = 512
    dropout: float = 0.0
    mup: bool = False
    base_d_model: int = 128   # the width LR was tuned on (only used if mup=True)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.n_heads = cfg.n_heads
        self.d_head = cfg.d_model // cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=True)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=True)
        self.drop = nn.Dropout(cfg.dropout)
        self.mup = cfg.mup
        # 1/d (µP) vs 1/sqrt(d) (SP)
        self.attn_scale = (1.0 / self.d_head) if cfg.mup else (1.0 / math.sqrt(self.d_head))
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(cfg.seq_len, cfg.seq_len)).view(1, 1, cfg.seq_len, cfg.seq_len),
            persistent=False,
        )

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=-1)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * self.attn_scale
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.drop(att)
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.fc1 = nn.Linear(cfg.d_model, cfg.d_ff, bias=True)
        self.fc2 = nn.Linear(cfg.d_ff, cfg.d_model, bias=True)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.drop(self.fc2(F.gelu(self.fc1(x))))


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = MLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if not cfg.mup:
            # weight tying under SP
            self.lm_head.weight = self.tok_emb.weight
        self.output_mult = (1.0 / cfg.d_model) if cfg.mup else 1.0
        self._init_weights()

    def _init_weights(self):
        cfg = self.cfg
        d = cfg.d_model
        # embeddings: N(0, 1)  (µP rule for input layer; for SP this is also fine since tied)
        nn.init.normal_(self.tok_emb.weight, mean=0.0, std=1.0 if cfg.mup else 0.02)
        nn.init.normal_(self.pos_emb.weight, mean=0.0, std=1.0 if cfg.mup else 0.02)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                if m is self.lm_head:
                    if cfg.mup:
                        nn.init.normal_(m.weight, mean=0.0, std=1.0 / d)
                    else:
                        pass  # tied; init handled by tok_emb
                else:
                    if cfg.mup:
                        nn.init.normal_(m.weight, mean=0.0, std=1.0 / math.sqrt(d))
                    else:
                        nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        x = self.drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        logits = self.lm_head(x) * self.output_mult
        if targets is None:
            return logits, None
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100
        )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None,
                 eos_id=None):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.cfg.seq_len else idx[:, -self.cfg.seq_len:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = float("-inf")
            if top_p is not None:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                probs = F.softmax(sorted_logits, dim=-1)
                cdf = probs.cumsum(dim=-1)
                mask = cdf > top_p
                mask[..., 1:] = mask[..., :-1].clone()
                mask[..., 0] = False
                sorted_logits[mask] = float("-inf")
                logits = torch.full_like(logits, float("-inf")).scatter(
                    1, sorted_idx, sorted_logits
                )
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, nxt], dim=1)
            if eos_id is not None and (nxt == eos_id).all():
                break
        return idx

    # ----- µP / param groups -----
    def param_groups(self, lr: float, weight_decay: float):
        """Return AdamW param groups with µP per-tensor LR multipliers when mup=True.

        Under µP (following Yang et al. 2022 SGD/Adam adaptations for Adam):
          - readout (lm_head): lr scaled by 1 / width_mult
          - hidden matrix params (Linear .weight inside blocks): lr scaled by 1 / width_mult
          - embeddings, LayerNorm, biases: lr unchanged
        where width_mult = d_model / base_d_model.
        Weight decay: applied to 2D matrix params, skipped for 1D (norms/biases/embeddings).
        """
        cfg = self.cfg
        wm = (cfg.d_model / cfg.base_d_model) if cfg.mup else 1.0

        decay_hidden, nodecay_hidden = [], []
        decay_readout, nodecay_readout = [], []
        nodecay_emb_norm = []

        for name, p in self.named_parameters():
            if not p.requires_grad: continue
            is_2d = p.dim() >= 2
            if name.startswith("lm_head"):
                (decay_readout if is_2d else nodecay_readout).append(p)
            elif name.startswith(("tok_emb", "pos_emb")) or "ln" in name or name.endswith(".bias"):
                nodecay_emb_norm.append(p)
            else:
                (decay_hidden if is_2d else nodecay_hidden).append(p)

        groups = []
        # hidden (µP-scaled)
        if decay_hidden:
            groups.append(dict(params=decay_hidden, lr=lr / wm, weight_decay=weight_decay))
        if nodecay_hidden:
            groups.append(dict(params=nodecay_hidden, lr=lr / wm, weight_decay=0.0))
        # readout (µP-scaled)
        if decay_readout:
            groups.append(dict(params=decay_readout, lr=lr / wm, weight_decay=weight_decay))
        if nodecay_readout:
            groups.append(dict(params=nodecay_readout, lr=lr / wm, weight_decay=0.0))
        # embeddings / norms / biases (no µP scaling, no decay)
        if nodecay_emb_norm:
            groups.append(dict(params=nodecay_emb_norm, lr=lr, weight_decay=0.0))
        return groups

    def num_params(self, non_embedding=False):
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.tok_emb.weight.numel() + self.pos_emb.weight.numel()
        return n
