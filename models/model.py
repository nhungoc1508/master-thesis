from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import ModelConfig

# ========== Fourier temporal encoder ==========

class FourierEncode(nn.Module):
    """
    Single-component learnable Fourier encoder
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        omega_init = torch.from_numpy(
            1 / 10 ** np.linspace(0, 9, embed_dim).astype('float32')
        )
        self.omega = nn.Parameter(omega_init)
        self.bias = nn.Parameter(torch.zeros(embed_dim))
        self.div_term = math.sqrt(1.0 / embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L) scalar feature -> (B, L, embed_dim)"""
        if x.dim() == 2:
            x = x.unsqueeze(-1) # (B, L, 1)
        encode = x * self.omega + self.bias # (B, L, embed_dim)
        return self.div_term * torch.cos(encode)
    
class FourierTemporalEncoder(nn.Module):
    """
    Map 4-D temporal feature tau_i = [DoW, HoH, MoH, d_t_norm] to R^{out_dim}
    Each component gets its own FourierEncode(embed_dim) module (separate
    learnable frequencies)
    Outputs are concatenated & projected: 4 * embed_dim -> out_dim
    """

    def __init__(self, n_components: int = 4, embed_dim: int = 64, out_dim: int = 64):
        super().__init__()
        self.encoders = nn.ModuleList(
            [FourierEncode(embed_dim) for _ in range(n_components)]
        )
        self.proj = nn.Sequential(
            nn.LeakyReLU(),
            nn.Linear(n_components * embed_dim, out_dim)
        )

    def forward(self, tau: torch.Tensor) -> torch.Tensor:
        """
        tau: (B, L, n_components) normalized temporal features -> (B, L, out_dim)
        """
        parts = [enc(tau[..., i]) for i, enc in enumerate(self.encoders)]
        feats = torch.cat(parts, dim=-1) # (B, L, n_components * embed_dim)
        return self.proj(feats) # (B, L, out_dim)

# ========== Sparse cross-domain mixture of experts ==========

class Expert(nn.Module):
    """One expert: 2-layer MLP with hidden dim 2*d & GELU"""

    def __init__(self, d_model: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class NoisyTopKRouter(nn.Module):
    """Top-K gating with noisy logits"""

    def __init__(self, d_model: int, n_experts: int, top_k: int):
        super().__init__()
        self.top_k = top_k
        self.n_experts = n_experts
        self.gate = nn.Linear(d_model, n_experts)
        self.noise = nn.Linear(d_model, n_experts)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x (N, d): flattened tokens

        Returns:
            sparse_gates (N, C): softmax gate values over the top-K experts (0 elsewhere) = g(x)
            topk_idx (N, K): indices of selected experts
            clean_logits (N, C): noise-free gate logits
            noisy_logits (N, C): gate logits actually used for routing
            noise_std (N, C): per-expert noise standard deviation (softplus)
        
        Formula:
            H(x)_i = (x \cdot W_g)_i + StandardNormal() \cdot Softplus((x \cdot W_noise)_i)
            G(x) = Softmax(KeepTopK(H(x), k))
        """
        clean_logits = self.gate(x) # x \cdot W_g
        noise_std = F.softplus(self.noise(x)) + 1e-2 # Softplus(x \cdot W_noise)
        if self.training:
            noisy_logits = clean_logits +  torch.randn_like(clean_logits) * noise_std
        else:
            noisy_logits = clean_logits
        
        topk_logits, topk_idx = noisy_logits.topk(self.top_k, dim=-1) # (N, K)
        sparse = torch.full_like(noisy_logits, float('-inf'))
        sparse.scatter_(-1, topk_idx, topk_logits)
        sparse_gates = F.softmax(sparse, dim=-1) # (N, C), 0 for non-top-K
        return sparse_gates, topk_idx, clean_logits, noisy_logits, noise_std
    
class SparseCrossDomainMoE(nn.Module):
    """
    Sparse Cross-Domain Mixture of Experts

    Replaces a single FFN with C experts. Each token is routed to its top-K
    experts and their outputs are combined by the gating weights
    Returns the mixed output plus a scalar load-balancing loss
    computed over valid tokens

    Formula: y = \sum_{i=1}^C G(x)_i \cdot E_i(x)

    The auxiliary balancing loss: L_aux = w \cdot (CV(Importance)^2 + CV(Load)^2)
        Importance(c) = \sum_x G_c(x) = batchwise sum of gate values
        Load(c) = \sum_x P(x, c) = smooth count of tokens routed to c
        CV() = std / mean = coefficient of variation
        P(x, c) = probability that expert c stays in/enters the top-K under
            a fresh draw of the gating noise
    """

    def __init__(self, d_model: int, hidden_dim: int, n_experts: int = 8,
                 top_k: int = 4, dropout: float = 0.1):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.router = NoisyTopKRouter(d_model, n_experts, top_k)
        self.experts = nn.ModuleList(
            [Expert(d_model, hidden_dim, dropout) for _ in range(n_experts)]
        )

    def _load(self, clean_logits: torch.Tensor, noisy_logits: torch.Tensor,
              noise_std: torch.Tensor) -> torch.Tensor:
        """
        Smooth per-expert load P(x, c) summed over tokens.
        """
        k = self.top_k
        # top-(K+1) of the noisy logits gives both thresholds
        top_vals, _ = noisy_logits.topk(k + 1, dim=-1)
        thr_kth = top_vals[..., k - 1:k]
        thr_k1th = top_vals[..., k:k + 1]

        # Is expert c currently in the top-K? (its noisy value >= K-th highest)
        is_in = noisy_logits >= thr_kth
        # Threshold to beat: (K+1)-th if already in, else K-th
        threshold = torch.where(is_in, thr_k1th, thr_kth)
        _NORMAL = torch.distributions.Normal(0.0, 1.0)
        prob = _NORMAL.cdf((clean_logits - threshold) / noise_std)
        return prob

    def forward(self, x: torch.Tensor, valid_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, L, d)
            valid_mask: (B, L) bool, True = real token (non-pad)

        Returns:
            output: (B, L, d)
            aux_loss: scalar
        """
        B, L, d = x.shape
        flat_x = x.reshape(-1, d)
        sparse_gates, topk_idx, clean_logits, noisy_logits, noise_std = self.router(flat_x)

        out = torch.zeros_like(flat_x)
        for c, expert in enumerate(self.experts):
            sel = (topk_idx == c).any(dim=-1)
            if sel.any():
                contrib = expert(flat_x[sel]) * sparse_gates[sel, c].unsqueeze(-1)
                out[sel] += contrib
        out = out.view(B, L, d)

        # Balancing loss over valid tokens
        if valid_mask is not None:
            vflat = valid_mask.reshape(-1)
        else:
            vflat = torch.ones(flat_x.size(0), dtype=torch.bool, device=x.device)
        vmask = vflat.unsqueeze(-1).float()

        importance = (sparse_gates * vmask).sum(dim=0)
        load = (self._load(clean_logits, noisy_logits, noise_std) * vmask).sum(dim=0)  # (C,)
        aux_loss = self._cv_squared(importance) + self._cv_squared(load)
        return out, aux_loss
    
    def _cv_squared(self, x: torch.Tensor, eps: float=1e-10) -> torch.Tensor:
        """Squared coefficient of variation: Var(x) / (Mean(x)^2); zero when uniform"""
        if x.numel() <= 1:
            return x.new_tensor(0.0)
        mean = x.mean()
        var = x.var(unbiased=False)
        return var / (mean ** 2 + eps)

# ========== Transformer with STPE & SCD-MoE ==========

class STPEMultiheadAttention(nn.Module):
    """
    Multi-head attention with spatio-temporal rotary positional encoding applied
    to rotate W_Q and W_K
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        n_freqs = self.d_head // 2

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        # Learnable spatial frequency projection (shared across heads)
        self.W_phi = nn.Parameter(torch.randn(n_freqs, 2) * 0.01)

    def forward(
        self,
        x: torch.Tensor,
        coords: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, L, _ = x.shape
        H, d_h = self.n_heads, self.d_head

        def _split(t: torch.Tensor) -> torch.Tensor:
            return t.view(B, L, H, d_h).transpose(1, 2)   # (B, H, L, d_h)

        q = _split(self.W_q(x))
        k = _split(self.W_k(x))
        v = _split(self.W_v(x))

        q, k = apply_spatial_rope(q, k, coords, self.W_phi)

        scale = math.sqrt(d_h)
        attn = torch.matmul(q, k.transpose(-2, -1)) / scale  # (B, H, L, L)

        if key_padding_mask is not None:
            # key_padding_mask: (B, L), True = ignore
            attn = attn.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)                          # (B, H, L, d_h)
        out = out.transpose(1, 2).contiguous().view(B, L, self.d_model)
        return self.W_o(out)
    
class DomainAwareSTPELayer(nn.Module):
    """
    Transformer layer with spatio-temporal rotary positional encoding (STPE)
    with per-domain spatial frequency offset + mixture of experts

    \phi_i = W_phi \cdot [x, y]^T + W_phi_dom[domain]
    """

    def __init__(self, d_model: int, n_heads: int, n_domains: int = 2,
                 n_experts: int = 8, top_k: int = 4, dropout: float = 0.1):
        super().__init__()
        d_h = d_model // n_heads
        n_freqs = d_h // 2
        self.n_heads = n_heads
        self.d_head = d_h

        self.attn = STPEMultiheadAttention(d_model, n_heads, dropout)

        # Per-domain spatial frequency bias
        self.W_phi_dom = nn.Embedding(n_domains, n_freqs)
        nn.init.zeros_(self.W_phi_dom.weight)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        # MoE replaces traditional FFN
        self.moe = SparseCrossDomainMoE(d_model, hidden_dim=2*d_model, n_experts=n_experts,
                                        top_k=top_k, dropout=dropout)
        # self.ffn = nn.Sequential(
        #     nn.Linear(d_model, ffn_dim),
        #     nn.GELU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(ffn_dim, d_model),
        #     nn.Dropout(dropout)
        # )
    
    def forward(self, x: torch.Tensor, coords: torch.Tensor, domain_ids: torch.Tensor,
                pad_mask: torch.Tensor | None = None) -> torch.Tensor:
        B, L, d = x.shape
        H, d_h = self.n_heads, self.d_head
        n_freqs = d_h // 2

        dom_offset = self.W_phi_dom(domain_ids)
        dom_coords_bias = dom_offset.unsqueeze(1).expand(B, L, n_freqs)

        # Manually apply attention with domain-aware rotation
        x_norm = self.norm1(x)
        q = self.attn.W_q(x_norm).view(B, L, H, d_h).transpose(1, 2)
        k = self.attn.W_k(x_norm).view(B, L, H, d_h).transpose(1, 2)
        v = self.attn.W_v(x_norm).view(B, L, H, d_h).transpose(1, 2)

        k_idx = torch.arange(n_freqs, dtype=torch.float32, device=x.device)
        theta = 10000.0 ** (-2 * k_idx / d_h)
        
        # Compute per-point spatial frequencies including domain offset
        phi_base = coords @ self.attn.W_phi.T
        phi = (phi_base + dom_coords_bias) * theta

        phi_inter = torch.repeat_interleave(phi, 2, dim=-1).unsqueeze(1)
        cos_phi = torch.cos(phi_inter)
        sin_phi = torch.sin(phi_inter)

        q_rot = q * cos_phi + _rotate_half(q) * sin_phi
        k_rot = k * cos_phi + _rotate_half(k) * sin_phi

        scale = math.sqrt(d_h)
        attn_w = torch.matmul(q_rot, k_rot.transpose(-2, -1)) / scale
        if pad_mask is not None:
            attn_w = attn_w.masked_fill(pad_mask.unsqueeze(1).unsqueeze(2), float('-inf'))
        attn_w = F.softmax(attn_w, dim=-1)
        attn_w = self.attn.dropout(attn_w)

        out = torch.matmul(attn_w, v).transpose(1, 2).contiguous().view(B, L, d)
        out = self.attn.W_o(out)
        x = x + out

        # x = x + self.ffn(self.norm2(x))
        # Moe block
        valid_mask = (~pad_mask) if pad_mask is not None else None
        moe_out, lb_loss = self.moe(self.norm2(x), valid_mask=valid_mask)
        x = x + moe_out
        return x, lb_loss

# ========== Semantic cross-attention ==========

class SemanticCrossAttention(nn.Module):
    """
    Cross-attention block g: H = CrossAttn(Q=Z, K=E_sem, V=E_sem) + Z

    Z & E_sem have same seq length (one emb/trajectory point)
    """

    def __init__(self, d_model: int, n_heads: int, sem_dim: int, dropout: float = 0.1):
        super().__init__()
        self.W_sem = nn.Linear(sem_dim, d_model, bias=False)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, z: torch.Tensor, e_sem: torch.Tensor,
                pad_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        z: (B, L, d)
        e_sem: (B, L, sem_dim); precomputed semantic embeddings
        pad_mask: (B, L) bool, True = padding
        """
        e = self.W_sem(e_sem) # (B, L, d_model)
        z_norm = self.norm(z)
        h, _ = self.cross_attn(
            query=z_norm,
            key=e,
            value=e,
            key_padding_mask=pad_mask
        )
        return z + self.dropout(h)

class TrajectoryMaskedAutoEncoder(nn.Module):
    """
    Main model
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model

        # Spatial embedding: [d_lat, d_lon] -> d/2
        self.W_s = nn.Linear(2, d // 2, bias=False)
        
        # Temporal embedding: 4-D tau -> d/2
        self.fourier_enc = FourierTemporalEncoder(4, cfg.fourier_embed_dim, d // 2)

        # Domain embedding: 2-D -> d
        self.E_dom = nn.Embedding(cfg.n_domains, d)

        # [KIN_UNK] token: replace 3 kinematic features when masked/unavailable
        # Masking scenario 1: pos_mask -> along with spatial, temporal, domain ID
        # Masking scenario 2: kin_group_masked = True
        self.kin_unk = nn.Parameter(torch.zeros(3))

        # Input projection from 6-D
        # Project spatial+temporal to d, then add kinematics separately after
        # projecting via W_kin: 3 -> d
        self.W_kin = nn.Linear(3, d, bias=False)

        # Domain-aware STRPE layers + MoE
        self.layers = nn.ModuleList([
            DomainAwareSTPELayer(
                d, cfg.n_heads, cfg.n_domains,
                n_experts=cfg.n_experts, top_k=cfg.moe_top_k,
                dropout=cfg.dropout)
            for _ in range(cfg.n_layers)
        ])
        self.norm = nn.LayerNorm(d)

        # Semantic cross-attention
        if cfg.use_semantics:
            self.g = SemanticCrossAttention(d, cfg.n_heads, cfg.sem_dim, cfg.dropout)
        else:
            self.g = None
        if cfg.no_sem_token:
            # [NO_SEM] token: replace semantic embedding when masked/unavailable
            # Masking scenario 1: no semantics available
            # Masking scenario 2: sem_group_masked = True
            # Masking scenario 3: pos_mask -> semantics need to be masked before cross attention
            self.no_sem = nn.Parameter(torch.zeros(cfg.sem_dim))
        
        # RoPE: [MASK_COORD] token to replace coords when whole point is masked
        self.mask_coords = nn.Parameter(torch.zeros(2))
        
        # Contrastive components
        self.W_traj_sem = nn.Linear(d, d, bias=False)
        self.log_tau = nn.Parameter(torch.tensor(cfg.tau_init).log())

        # Output head: d -> 6
        self.output_head = nn.Linear(d, cfg.input_dim)

        # Learned [MASK] token
        self.mask_token = nn.Parameter(torch.randn(d) * 0.02)

        self._layer_norm_emb = nn.LayerNorm(d)

        # Auxiliary semantic prediction head (zero-initialised)
        # self.W_sem_pred = nn.Linear(cfg.d_model, cfg.sem_dim, bias=False)
        # nn.init.zeros_(self.W_sem_pred.weight)

    def _embed(self, x_spatial: torch.Tensor, tau: torch.Tensor, kinematics: torch.Tensor,
               domain_ids: torch.Tensor) -> torch.Tensor:
        e_s = self.W_s(x_spatial)
        e_t = self.fourier_enc(tau)
        e_st = torch.cat([e_s, e_t], dim=-1)

        # Add kinematic contribution
        e_kin = self.W_kin(kinematics)           # (B, L, d)
        e = e_st + e_kin

        # Domain embedding (broadcast over L)
        e_dom = self.E_dom(domain_ids).unsqueeze(1)  # (B, 1, d)
        e = self._layer_norm_emb(e + e_dom)

        return e
    
    def _apply_kin_unk(
        self, kinematics: torch.Tensor, kin_group_masked: bool
    ) -> torch.Tensor:
        """Replace kinematic features with [KIN_UNK] if group-masked."""
        if kin_group_masked:
            return self.kin_unk.view(1, 1, 3).expand_as(kinematics)
        return kinematics

    def _run_layers(self, z: torch.Tensor, coords: torch.Tensor, domain_ids: torch.Tensor,
                    pad_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run all MoE layers, accumulating mean load-balance loss"""
        lb_total = z.new_tensor(0.0)
        for layer in self.layers:
            z, lb = layer(z, coords, domain_ids, pad_mask=pad_mask)
            lb_total = lb_total + lb
        lb_total = lb_total / max(len(self.layers), 1)
        return self.norm(z), lb_total

    def encode(
        self,
        x_spatial: torch.Tensor,
        tau: torch.Tensor,
        kinematics: torch.Tensor,
        coords: torch.Tensor,
        pad_mask: torch.Tensor,
        domain_ids: torch.Tensor,
        e_sem: torch.Tensor | None = None,
        kin_group_masked: bool = False,
    ) -> torch.Tensor:
        B, L, _ = x_spatial.shape
        kin = self._apply_kin_unk(kinematics, kin_group_masked)
        e = self._embed(x_spatial, tau, kin, domain_ids)

        # z = e
        # for layer in self.layers:
        #     z = layer(z, coords, domain_ids, pad_mask=pad_mask)
        # z = self.norm(z)
        z, _ = self._run_layers(e, coords, domain_ids, pad_mask)

        if self.g is not None:
            if e_sem is None and hasattr(self, 'no_sem'):
                e_sem = self.no_sem.view(1, 1, -1).expand(B, L, -1)
            if e_sem is not None:
                z = self.g(z, e_sem, pad_mask=pad_mask)

        return z
    
    def trajectory_embedding(self, *args, **kwargs) -> torch.Tensor:
        z = self.encode(*args, **kwargs)
        pad_mask = kwargs.get('pad_mask', args[4])
        valid = (~pad_mask).float().unsqueeze(-1)
        return (z * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)
    
    def forward(
        self,
        x_spatial: torch.Tensor,           # (B, L, 2)
        tau: torch.Tensor,                  # (B, L, 4)
        kinematics: torch.Tensor,           # (B, L, 3)
        coords: torch.Tensor,               # (B, L, 2)
        pad_mask: torch.Tensor,             # (B, L) bool
        pos_mask: torch.Tensor,             # (B, L) bool
        domain_ids: torch.Tensor,           # (B,)   int
        e_sem: torch.Tensor | None,         # (B, L, sem_dim) or None
        kin_group_masked: bool = False,
        sem_group_masked: bool = False,
        e_sem_target: torch.Tensor | None = None,  # (B, sem_dim) detached
        alpha: float = 0.05,
    ) -> dict:
        """
        Single forward pass covering all 6 masking modes.

        When sem_group_masked=True:
          - E_sem is replaced with the broadcast [NO_SEM] token before g.
          - L_sem_pred is suppressed (no gradient from semantic prediction).

        When kin_group_masked=True:
          - All kinematic features are replaced with [KIN_UNK] (handled by
            _apply_kin_unk before the embeddings are built).
        """
        B, L = x_spatial.shape[:2]

        # Resolve semantic input for g
        if sem_group_masked or e_sem is None:
            if hasattr(self, 'no_sem'):
                e_sem_use = self.no_sem.view(1, 1, -1).expand(B, L, -1)
            else:
                e_sem_use = None
        else:
            e_sem_use = e_sem
            # Mask semantics depending on pos_mask
            if pos_mask is not None and hasattr(self, 'no_sem'):
                no_sem_tok = self.no_sem.view(1, 1, -1).expand(B, L, -1)
                e_sem_use = torch.where(pos_mask.unsqueeze(-1), no_sem_tok, e_sem_use)

        # Full forward (handles spatial masking, kin_unk, g cross-attention)

        # Target: [Δlat, Δlon, Δt_norm, speed_n, heading_n, turn_n]
        target = torch.cat([x_spatial, tau[..., 3:4], kinematics], dim=-1)
        # Mask kinematic group if needed
        kin = self._apply_kin_unk(kinematics, kin_group_masked)
        e = self._embed(x_spatial, tau, kin, domain_ids)
        
        if pos_mask is not None:
            # Mask whole position (spatial - tau - kin - domain)
            mask_tok = self.mask_token.view(1, 1, -1).expand(B, L, -1)
            e = torch.where(pos_mask.unsqueeze(-1), mask_tok, e)
            # Mask coordinates (for RoPE)
            mask_coords_tok = self.mask_coords.view(1, 1, 2).expand(B, L, 2)
            coords_input = torch.where(pos_mask.unsqueeze(-1), mask_coords_tok, coords)
        else:
            # If not masked, coords remains visible
            coords_input = coords

        # z = e
        # for layer in self.layers:
        #     z = layer(z, coords_input, domain_ids, pad_mask=pad_mask)
        # z = self.norm(z)
        z, lb_loss = self._run_layers(e, coords_input, domain_ids, pad_mask)

        if self.g is not None:
            if e_sem_use is not None:
                h = self.g(z, e_sem_use, pad_mask=pad_mask)
            else:
                h = z
        else:
            h = z

        pred = self.output_head(h)

        out = {'pred': pred, 'z': z, 'h': h, 'loss_balance': lb_loss}
        if pos_mask is not None:
            real_masked = pos_mask & ~pad_mask
            if real_masked.any():
                out['loss'] = ((pred[real_masked] - target[real_masked]) ** 2).mean()
            else:
                out['loss'] = torch.tensor(0.0, device=x_spatial.device)

        # loss_recovery = out['loss']
        # loss_sem_pred = torch.tensor(0.0, device=x_spatial.device)

        # # Semantic alignment auxiliary loss (modes 1–4 only)
        # if not sem_group_masked and e_sem_target is not None:
        #     h = out['h']   # (B, L, d) — output of g (or encoder if g=None)
        #     valid = (~pad_mask).float().unsqueeze(-1)              # (B, L, 1)
        #     h_pool = (h * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)  # (B, d)
        #     e_pred = self.W_sem_pred(h_pool)                       # (B, sem_dim)
        #     loss_sem_pred = F.mse_loss(e_pred, e_sem_target)

        # out['loss'] = loss_recovery + alpha * loss_sem_pred
        # out['loss_recovery'] = loss_recovery
        # out['loss_sem_pred'] = loss_sem_pred
        return out
    
    def semantic_trajectory_embedding(self, e_sem: torch.Tensor,
                                      pad_mask: torch.Tensor) -> torch.Tensor:
        e = self.g.W_sem(e_sem)
        valid = (~pad_mask).float().unsqueeze(-1)
        pooled = (e * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)
        return self.W_traj_sem(pooled)
    
    def info_nce(self, z_traj: torch.Tensor, e_traj: torch.Tensor,
                 detach_e: bool = False) -> torch.Tensor:
        if detach_e:
            e_traj = e_traj.detach()
        z = F.normalize(z_traj, dim=-1)
        e = F.normalize(e_traj, dim=-1)
        tau = self.log_tau.exp().clamp(0.01, 1.0)
        labels = torch.arange(z.size(0), device=z.device)
        loss_ze = F.cross_entropy((z @ e.T) / tau, labels)
        loss_ez = F.cross_entropy((e @ z.T) / tau, labels)
        return (loss_ze + loss_ez) / 2.0
    
    def forward_stage1(
            self,
            x_spatial: torch.Tensor,
            tau: torch.Tensor,
            kinematics: torch.Tensor,
            coords: torch.Tensor,
            pad_mask: torch.Tensor,
            domain_ids: torch.Tensor,
            e_sem: torch.Tensor
    ) -> dict:
        """Contrastive learning (trajectory - semantic)"""
        z_traj = self.trajectory_embedding(
            x_spatial, tau, kinematics, coords, pad_mask, domain_ids, e_sem=None
        )
        e_traj = self.semantic_trajectory_embedding(e_sem, pad_mask)
        loss = self.info_nce(z_traj, e_traj)
        return {'loss': loss, 'z_traj': z_traj, 'e_traj': e_traj}
    
    def forward_stage2(
            self,
            x_spatial: torch.Tensor,
            tau: torch.Tensor,
            kinematics: torch.Tensor,
            coords: torch.Tensor,
            pad_mask: torch.Tensor,
            pos_mask: torch.Tensor,
            domain_ids: torch.Tensor,
            e_sem: torch.Tensor | None,
            kin_group_masked: bool = False,
            e_traj_sem_detached: torch.Tensor | None = None
    ) -> dict:
        """Masking-recovery + soft contrastive regulariser + MoE load balancing"""
        out = self.forward(
            x_spatial, tau, kinematics, coords, pad_mask, pos_mask,
            domain_ids, e_sem, kin_group_masked
        )
        loss_recovery = out['loss']
        lb_loss = out['loss_balance']

        total = loss_recovery + self.cfg.moe_lambda * lb_loss

        if e_traj_sem_detached is not None:
            z_traj = self.trajectory_embedding(
                x_spatial, tau, kinematics, coords, pad_mask, domain_ids, e_sem
            )
            loss_ctr = self.info_nce(z_traj, e_traj_sem_detached, detach_e=True)
            total = total + self.cfg.contrastive_lambda * loss_ctr
            out['loss_contrastive'] = loss_ctr
        else:
            out['loss_contrastive'] = torch.tensor(0.0, device=x_spatial.device)

        out['loss'] = total
        out['loss_recovery'] = loss_recovery
        return out

def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate pairs: [x0, x1, x2, x3, ...] -> [-x1, x0, -x3, x2, ...]."""
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack([-x2, x1], dim=-1).flatten(-2)

def apply_spatial_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    coords: torch.Tensor,
    W_phi: nn.Parameter,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply Spatial Rotary Positional Encoding to Q and K.

    Args:
        q, k : (B, H, L, d_h) — query and key from projection
        coords: (B, L, 2) — [Δlat_norm, Δlon_norm] per point
        W_phi : (d_h//2, 2) — learnable spatial frequency projection

    Returns rotated (q, k) of same shape.
    """
    d_h = q.size(-1)
    n_freqs = d_h // 2

    # Compute per-point spatial frequencies: (B, L, n_freqs)
    phi = coords @ W_phi.T   # (B, L, 2) × (2, n_freqs) → (B, L, n_freqs)

    # Base frequencies θ_k = 10000^{-2k / d_h}  for k = 0..n_freqs-1
    k_idx = torch.arange(n_freqs, dtype=torch.float32, device=q.device)
    theta = 10000.0 ** (-2 * k_idx / d_h)          # (n_freqs,)
    phi = phi * theta                                # (B, L, n_freqs)

    # Interleave: (B, L, d_h) where pair 2k, 2k+1 share frequency phi_k
    phi_interleaved = torch.repeat_interleave(phi, 2, dim=-1)  # (B, L, d_h)

    cos_phi = torch.cos(phi_interleaved).unsqueeze(1)  # (B, 1, L, d_h)
    sin_phi = torch.sin(phi_interleaved).unsqueeze(1)  # (B, 1, L, d_h)

    q_rot = q * cos_phi + _rotate_half(q) * sin_phi
    k_rot = k * cos_phi + _rotate_half(k) * sin_phi
    return q_rot, k_rot