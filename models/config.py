from __future__ import annotations
from dataclasses import dataclass, field

@dataclass
class ModelConfig:
    input_dim: int = 6 # [d_lat, d_lon, d_t, speed_n, heading_n, turn_n]
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 6
    ffn_dim: int = 512
    dropout: float = 0.1
    max_len: int = 256
    fourier_embed_dim: int = 64
    sem_dim: int = 256
    use_semantics: bool = True
    no_sem_token: bool = True
    n_domains: int = 2

    # Recovery loss per-group weights
    loss_w_spatial: float = 5.0
    loss_w_temporal: float = 1.0
    loss_w_kin: float = 0.25

    # Sparse Cross-Domain Mixture of Experts replacing per-block FFN
    n_experts: int = 8          # C: total number of expert networks
    moe_top_k: int = 4          # K: active experts per token
    moe_lambda: float = 0.1    # load-balancing loss weight
    
    # Single-stage training with auxiliary semantic prediction head
    epochs: int = 50
    sem_pred_alpha: float = 0.05

    # Two-stage training with contrastive learning
    stage1_epochs: int = 15
    stage2_epochs: int = 35
    tau_init: float = 0.07
    contrastive_lambda: float = 0.1

    batch_size: int = 128
    num_workers: int = 8
    lr: float = 1e-3
    lr_min: float = 1e-5
    weight_decay: float = 1e-4
    patience: int = 10

    urban_train_parquets: list[str] = field(default_factory=list)
    urban_val_parquets: list[str] = field(default_factory=list)
    maritime_train_parquets: list[str] = field(default_factory=list)
    maritime_val_parquets: list[str] = field(default_factory=list)
    urban_train_sem_npy: list[str] = field(default_factory=list)
    urban_val_sem_npy: list[str] = field(default_factory=list)
    maritime_train_sem_npy: list[str] = field(default_factory=list)
    maritime_val_sem_npy: list[str] = field(default_factory=list)

    checkpoint_dir: str = 'checkpoints/'
    log_every: int = 100