"""
Usage:
    Directories:
        python train.py \
        --urban-train  data/urban/enriched/ \
        --urban-sem-npy data/urban/encoded/ \
        --maritime-train data/maritime/canonical/ \
        --maritime-sem-npy data/maritime/encoded/

    Explicit list:
        python train.py \\
            --urban-train   data/urban/train/     --urban-val   data/urban/val/ \\
            --maritime-train data/maritime/train/ --maritime-val data/maritime/val/ \\
            --urban-sem-npy      urban_train.npy  --urban-val-sem-npy      urban_val.npy \\
            --maritime-sem-npy   mar_train.npy    --maritime-val-sem-npy   mar_val.npy
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from itertools import chain

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from data import TrajectoryDataset, collate_fn
from masking import sample_mode, make_pos_mask
# from v1.train import _temporal_features
# from v3.train import _domain_ids, _kinematics_from_batch

from config import ModelConfig
from model import TrajectoryMaskedAutoEncoder

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)-8s %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)

# ========== Helper functions ==========

def _batch_domain_str(batch: dict) -> str:
    domain = batch.get('domain', 0)
    if isinstance(domain, torch.Tensor):
        return 'maritime' if int(domain[0].item()) == 1 else 'urban'
    if isinstance(domain, list):
        return 'maritime' if int(domain[0]) == 1 else 'urban'
    return 'maritime' if int(domain) == 1 else 'urban'

def _sem_target(e_sem: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
    """Mean-pool per-point semantic embs over non-padded positions"""
    valid = (~pad_mask).float().unsqueeze(-1) # (B, L, 1)
    return (e_sem * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)

def make_masks(batch: dict, mode: str, max_len: int,
               device: torch.device) -> tuple[torch.Tensor, bool, bool]:
    """
    Build per-sample position masks for the given batch mode
    Returns (pos_mask, kin_group_masked, sem_group_masked)
    Mode is applied uniformly to all samples in the batch,
    only positional mask varies per sample (different block offsets, etc.)
    """
    B = batch['x'].shape[0]
    coords_np = batch['coords'].numpy()
    pos_masks = np.zeros((B, max_len), dtype=bool)

    rng = np.random.default_rng()
    for b in range(B):
        traj_len = int(batch['traj_len'][b])
        pos_masks[b, :traj_len] = make_pos_mask(
            mode, traj_len, coords=coords_np[b, :traj_len], rng=rng
        )

    return (torch.from_numpy(pos_masks).to(device),
            mode == 'kinematic_group',
            mode == 'semantic_group')

def _temporal_features(batch: dict, device: torch.device) -> torch.Tensor:
    """Build tau = [DoW_norm, HoD_norm, MoH_norm, d_t_min_norm] from batch"""
    B, L, _ = batch['x'].shape
    tau = torch.zeros(B, L, 4, device=device)
    tau[..., 3] = batch['x'][..., 2].to(device)
    if 'tau' in batch and batch['tau'] is not None:
        tau = batch['tau'].to(device)
    return tau

def _domain_ids(batch: dict, device: torch.device) -> torch.Tensor:
    domain = batch.get('domain', 0)
    if isinstance(domain, torch.Tensor):
        return domain.to(device)
    if isinstance(domain, list):
        return torch.tensor(domain, dtype=torch.long, device=device)
    return torch.zeros(batch['x'].shape[0], dtype=torch.long, device=device)

def _kinematics(batch: dict, device: torch.device) -> torch.Tensor:
    """Return kinematic tensor (B, L, 3); zeros for urban if SOG/COG/ROT absent"""
    k = batch['kinematics']
    if k is not None:
        return k.to(device)
    B, L = batch['x'].shape[:2]
    return torch.zeros(B, L, 3, device=device)

# ========== Train/val loops ==========

def train_epoch(model: TrajectoryMaskedAutoEncoder, loader: DataLoader, optimizer: torch.optim.Optimizer,
                device: torch.device, cfg: ModelConfig, step: int, rng: np.random.Generator)-> tuple[float, float, float, int]:
    model.train()
    tot_loss = tot_rec = tot_sem = 0.0
    n_batches = 0
    for batch in loader:
        domain_str = _batch_domain_str(batch)
        mode = sample_mode(domain_str, rng)

        x_full = batch['x'].to(device)
        x_spatial = x_full[..., :2]
        tau = _temporal_features(batch, device)
        kin = _kinematics(batch, device)
        coords = batch['coords'].to(device)
        pad_mask = batch['pad_mask'].to(device)
        domain_ids = _domain_ids(batch, device)

        pos_mask, kin_masked, sem_masked = make_masks(batch, mode, cfg.max_len, device)

        e_sem = batch['e_sem']
        e_sem_target = None
        if e_sem is not None:
            e_sem = e_sem.to(device)
            if not sem_masked:
                e_sem_target = _sem_target(e_sem, pad_mask).detach()
        
        out = model.forward(
            x_spatial=x_spatial, tau=tau, kinematics=kin,
            coords=coords, pad_mask=pad_mask, pos_mask=pos_mask,
            domain_ids=domain_ids, e_sem=e_sem, kin_group_masked=kin_masked,
            sem_group_masked=sem_masked, e_sem_target=e_sem_target,
            alpha=cfg.sem_pred_alpha
        )

        loss = out['loss']
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        tot_loss += loss.item()
        tot_rec += out['loss_recovery'].item()
        tot_sem += out['loss_sem_pred'].item()
        n_batches += 1
        step += 1

        if step % cfg.log_every == 0:
            logger.info(
                '\tstep=%d mode=%-16s loss=%.8f rec=%.8f sem=%.8f',
                step, mode, loss.item(),
                out['loss_recovery'].item(), out['loss_sem_pred'].item()
            )
        
    n = max(n_batches, 1)
    return tot_loss / n, tot_rec / n, tot_sem / n, step

@torch.no_grad()
def val_epoch(model: TrajectoryMaskedAutoEncoder, loader: DataLoader, device: torch.device,
              cfg: ModelConfig, rng: np.random.Generator) -> tuple[float, float]:
    model.eval()
    tot_loss = tot_rec = 0.0
    n_batches = 0

    for batch in loader:
        domain_str = _batch_domain_str(batch)
        mode = sample_mode(domain_str, rng)

        x_full = batch['x'].to(device)
        x_spatial = x_full[..., :2]
        tau = _temporal_features(batch, device)
        kin = _kinematics(batch, device)
        coords = batch['coords'].to(device)
        pad_mask = batch['pad_mask'].to(device)
        domain_ids = _domain_ids(batch, device)

        pos_mask, kin_masked, sem_masked = make_masks(batch, mode, cfg.max_len, device)

        e_sem = batch['e_sem']
        e_sem_target = None
        if e_sem is not None:
            e_sem = e_sem.to(device)
            if not sem_masked:
                e_sem_target = _sem_target(e_sem, pad_mask).detach()

        out = model.forward(
            x_spatial=x_spatial, tau=tau, kinematics=kin,
            coords=coords, pad_mask=pad_mask, pos_mask=pos_mask,
            domain_ids=domain_ids, e_sem=e_sem, kin_group_masked=kin_masked,
            sem_group_masked=sem_masked, e_sem_target=e_sem_target,
            alpha=cfg.sem_pred_alpha
        )

        tot_loss += out['loss'].item()
        tot_rec += out['loss_recovery'].item()
        n_batches += 1

    n = max(n_batches, 1)
    return tot_loss / n, tot_rec / n

# ========== Main entry ==========

def _sem_base(p: Path, domain: str) -> str:
    """
    Return the base name used for the described_sem .npy file.

    Urban:    porto_enriched.parquet            -> porto_enriched
    Maritime: aisdk-2024-03-01_canonical.parquet -> aisdk-2024-03-01
    """
    stem = p.stem
    if domain == 'maritime':
        stem = stem.removesuffix('_canonical')
    return stem

def _match_sem_npys(
    parquet_paths: list[Path],
    sem_input: list[str] | None,
    domain: str = 'urban',
) -> list[Path | None]:
    """
    Resolve sem_npy_paths aligned to parquet_paths.

    Three calling modes:
      - None / empty list  -> all None (no semantics)
      - Single directory   -> auto-match: {sem_dir}/{base}_described_sem.npy
                              where base strips the domain stage suffix
                              (urban keeps full stem; maritime strips _canonical)
      - Explicit list      -> use as-is, converted to Path objects
    """
    if not sem_input:
        return [None] * len(parquet_paths)

    if len(sem_input) == 1 and Path(sem_input[0]).is_dir():
        sem_dir = Path(sem_input[0])
        result: list[Path | None] = []
        for p in parquet_paths:
            candidate = sem_dir / f"{_sem_base(p, domain)}_described_sem.npy"
            if candidate.exists():
                result.append(candidate)
            else:
                logger.warning("No sem .npy for %s (expected %s)", p.name, candidate)
                result.append(None)
        n_matched = sum(r is not None for r in result)
        logger.info('Sem .npy auto-match: %d/%d found in %s', n_matched, len(result), sem_dir)
        return result

    return [Path(s) for s in sem_input]

def build_loader(parquet_paths: list[Path], domain: str, sem_input: list[str] | None,
                 cfg: ModelConfig, shuffle: bool) -> DataLoader:
    split = 'train' if shuffle else 'val'
    logger.info('Building %s %s loader from %d parquet(s)', domain, split, len(parquet_paths))
    sem_npy_paths = _match_sem_npys(parquet_paths, sem_input, domain=domain)
    ds = TrajectoryDataset(
        parquet_paths,
        domain=domain,
        max_len=cfg.max_len,
        input_dim=cfg.input_dim,
        sem_npy_paths=sem_npy_paths,
        include_kinematics=True
    )
    batch_size = cfg.batch_size // 2
    n_batches = (len(ds) + batch_size - 1) // batch_size
    logger.info('\t%s %s: %d trajectories -> %d batches/epoch (batch_size=%d)',
                domain, split, len(ds), n_batches, batch_size)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=4 if shuffle else 2,
        pin_memory=True
    )

def train(cfg: ModelConfig, args: argparse.Namespace) -> None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info('Device: %s', device)

    def _paths(p: str | None) -> list[Path]:
        if not p:
            return []
        base = Path(p)
        return sorted(base.glob('**/*.parquet')) if base.is_dir() else [base]

    urban_train_loader = build_loader(
        _paths(args.urban_train), 'urban', args.urban_sem_npy or [], cfg, True
    )
    maritime_train_loader = build_loader(
        _paths(args.maritime_train), 'maritime', args.maritime_sem_npy or [], cfg, True
    )
    urban_val_loader = build_loader(
        _paths(args.urban_val), 'urban', args.urban_val_sem_npy or [], cfg, False
    )
    maritime_val_loader = build_loader(
        _paths(args.maritime_val), 'maritime', args.maritime_val_sem_npy or [], cfg, False
    )

    class InterleavedLoader:
        """Alternate urban and maritime batches"""
        def __init__(self, a: DataLoader, b: DataLoader):
            self.a = a
            self.b = b

        def __iter__(self):
            return chain.from_iterable(zip(self.a, self.b))
        
        def __len__(self):
            return min(len(self.a), len(self.b)) * 2

    train_loader = InterleavedLoader(urban_train_loader, maritime_train_loader)
    val_loader   = InterleavedLoader(urban_val_loader,   maritime_val_loader)
    logger.info('Train batches/epoch: %d  |  Val batches/epoch: %d',
                len(train_loader), len(val_loader))

    model = TrajectoryMaskedAutoEncoder(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info('Model: %s params  d_model=%d  n_layers=%d  n_heads=%d',
                f'{n_params:,}', cfg.d_model, cfg.n_layers, cfg.n_heads)
    logger.info('Training for %d epochs  lr=%.1e  batch_size=%d',
                cfg.epochs, cfg.lr, cfg.batch_size)
    
    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=cfg.lr_min)

    checkpoint_dir = Path(cfg.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng()
    best_val = float('inf')
    patience_count = 0
    step = 0

    for epoch in range(1, cfg.epochs + 1):
        train_loss, train_rec, train_sem, step = train_epoch(
            model, train_loader, optimizer, device, cfg, step, rng
        )
        val_loss, val_rec = val_epoch(model, val_loader, device, cfg, rng)
        scheduler.step()

        logger.info(
            'Epoch %3d/%d | train=%.8f (rec=%.8f sem=%.8f) | val=%.8f (rec=%.8f) | lr=%.2e',
            epoch, cfg.epochs, train_loss, train_rec, train_sem,
            val_loss, val_rec, scheduler.get_last_lr()[0]
        )

        if val_rec < best_val:
            best_val = val_rec
            patience_count = 0
            torch.save(
                {'epoch': epoch, 'model': model.state_dict(),
                 'val_rec': best_val, 'cfg': cfg},
                 checkpoint_dir / 'best.pt'
            )
            logger.info('\tNew best model saved to %s', checkpoint_dir)
        else:
            patience_count += 1
            if patience_count >= cfg.patience:
                logger.info('Early stopping at epoch %d', epoch)
                break
    
    logger.info('Model training complete. Best val recovery: %.8f', best_val)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--urban-train', default=None)
    parser.add_argument('--urban-val', default=None)
    parser.add_argument('--maritime-train', default=None)
    parser.add_argument('--maritime-val', default=None)
    parser.add_argument('--urban-sem-npy', nargs='*', default=None)
    parser.add_argument('--urban-val-sem-npy', nargs='*', default=None)
    parser.add_argument('--maritime-sem-npy', nargs='*', default=None)
    parser.add_argument('--maritime-val-sem-npy', nargs='*', default=None)

    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--d-model', type=int, default=128)
    parser.add_argument('--max-len', type=int, default=256)
    parser.add_argument('--alpha', type=float, default=0.05)
    parser.add_argument('--no-semantics', action='store_true')
    parser.add_argument('--checkpoint-dir', default='checkpoints')
    args = parser.parse_args()

    cfg = ModelConfig(
        d_model=args.d_model,
        max_len=args.max_len,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        sem_pred_alpha=args.alpha,
        use_semantics=not args.no_semantics,
        checkpoint_dir=args.checkpoint_dir,
        urban_train_sem_npy=args.urban_sem_npy or [],
        urban_val_sem_npy=args.urban_val_sem_npy or [],
        maritime_train_sem_npy=args.maritime_sem_npy or [],
        maritime_val_sem_npy=args.maritime_val_sem_npy or [],
    )
    train(cfg, args)

if __name__ == '__main__':
    main()