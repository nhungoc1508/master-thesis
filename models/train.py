"""
Usage:
    Directories:
        python train.py \
        --urban-train  data/urban/enriched/ \
        --urban-sem-npy data/urban/encoded/ \
        --maritime-train data/maritime/canonical/ \
        --maritime-sem-npy data/maritime/encoded/ \
        --hf-repo nhungoc1508/tfm

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
import random
import sys
from pathlib import Path
from itertools import chain

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import dataclasses

sys.path.insert(0, str(Path(__file__).parent.parent))

from data import TrajectoryDataset, collate_fn
from masking import sample_mode, make_pos_mask

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

def _group_per_dim(per_dim: torch.Tensor) -> tuple[float, float, float]:
    """
    Group per-dimension recovery MSE [d_lat, d_lon, d_t, speed, heading, turn]
    into (spatial, temporal, kinematic) scalar MSEs for logging
    """
    pd = per_dim.detach().cpu()
    spatial = pd[:2].mean().item()
    temporal = pd[2].item() if pd.numel() > 2 else 0.0
    kinematic = pd[3:].mean().item() if pd.numel() > 3 else 0.0
    return spatial, temporal, kinematic

# ========== HuggingFace checkpoint backup ==========

def _ensure_hf_repo(repo_id: str) -> None:
    """Create the HuggingFace model repo if does not already exist"""
    try:
        from huggingface_hub import create_repo
        create_repo(repo_id, repo_type='model', private=True, exist_ok=True)
        logger.info('HuggingFace model repo ready: hf://%s', repo_id)
    except Exception as exc:
        logger.warning('Could not create/verify HF repo %s: %s', repo_id, exc)

def _hf_upload_checkpoint(local_path: Path, repo_id: str,
                          path_in_repo: str | None = None) -> None:
    """
    Upload a checkpoint file to a HuggingFace MODEL repo, overwriting any
    existing file at the same path. Failures are logged, never raised, so a
    network error cannot abort training.
    """
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        path_in_repo = path_in_repo or Path(local_path).name
        logger.info('Uploading %s to hf://%s', Path(local_path).name, repo_id)
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type='model',
        )
        logger.info('Upload complete.')
    except Exception as exc:
        logger.warning('HuggingFace upload failed (training continues): %s', exc)

# ========== Checkpoint resume ==========

def _rng_state() -> dict:
    """Snapshot all global RNG states for exact resume."""
    st = {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        st['cuda'] = torch.cuda.get_rng_state_all()
    return st

def _restore_rng(st: dict | None) -> None:
    if not st:
        return
    try:
        random.setstate(st['python'])
        np.random.set_state(st['numpy'])
        torch.set_rng_state(st['torch'])
        if 'cuda' in st and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(st['cuda'])
    except Exception as exc:
        logger.warning('Could not fully restore RNG state: %s', exc)

def _load_checkpoint(model: TrajectoryMaskedAutoEncoder, ckpt_path: str,
                     device: torch.device) -> dict:
    """Load model weights from a saved checkpoint and return the full ckpt dict."""
    path = Path(ckpt_path)
    if not path.exists():
        raise FileNotFoundError(f'Checkpoint not found: {path}')
    ckpt = torch.load(path, map_location=device, weights_only=False)
    state = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    logger.info('Loaded weights from %s (epoch=%s, stage=%s)', path.name,
                ckpt.get('epoch', '?') if isinstance(ckpt, dict) else '?',
                ckpt.get('stage', '?') if isinstance(ckpt, dict) else '?')
    if missing:
        logger.warning('\tmissing keys: %d (e.g. %s)', len(missing), missing[:3])
    if unexpected:
        logger.warning('\tunexpected keys: %d (e.g. %s)', len(unexpected), unexpected[:3])
    return ckpt if isinstance(ckpt, dict) else {'model': ckpt}

def _restore_train_state(optimizer, scheduler, resume_state: dict | None,
                         start_epoch: int, expected_stage: int,
                         fresh_optim: bool = False) -> tuple[float, int]:
    """
    Restore optimizer/scheduler/RNG from a resume checkpoint. Returns (best_val, step).
    Falls back to fast-forwarding the scheduler if optimizer/scheduler state is absent
    (e.g. an older checkpoint that only saved weights).

    fresh_optim=True is for extending training beyond the original horizon: keep the
    weights (+ step/best_val/RNG) but ignore the saved optimizer/scheduler. This avoids
    re-imposing the old, too-short cosine T_max (which is annealed to ~eta_min); instead
    the freshly-built longer scheduler is fast-forwarded by start_epoch so the LR resumes
    mid-cosine rather than at ~0.
    """
    best_val, step = float('inf'), 0
    if resume_state is None:
        return best_val, step

    ckpt_stage = resume_state.get('stage')
    if ckpt_stage is not None and ckpt_stage != expected_stage:
        logger.warning('Checkpoint stage=%s but resuming stage %d - optimizer state '
                       'may not correspond; continuing anyway.', ckpt_stage, expected_stage)

    if fresh_optim:
        logger.info('\tfresh-optim: ignoring saved optimizer/scheduler (extending schedule)')
        for _ in range(start_epoch):
            scheduler.step()
        logger.info('\tfast-forwarded LR by %d epochs on the extended schedule', start_epoch)
    else:
        if 'optimizer' in resume_state:
            optimizer.load_state_dict(resume_state['optimizer'])
            logger.info('\trestored optimizer state')
        else:
            logger.warning('\tno optimizer state in checkpoint - Adam moments start fresh')

        if 'scheduler' in resume_state:
            scheduler.load_state_dict(resume_state['scheduler'])
            logger.info('\trestored LR scheduler state')
        else:
            for _ in range(start_epoch):
                scheduler.step()
            logger.info('\tno scheduler state - fast-forwarded LR by %d epochs', start_epoch)

    best_val = resume_state.get('best_val', resume_state.get('val_rec', float('inf')))
    step = resume_state.get('step', 0)
    _restore_rng(resume_state.get('rng'))
    logger.info('\trestored best_val=%.6f, step=%d, RNG state', best_val, step)
    return best_val, step

# ========== Train/val loops ==========

# Two-stage training

def run_stage1(model: TrajectoryMaskedAutoEncoder, train_loader: DataLoader, val_loader: DataLoader,
               cfg: ModelConfig, device: torch.device, checkpoint_dir: Path,
               hf_repo: str | None = None, start_epoch: int = 0,
               resume_state: dict | None = None, ckpt_name: str = 'stage1_best.pt',
               fresh_optim: bool = False) -> None:
    logger.info('===== Stage 1: Contrastive learning (%d epochs%s) =====', cfg.stage1_epochs,
                f', resuming after epoch {start_epoch}' if start_epoch else '')
    if start_epoch >= cfg.stage1_epochs:
        logger.info('Stage 1 already complete (%d/%d epochs); skipping.',
                    start_epoch, cfg.stage1_epochs)
        return
    optimizer = AdamW(list(model.parameters()), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.stage1_epochs, eta_min = cfg.lr_min)
    best_val, step = _restore_train_state(optimizer, scheduler, resume_state,
                                          start_epoch, expected_stage=1,
                                          fresh_optim=fresh_optim)

    for epoch in range(start_epoch + 1, cfg.stage1_epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            x_full = batch['x'].to(device)
            x_spatial = x_full[..., :2]
            tau = _temporal_features(batch, device)
            kin = _kinematics(batch, device)
            coords = batch['coords'].to(device)
            pad_mask = batch['pad_mask'].to(device)
            domain_ids = _domain_ids(batch, device)

            e_sem = batch.get('e_sem')
            if e_sem is None:
                continue
            e_sem = e_sem.to(device)

            out = model.forward_stage1(x_spatial, tau, kin, coords, pad_mask, domain_ids, e_sem)
            loss = out['loss']
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
            step += 1
            if step % cfg.log_every == 0:
                logger.info('\tStage 1: step=%d | loss=%.8f | tau=%.3f', step,
                            loss.item(), model.log_tau.exp().item())
                
        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for batch in val_loader:
                x_full = batch['x'].to(device)
                x_spatial = x_full[..., :2]
                tau = _temporal_features(batch, device)
                kin = _kinematics(batch, device)
                coords = batch['coords'].to(device)
                pad_mask = batch['pad_mask'].to(device)
                domain_ids = _domain_ids(batch, device)
                e_sem = batch.get('e_sem')
                if e_sem is None:
                    continue
                e_sem = e_sem.to(device)
                out = model.forward_stage1(x_spatial, tau, kin, coords, pad_mask, domain_ids, e_sem)
                val_loss += out['loss'].item()
                n_val += 1
        val_loss /= max(n_val, 1)
        scheduler.step()
        logger.info('Stage 1: epoch %3d/%d | train=%.8f | val=%.8f', epoch, cfg.stage1_epochs,
                    total_loss / max(n_batches, 1), val_loss)
        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {'epoch': epoch, 'stage': 1, 'model': model.state_dict(),
                 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                 'best_val': best_val, 'step': step, 'rng': _rng_state(),
                 'cfg': dataclasses.asdict(cfg)},
                 checkpoint_dir / ckpt_name
            )
            logger.info('\tNew best model saved to %s', checkpoint_dir / ckpt_name)
            if hf_repo:
                _hf_upload_checkpoint(checkpoint_dir / ckpt_name, hf_repo)

    logger.info('Stage 1 complete; best val: %.8f', best_val)

def run_stage2(model: TrajectoryMaskedAutoEncoder, train_loader: DataLoader, val_loader: DataLoader,
               cfg: ModelConfig, device: torch.device, checkpoint_dir: Path,
               rng: np.random.Generator, hf_repo: str | None = None,
               start_epoch: int = 0, resume_state: dict | None = None,
               ckpt_name: str = 'best.pt', fresh_optim: bool = False) -> None:
    logger.info('===== Stage 2: Masking-recovery + soft contrastive (%d epochs%s) =====',
                cfg.stage2_epochs,
                f', resuming after epoch {start_epoch}' if start_epoch else '')
    if start_epoch >= cfg.stage2_epochs:
        logger.info('Stage 2 already complete (%d/%d epochs); skipping.',
                    start_epoch, cfg.stage2_epochs)
        return
    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.stage2_epochs, eta_min=cfg.lr_min)
    best_val, step = _restore_train_state(optimizer, scheduler, resume_state,
                                          start_epoch, expected_stage=2,
                                          fresh_optim=fresh_optim)
    patience_count = 0

    for epoch in range(start_epoch + 1, cfg.stage2_epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0
        pd_sum = torch.zeros(cfg.input_dim) # accumulate per-dim recovery MSE
        for batch in train_loader:
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
            e_sem = batch.get('e_sem')
            if e_sem is not None:
                e_sem = e_sem.to(device)

            # Detached semantic embedding for soft contrastive regulariser
            e_traj_det = None
            if e_sem is not None:
                with torch.no_grad():
                    e_traj_det = model.semantic_trajectory_embedding(e_sem, pad_mask)

            out = model.forward_stage2(
                x_spatial, tau, kin, coords, pad_mask, pos_mask, domain_ids,
                e_sem, kin_masked, sem_masked, e_traj_det
            )
            loss = out['loss']
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
            step += 1
            if 'loss_per_dim' in out:
                pd_sum += out['loss_per_dim'].detach().cpu()
            if step % cfg.log_every == 0:
                sp, tp, kn = _group_per_dim(out.get('loss_per_dim', torch.zeros(cfg.input_dim)))
                logger.info('\tStage 2: step=%d | loss=%.6f | rec=%.6f | ctr=%.6f | lb=%.6f '
                            '|| rec[spatial=%.6f temporal=%.6f kin=%.6f]',
                            step, loss.item(), out['loss_recovery'].item(),
                            out['loss_contrastive'].item(), out['loss_balance'].item(),
                            sp, tp, kn)

        model.eval()
        val_loss = 0.0
        n_val = 0
        val_pd_sum = torch.zeros(cfg.input_dim)
        with torch.no_grad():
            for batch in val_loader:
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
                e_sem = batch.get('e_sem')
                if e_sem is not None:
                    e_sem = e_sem.to(device)
                out = model.forward_stage2(
                    x_spatial, tau, kin, coords, pad_mask, pos_mask, domain_ids,
                    e_sem, kin_masked, sem_masked
                )
                val_loss += out['loss_recovery'].item()
                if 'loss_per_dim' in out:
                    val_pd_sum += out['loss_per_dim'].detach().cpu()
                n_val += 1

        val_loss /= max(n_val, 1)
        scheduler.step()
        tr_sp, tr_tp, tr_kn = _group_per_dim(pd_sum / max(n_batches, 1))
        va_sp, va_tp, va_kn = _group_per_dim(val_pd_sum / max(n_val, 1))
        logger.info('Stage 2: epoch %3d/%d | train=%.6f | val_rec=%.6f | val_spatial=%.6f', epoch,
                    cfg.stage2_epochs, total_loss / max(n_batches, 1), val_loss, va_sp)
        logger.info('\ttrain rec[spatial=%.6f temporal=%.6f kin=%.6f] | '
                    'val rec[spatial=%.6f temporal=%.6f kin=%.6f]',
                    tr_sp, tr_tp, tr_kn, va_sp, va_tp, va_kn)

        # Select/early-stop on the SPATIAL val MSE
        if va_sp < best_val:
            best_val = va_sp
            patience_count = 0
            torch.save(
                {'epoch': epoch, 'stage': 2, 'model': model.state_dict(),
                 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                 'val_spatial': best_val, 'val_rec': val_loss, 'best_val': best_val, 'step': step,
                 'rng': _rng_state(), 'cfg': dataclasses.asdict(cfg)},
                 checkpoint_dir / ckpt_name
            )
            logger.info('\tNew best model saved to %s', checkpoint_dir / ckpt_name)
            if hf_repo:
                _hf_upload_checkpoint(checkpoint_dir / ckpt_name, hf_repo)
        else:
            patience_count += 1
            if patience_count >= cfg.patience:
                logger.info('Early stopping at epoch %d', epoch)
                break

    logger.info('Stage 2 completed; best val spatial MSE: %.8f', best_val)

# Original training

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
            candidate = sem_dir / f"{_sem_base(p, domain)}_sem.npy"
            if candidate.exists():
                result.append(candidate)
            else:
                candidate_alt = sem_dir / f"{_sem_base(p, domain)}_described_sem.npy"
                if candidate_alt.exists():
                    result.append(candidate_alt)
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
        num_workers=cfg.num_workers if shuffle else max(2, cfg.num_workers // 2),
        pin_memory=True,
        persistent_workers=cfg.num_workers > 0,
        prefetch_factor=4 if cfg.num_workers > 0 else None,
    )

def train(cfg: ModelConfig, args: argparse.Namespace) -> None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info('Device: %s', device)

    # Enable TF32 matmul/conv on Ampere+ (A100)
    if device.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    def _paths(p: str | None) -> list[Path]:
        if not p:
            return []
        base = Path(p)
        return sorted(base.glob('**/*.parquet')) if base.is_dir() else [base]

    has_urban = bool(_paths(args.urban_train))
    has_maritime = bool(_paths(args.maritime_train))
    if not (has_urban or has_maritime):
        raise ValueError('No training data: provide --urban-train and/or --maritime-train')

    urban_train_loader = build_loader(
        _paths(args.urban_train), 'urban', args.urban_sem_npy or [], cfg, True
    ) if has_urban else None
    maritime_train_loader = build_loader(
        _paths(args.maritime_train), 'maritime', args.maritime_sem_npy or [], cfg, True
    ) if has_maritime else None
    urban_val_loader = build_loader(
        _paths(args.urban_val), 'urban', args.urban_val_sem_npy or [], cfg, False
    ) if has_urban else None
    maritime_val_loader = build_loader(
        _paths(args.maritime_val), 'maritime', args.maritime_val_sem_npy or [], cfg, False
    ) if has_maritime else None

    class InterleavedLoader:
        """Alternate urban and maritime batches"""
        def __init__(self, a: DataLoader, b: DataLoader):
            self.a = a
            self.b = b

        def __iter__(self):
            return chain.from_iterable(zip(self.a, self.b))

        def __len__(self):
            return min(len(self.a), len(self.b)) * 2

    def _combine(a, b):
        """Interleave both domains, or pass through the single active loader."""
        if a is not None and b is not None:
            return InterleavedLoader(a, b)
        return a if a is not None else b

    train_loader = _combine(urban_train_loader, maritime_train_loader)
    val_loader = _combine(urban_val_loader, maritime_val_loader)
    mode_str = ('cross-domain' if (has_urban and has_maritime)
                else 'urban-only' if has_urban else 'maritime-only')
    logger.info('Training mode: %s', mode_str)
    logger.info('Train batches/epoch: %d  |  Val batches/epoch: %d',
                len(train_loader), len(val_loader))

    model = TrajectoryMaskedAutoEncoder(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info('Model: %s params  d_model=%d  n_layers=%d  n_heads=%d',
                f'{n_params:,}', cfg.d_model, cfg.n_layers, cfg.n_heads)
    # logger.info('Training for %d epochs  lr=%.1e  batch_size=%d',
    #             cfg.epochs, cfg.lr, cfg.batch_size)
    
    # optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    # scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=cfg.lr_min)

    checkpoint_dir = Path(cfg.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    hf_repo = getattr(args, 'hf_repo', None)
    if hf_repo:
        _ensure_hf_repo(hf_repo)
        logger.info('Best checkpoints will be backed up to hf://%s', hf_repo)

    rng = np.random.default_rng()

    run_name = getattr(args, 'run_name', None)
    stage1_ckpt = f'{run_name}_stage1_best.pt' if run_name else 'stage1_best.pt'
    stage2_ckpt = f'{run_name}_best.pt' if run_name else 'best.pt'
    if run_name:
        logger.info('Run name: %s -> checkpoints %s / %s', run_name, stage1_ckpt, stage2_ckpt)

    # ----- Resume handling -----
    ckpt_file = getattr(args, 'checkpoint_file', None)
    resume_s1 = getattr(args, 'resume_stage1', 0) or 0
    resume_s2 = getattr(args, 'resume_stage2', 0) or 0
    if (resume_s1 or resume_s2) and not ckpt_file:
        raise ValueError('--resume-stage1/--resume-stage2 require --checkpoint-file')
    ckpt = _load_checkpoint(model, ckpt_file, device) if ckpt_file else None

    # Two-stage training
    logger.info('Training: stage1=%d epochs | stage2=%d epochs | lr=%.1e | batch_size=%d',
                cfg.stage1_epochs, cfg.stage2_epochs, cfg.lr, cfg.batch_size)

    if resume_s2:
        # Resume mid-Stage-2: skip Stage 1 entirely, continue Stage 2 from epoch resume_s2,
        # restoring optimizer/scheduler/RNG for an exact continuation.
        logger.info('Resuming: skipping Stage 1, continuing Stage 2 after epoch %d', resume_s2)
        run_stage2(model, train_loader, val_loader, cfg, device, checkpoint_dir, rng,
                   hf_repo=hf_repo, start_epoch=resume_s2, resume_state=ckpt,
                   ckpt_name=stage2_ckpt, fresh_optim=getattr(args, 'fresh_optim', False))
    else:
        # Fresh, warm-start (ckpt without resume → weights only), or resume mid-Stage-1.
        # resume_state is only passed when actually resuming Stage 1, so a warm start
        # does not pull a (possibly Stage-2) optimizer state into a fresh Stage 1.
        run_stage1(model, train_loader, val_loader, cfg, device, checkpoint_dir,
                   hf_repo=hf_repo, start_epoch=resume_s1,
                   resume_state=ckpt if resume_s1 else None, ckpt_name=stage1_ckpt,
                   fresh_optim=getattr(args, 'fresh_optim', False))
        run_stage2(model, train_loader, val_loader, cfg, device, checkpoint_dir, rng,
                   hf_repo=hf_repo, ckpt_name=stage2_ckpt)
    logger.info('Training complete. Best checkpoint: %s', checkpoint_dir / stage2_ckpt)

    # Single-stage training
    # logger.info('Training for %d epochs  lr=%.1e  batch_size=%d',
    #             cfg.epochs, cfg.lr, cfg.batch_size)
    # optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    # scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=cfg.lr_min)
    # best_val = float('inf')
    # patience_count = 0
    # step = 0
    # for epoch in range(1, cfg.epochs + 1):
    #     train_loss, train_rec, train_sem, step = train_epoch(
    #         model, train_loader, optimizer, device, cfg, step, rng
    #     )
    #     val_loss, val_rec = val_epoch(model, val_loader, device, cfg, rng)
    #     scheduler.step()
    #     logger.info(
    #         'Epoch %3d/%d | train=%.8f (rec=%.8f sem=%.8f) | val=%.8f (rec=%.8f) | lr=%.2e',
    #         epoch, cfg.epochs, train_loss, train_rec, train_sem,
    #         val_loss, val_rec, scheduler.get_last_lr()[0]
    #     )
    #     if val_rec < best_val:
    #         best_val = val_rec
    #         patience_count = 0
    #         torch.save(
    #             {'epoch': epoch, 'model': model.state_dict(),
    #              'val_rec': best_val, 'cfg': dataclasses.asdict(cfg)},
    #              checkpoint_dir / 'best.pt'
    #         )
    #         logger.info('\tNew best model saved to %s', checkpoint_dir)
    #     else:
    #         patience_count += 1
    #         if patience_count >= cfg.patience:
    #             logger.info('Early stopping at epoch %d', epoch)
    #             break
    # logger.info('Model training complete. Best val recovery: %.8f', best_val)

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

    parser.add_argument('--epochs', type=int, default=50)           # single-stage
    parser.add_argument('--stage1-epochs', type=int, default=15)    # two-stage
    parser.add_argument('--stage2-epochs', type=int, default=35)    # two-stage
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--d-model', type=int, default=128)
    parser.add_argument('--max-len', type=int, default=256)
    parser.add_argument('--alpha', type=float, default=0.05)
    parser.add_argument('--no-semantics', action='store_true')
    parser.add_argument('--checkpoint-dir', default='checkpoints')
    parser.add_argument('--run-name', default=None, metavar='NAME')
    parser.add_argument('--hf-repo', default=None, metavar='REPO_ID')
    # Resume from a saved checkpoint. Provide --checkpoint-file plus exactly one
    # of --resume-stage1 / --resume-stage2 (the number of epochs already done)
    parser.add_argument('--checkpoint-file', default=None, metavar='PT')
    parser.add_argument('--resume-stage1', type=int, default=0, metavar='N')
    parser.add_argument('--resume-stage2', type=int, default=0, metavar='N')
    parser.add_argument('--fresh-optim', action='store_true')
    args = parser.parse_args()

    if args.resume_stage1 and args.resume_stage2:
        parser.error('Use only one of --resume-stage1 OR --resume-stage2.')

    cfg = ModelConfig(
        d_model=args.d_model,
        max_len=args.max_len,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        lr=args.lr,
        sem_pred_alpha=args.alpha,
        use_semantics=not args.no_semantics,
        checkpoint_dir=args.checkpoint_dir,
        stage1_epochs=args.stage1_epochs,
        stage2_epochs=args.stage2_epochs,
        urban_train_sem_npy=args.urban_sem_npy or [],
        urban_val_sem_npy=args.urban_val_sem_npy or [],
        maritime_train_sem_npy=args.maritime_sem_npy or [],
        maritime_val_sem_npy=args.maritime_val_sem_npy or [],
    )
    train(cfg, args)

if __name__ == '__main__':
    main()