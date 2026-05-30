"""
Frozen-weight text encoder for encoding per-point semantic description

Default: use google/embeddinggemma-300m
    - https://huggingface.co/google/embeddinggemma-300m
    - https://ai.google.dev/gemma/docs/embeddinggemma
    - Multilingual, 1024D
    - Supports Matryoshka Representation Learning for truncating embeddings

Usage:
    encoder = SemanticEncoder()
    embeddings = encoder.encode([sentence, sentence, ...])
    -> returns: np.ndarray, shape (N, embed_dim), dtype float16
"""
from __future__ import annotations

import logging
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_MODEL = 'google/embeddinggemma-300m'
EMBED_DIM = 1024
MATRYOSHKA_MODELS = {
    'jinaai/jina-embeddings-v3',
    'nomic-ai/nomic-embed-text-v1.5',
    'google/embeddinggemma-300m',
    'google/embeddinggemma-1b',
}

class SemanticEncoder:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: Optional[str] = None,
        batch_size: int = 1024,
        normalize: bool = True,
        max_seq_length: int = 512,
        truncate_dim: Optional[int] = None,
        trust_remote_code: bool = False,
        attn_implementation: Optional[str] = None,
    ):
        """
        Params:
            model_name:           HuggingFace model ID, default: 'google/embeddinggemma-300m'
            device:               'cpu' or 'cuda', default None (auto-detect)
            batch_size:           encoding batch size, default 1024
            normalize:            L2-normalize embeddings
            max_seq_length:       truncate descriptions longer than this threshold
            truncate_dim:         # of first output dimensions to keep for Matryoshka models
            trust_remote_code:    required for some models
            attn_implementation:  'sdpa' (PyTorch 2.0+ native, no install needed) or
                                  'flash_attention_2' (requires: pip install flash-attn).
                                  Both reduce memory pressure; flash_attention_2 is fastest
                                  on H100/A100 and enables larger batch sizes.
        """
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.batch_size = batch_size
        self.normalize = normalize
        self.model_name = model_name
        self.truncate_dim = truncate_dim

        on_cuda = device.startswith('cuda')
        logger.info('Loading %s on %s (bf16=%s, truncate_dim=%s, attn=%s)',
                    model_name, device, on_cuda, truncate_dim, attn_implementation)

        model_kwargs: dict = {}
        if on_cuda:
            model_kwargs['dtype'] = torch.bfloat16
        # Default to 'eager' (pure PyTorch)
        model_kwargs['attn_implementation'] = attn_implementation or 'eager'

        self._model = SentenceTransformer(
            model_name,
            device=device,
            trust_remote_code=trust_remote_code,
            truncate_dim=truncate_dim,
            model_kwargs=model_kwargs,
        )
        self._model.max_seq_length = max_seq_length

        for p in self._model.parameters():
            p.requires_grad_(False)
        self._model.eval()

        logger.info('Encoder ready: dim=%d, max_seq=%d, bf16=%s',
                    self.embed_dim, max_seq_length, on_cuda)
    
    @property
    def embed_dim(self) -> int:
        # v5.3: get_sentence_embedding_dimension()
        # Later version: swap to get_embedding_dimension() - Google Colab env
        # https://sbert.net/docs/migration_guide.html#id1
        return self._model.get_embedding_dimension()

    def encode(self, texts: list[str], show_progress: bool = False) -> np.ndarray:
        """
        Encode a list of description strings

        Returns:
            np.ndarray of shape (len(texts), embed_dim), dtype float16
        """
        with torch.no_grad():
            embs = self._model.encode(
                texts,
                batch_size=self.batch_size,
                normalize_embeddings=self.normalize,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
            )
        return embs.astype(np.float16)