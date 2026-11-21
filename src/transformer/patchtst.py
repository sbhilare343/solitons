"""PatchTST model (EXP-005): small univariate channel-independent Patch Transformer.

Core architecture of Zhou et al. (2023) "A Time Series is Worth 64 Words", scoped
down for the registered "small first model": patching -> linear projection ->
learnable positional embedding -> Transformer encoder -> flatten head. Declared
scope reductions (see docs/experiment_log.md EXP-005): no RevIN, no HLF encoder.
"""

from __future__ import annotations

import math

import torch
from torch import nn


class PatchTST(nn.Module):
    """Univariate PatchTST producing a one-step-ahead point forecast.

    Parameters mirror the pre-declared EXP-005 configuration; changing any of
    them invalidates the registered run.
    """

    def __init__(
        self,
        lookback: int = 96,
        patch_len: int = 16,
        stride: int = 8,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        ffn_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if (lookback - patch_len) % stride != 0:
            raise ValueError("lookback - patch_len must be divisible by stride.")
        self.lookback = lookback
        self.patch_len = patch_len
        self.stride = stride
        self.n_patches = (lookback - patch_len) // stride + 1

        self.unfold = nn.Unfold(
            kernel_size=(1, patch_len), stride=(1, stride)
        )  # (B, p, P)
        self.proj = nn.Linear(patch_len, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches, d_model))
        nn.init.normal_(self.pos_embed, mean=0.0, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Linear(d_model * self.n_patches, 1)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear) and module is not self.head:
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.xavier_uniform_(self.head.weight)
        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, lookback) scaled returns -> (batch, 1) scaled next return."""
        if x.dim() != 2 or x.shape[1] != self.lookback:
            raise ValueError(f"Expected (B, {self.lookback}), got {tuple(x.shape)}")
        # nn.Unfold operates on 2-D spatial inputs; add a singleton height so
        # the temporal kernel moves only along the lookback axis.
        patches = self.unfold(x.unsqueeze(1).unsqueeze(2))  # (B, patch_len, P)
        tokens = patches.transpose(1, 2)               # (B, P, patch_len)
        tokens = self.proj(tokens) + self.pos_embed    # (B, P, d_model)
        encoded = self.encoder(tokens)                 # (B, P, d_model)
        return self.head(encoded.reshape(encoded.shape[0], -1))

    @staticmethod
    def config_hash(config: dict) -> str:
        """Short deterministic fingerprint of the architecture configuration."""
        import hashlib
        import json

        blob = json.dumps(config, sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:12]


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def default_config() -> dict:
    """The pre-declared EXP-005 configuration (no tuning)."""
    return {
        "lookback": 96,
        "patch_len": 16,
        "stride": 8,
        "d_model": 64,
        "n_heads": 4,
        "n_layers": 2,
        "ffn_dim": 128,
        "dropout": 0.1,
    }


def build_model(config: dict | None = None) -> PatchTST:
    return PatchTST(**(config or default_config()))
