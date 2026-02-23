"""
src/models/tabtransformer.py
-----------------------------
PyTorch TabTransformer implementation for fraud detection.
Based on "TabTransformer: Tabular Data Modeling Using Contextual Embeddings" (Huang et al., 2020).

Applies multi-head self-attention to categorical embeddings, then concatenates
with numerical features before passing through an MLP classifier head.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, TensorDataset

from src.core.exceptions import ModelNotFittedError, TrainingError
from src.core.interfaces import BaseModel

logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MultiHeadAttentionBlock(nn.Module):
    """Transformer encoder block for categorical embeddings."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention with residual
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)
        # Feed-forward with residual
        x = self.norm2(x + self.ff(x))
        return x


class TabTransformerNet(nn.Module):
    """
    TabTransformer network architecture.

    Architecture:
        Categorical features → Embeddings → N × TransformerBlock → Flatten
        Numerical features → BatchNorm
        [Categorical Embeddings ‖ Numerical Features] → MLP → Sigmoid
    """

    def __init__(
        self,
        n_categorical: int,
        categorical_vocab_sizes: list[int],
        n_numerical: int,
        embedding_dim: int = 64,
        num_heads: int = 8,
        num_layers: int = 6,
        mlp_hidden_dims: list[int] | None = None,
        dropout: float = 0.15,
        attention_dropout: float = 0.10,
    ) -> None:
        super().__init__()

        mlp_hidden_dims = mlp_hidden_dims or [256, 128, 64]

        # Categorical embeddings
        self.embeddings = nn.ModuleList([
            nn.Embedding(vocab_size + 1, embedding_dim, padding_idx=0)
            for vocab_size in categorical_vocab_sizes
        ])

        # Transformer blocks
        self.transformer_blocks = nn.Sequential(*[
            MultiHeadAttentionBlock(embedding_dim, num_heads, attention_dropout)
            for _ in range(num_layers)
        ])

        # Numerical feature normalization
        self.num_bn = nn.BatchNorm1d(n_numerical) if n_numerical > 0 else None

        # MLP classifier head
        cat_output_dim = n_categorical * embedding_dim
        mlp_input_dim = cat_output_dim + n_numerical

        layers: list[nn.Module] = []
        in_dim = mlp_input_dim
        for h_dim in mlp_hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.GELU(),
                nn.BatchNorm1d(h_dim),
                nn.Dropout(dropout),
            ])
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, 1))

        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        x_cat: torch.Tensor,   # (batch, n_categorical)
        x_num: torch.Tensor,   # (batch, n_numerical)
    ) -> torch.Tensor:
        # Embed categoricals: (batch, n_cat, embed_dim)
        cat_embeds = torch.stack([emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)], dim=1)

        # Apply transformer blocks
        cat_out = self.transformer_blocks(cat_embeds)  # (batch, n_cat, embed_dim)
        cat_out = cat_out.flatten(1)  # (batch, n_cat * embed_dim)

        # Normalize numericals
        if self.num_bn is not None and x_num.shape[1] > 0:
            num_out = self.num_bn(x_num)
        else:
            num_out = x_num

        # Concatenate and classify
        combined = torch.cat([cat_out, num_out], dim=1)
        logits = self.mlp(combined).squeeze(-1)
        return logits


class TabTransformerModel(BaseModel):
    """
    Wraps TabTransformerNet with sklearn-compatible interface.
    Handles training loop, early stopping, and serialization.
    """

    MODEL_FILENAME = "tabtransformer.pt"
    CONFIG_FILENAME = "tabtransformer_config.joblib"

    def __init__(self, hyperparams: dict[str, Any] | None = None) -> None:
        self._hyperparams = hyperparams or self._default_hyperparams()
        self._net: TabTransformerNet | None = None
        self._cat_columns: list[str] = []
        self._num_columns: list[str] = []
        self._cat_vocab_sizes: list[int] = []
        self._feature_names: list[str] = []
        self._is_fitted = False
        self._training_losses: list[float] = []

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Train TabTransformer."""
        logger.info("Training TabTransformer on %d samples (device=%s)", len(X), DEVICE)

        self._feature_names = list(X.columns)
        self._identify_column_types(X)

        # Build vocabulary sizes from training data
        self._cat_vocab_sizes = []
        for col in self._cat_columns:
            self._cat_vocab_sizes.append(int(X[col].max()) + 1)

        # Initialize network
        self._net = TabTransformerNet(
            n_categorical=len(self._cat_columns),
            categorical_vocab_sizes=self._cat_vocab_sizes,
            n_numerical=len(self._num_columns),
            embedding_dim=self._hyperparams["embedding_dim"],
            num_heads=self._hyperparams["num_heads"],
            num_layers=self._hyperparams["num_transformer_layers"],
            mlp_hidden_dims=self._hyperparams["mlp_hidden_dims"],
            dropout=self._hyperparams["dropout"],
            attention_dropout=self._hyperparams["attention_dropout"],
        ).to(DEVICE)

        # Class-weighted BCE loss
        pos_weight = torch.tensor([self._hyperparams["pos_weight"]], device=DEVICE)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        optimizer = AdamW(
            self._net.parameters(),
            lr=self._hyperparams["learning_rate"],
            weight_decay=self._hyperparams["weight_decay"],
        )
        scheduler = CosineAnnealingWarmRestarts(
            optimizer, T_0=self._hyperparams.get("t_0", 10), T_mult=2
        )

        train_loader = self._make_dataloader(X, y, shuffle=True)
        val_loader = self._make_dataloader(X_val, y_val, shuffle=False) if X_val is not None else None

        best_val_loss = float("inf")
        patience = self._hyperparams["patience"]
        patience_counter = 0
        best_state = None

        for epoch in range(self._hyperparams["max_epochs"]):
            train_loss = self._train_epoch(self._net, train_loader, criterion, optimizer)
            scheduler.step()
            self._training_losses.append(train_loss)

            if val_loader is not None:
                val_loss = self._eval_epoch(self._net, val_loader, criterion)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = {k: v.clone() for k, v in self._net.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    logger.info("Early stopping at epoch %d (patience=%d)", epoch + 1, patience)
                    break

                if (epoch + 1) % 10 == 0:
                    logger.info("Epoch %d | train_loss=%.4f | val_loss=%.4f", epoch + 1, train_loss, val_loss)

        # Restore best weights
        if best_state is not None:
            self._net.load_state_dict(best_state)

        self._is_fitted = True
        logger.info("TabTransformer training complete")
        return {"best_val_loss": best_val_loss, "epochs_trained": epoch + 1}

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return fraud probabilities."""
        if not self._is_fitted or self._net is None:
            raise ModelNotFittedError("TabTransformer not fitted")

        self._net.eval()
        x_cat, x_num = self._prepare_tensors(X)

        with torch.no_grad():
            logits = self._net(x_cat, x_num)
            probs = torch.sigmoid(logits).cpu().numpy()

        return probs

    def get_feature_importance(self) -> dict[str, float]:
        """Return uniform importance (attention-based importance requires SHAP)."""
        if not self._is_fitted:
            raise ModelNotFittedError("Model not fitted")
        # Uniform as placeholder; use SHAP for production explainability
        n = len(self._feature_names)
        return {name: 1.0 / n for name in self._feature_names}

    def save(self, path: str) -> None:
        """Save model checkpoint and configuration."""
        import joblib

        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)

        torch.save(self._net.state_dict(), save_path / self.MODEL_FILENAME)
        config = {
            "hyperparams": self._hyperparams,
            "cat_columns": self._cat_columns,
            "num_columns": self._num_columns,
            "cat_vocab_sizes": self._cat_vocab_sizes,
            "feature_names": self._feature_names,
        }
        joblib.dump(config, save_path / self.CONFIG_FILENAME)
        logger.info("TabTransformer saved to %s", save_path)

    def load(self, path: str) -> None:
        """Load model from checkpoint."""
        import joblib

        load_path = Path(path)
        config = joblib.load(load_path / self.CONFIG_FILENAME)

        self._hyperparams = config["hyperparams"]
        self._cat_columns = config["cat_columns"]
        self._num_columns = config["num_columns"]
        self._cat_vocab_sizes = config["cat_vocab_sizes"]
        self._feature_names = config["feature_names"]

        self._net = TabTransformerNet(
            n_categorical=len(self._cat_columns),
            categorical_vocab_sizes=self._cat_vocab_sizes,
            n_numerical=len(self._num_columns),
            **{k: self._hyperparams[k] for k in [
                "embedding_dim", "num_heads", "dropout", "attention_dropout",
            ]},
            num_layers=self._hyperparams["num_transformer_layers"],
            mlp_hidden_dims=self._hyperparams["mlp_hidden_dims"],
        ).to(DEVICE)

        self._net.load_state_dict(torch.load(load_path / self.MODEL_FILENAME, map_location=DEVICE))
        self._net.eval()
        self._is_fitted = True
        logger.info("TabTransformer loaded from %s", load_path)

    # --- Private helpers ---

    def _identify_column_types(self, X: pd.DataFrame) -> None:
        """Identify categorical vs numerical columns."""
        self._cat_columns = [c for c in X.columns if X[c].dtype in ["int32", "int64", "object", "category"]
                            and X[c].nunique() < 50]
        self._num_columns = [c for c in X.columns if c not in self._cat_columns]

    def _prepare_tensors(self, X: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor]:
        cat_data = X[self._cat_columns].fillna(0).astype(int).values if self._cat_columns else np.zeros((len(X), 0), dtype=int)
        num_data = X[self._num_columns].fillna(0).astype(float).values if self._num_columns else np.zeros((len(X), 0))

        x_cat = torch.tensor(cat_data, dtype=torch.long, device=DEVICE)
        x_num = torch.tensor(num_data, dtype=torch.float32, device=DEVICE)
        return x_cat, x_num

    def _make_dataloader(
        self,
        X: pd.DataFrame | None,
        y: pd.Series | None,
        shuffle: bool = False,
    ) -> DataLoader | None:
        if X is None or y is None:
            return None
        x_cat, x_num = self._prepare_tensors(X)
        y_tensor = torch.tensor(y.values, dtype=torch.float32, device=DEVICE)
        dataset = TensorDataset(x_cat, x_num, y_tensor)
        return DataLoader(dataset, batch_size=self._hyperparams["batch_size"], shuffle=shuffle)

    def _train_epoch(
        self,
        net: TabTransformerNet,
        loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> float:
        net.train()
        total_loss = 0.0
        for x_cat, x_num, y_batch in loader:
            optimizer.zero_grad()
            logits = net(x_cat, x_num)
            loss = criterion(logits, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), self._hyperparams.get("gradient_clip_norm", 1.0))
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)

    def _eval_epoch(
        self,
        net: TabTransformerNet,
        loader: DataLoader,
        criterion: nn.Module,
    ) -> float:
        net.eval()
        total_loss = 0.0
        with torch.no_grad():
            for x_cat, x_num, y_batch in loader:
                logits = net(x_cat, x_num)
                loss = criterion(logits, y_batch)
                total_loss += loss.item()
        return total_loss / len(loader)

    @staticmethod
    def _default_hyperparams() -> dict[str, Any]:
        return {
            "embedding_dim": 64,
            "num_heads": 8,
            "num_transformer_layers": 6,
            "mlp_hidden_dims": [256, 128, 64],
            "dropout": 0.15,
            "attention_dropout": 0.10,
            "learning_rate": 0.0003,
            "weight_decay": 0.00012,
            "batch_size": 1024,
            "max_epochs": 100,
            "patience": 12,
            "gradient_clip_norm": 1.0,
            "pos_weight": 40.0,
            "t_0": 10,
        }
