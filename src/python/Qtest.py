import os
import glob
import copy
import random
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.ao.quantization as tq
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


# =========================================================
# Utilities
# =========================================================

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    print("torch version:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    print("cuda device count:", torch.cuda.device_count())

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("Using device: cuda:0")
        print("GPU:", torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        print("Using device: cpu")

    return device


# =========================================================
# Data loading
# =========================================================

def load_all_files(data_dir: str, pattern: str = "*.h5") -> pd.DataFrame:
    file_paths = sorted(glob.glob(os.path.join(data_dir, pattern)))
    if len(file_paths) == 0:
        raise FileNotFoundError(f"No files found in {data_dir} matching {pattern}")

    dfs = []
    for fp in file_paths:
        temp_df = pd.read_hdf(fp).copy()
        if "source_file" not in temp_df.columns:
            temp_df["source_file"] = os.path.basename(fp)
        dfs.append(temp_df)

    return pd.concat(dfs, ignore_index=True)


def create_windowed_dataset(
    df: pd.DataFrame,
    subject_col: str = "SubjectID",
    time_col: str = "timestamp",
    target_col: str = "SpO2_Rad",
    seq_feature_cols: List[str] = None,
    window_size: int = 400,
    stride: int = 400,
    drop_incomplete: bool = True,
) -> pd.DataFrame:
    if seq_feature_cols is None:
        seq_feature_cols = ["red_win_filtered", "ir_win_filtered"]

    required_cols = seq_feature_cols + [target_col, subject_col, time_col, "skintone"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    all_samples = []

    for subject_id, group in df.groupby(subject_col):
        group = group.sort_values(time_col).reset_index(drop=True)
        n = len(group)

        starts = range(0, n - window_size + 1, stride) if drop_incomplete else range(0, n, stride)

        for start in starts:
            end = start + window_size
            window = group.iloc[start:end]

            if len(window) < window_size and drop_incomplete:
                continue

            sample = {
                "SubjectID": subject_id,
                "window_start_idx": start,
                "window_end_idx": min(end - 1, n - 1),
                "source_file": window["source_file"].iloc[0] if "source_file" in window.columns else "",
                "skintone": np.float32(window["skintone"].iloc[0]),
            }

            for col in seq_feature_cols:
                arr = window[col].to_numpy(dtype=np.float32)
                if arr.ndim != 1 or len(arr) != window_size:
                    raise ValueError(
                        f"Column {col} did not form a 1D window of length {window_size}. "
                        f"Got shape {arr.shape}."
                    )
                sample[col] = arr

            sample[target_col] = np.float32(window[target_col].iloc[0])
            all_samples.append(sample)

    return pd.DataFrame(all_samples)


def subject_wise_split(
    windowed_df: pd.DataFrame,
    subject_col: str = "SubjectID",
    val_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    subjects = windowed_df[subject_col].unique()
    train_subjects, val_subjects = train_test_split(
        subjects,
        test_size=val_size,
        random_state=random_state,
        shuffle=True,
    )

    train_df = windowed_df[windowed_df[subject_col].isin(train_subjects)].reset_index(drop=True)
    val_df = windowed_df[windowed_df[subject_col].isin(val_subjects)].reset_index(drop=True)

    return train_df, val_df


# =========================================================
# Dataset
# =========================================================

class RawSpO2WindowDataset(Dataset):
    def __init__(
        self,
        windowed_df: pd.DataFrame,
        seq_feature_cols: List[str],
        target_col: str = "SpO2_Rad",
        seq_len: int = 400,
        normalize_seq: bool = True,
        y_mean: float = None,
        y_std: float = None,
    ):
        self.df = windowed_df.reset_index(drop=True).copy()
        self.seq_feature_cols = seq_feature_cols
        self.target_col = target_col
        self.seq_len = seq_len
        self.normalize_seq = normalize_seq
        self.y_mean = y_mean
        self.y_std = y_std

        required_cols = seq_feature_cols + [target_col, "skintone"]
        for col in required_cols:
            if col not in self.df.columns:
                raise ValueError(f"Missing column: {col}")

    def __len__(self):
        return len(self.df)

    def _to_1d_float_array(self, x, col_name: str, expected_len: int) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float32)
        if arr.ndim != 1:
            raise ValueError(f"Column '{col_name}' is not 1D. Got shape {arr.shape}.")
        if len(arr) != expected_len:
            raise ValueError(f"Column '{col_name}' length mismatch. Expected {expected_len}, got {len(arr)}.")
        return arr

    def _normalize_signal(self, x: np.ndarray) -> np.ndarray:
        std = x.std()
        if std < 1e-8:
            std = 1.0
        return (x - x.mean()) / std

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        seq_features = []
        for col in self.seq_feature_cols:
            arr = self._to_1d_float_array(row[col], col, self.seq_len)
            if self.normalize_seq:
                arr = self._normalize_signal(arr)
            seq_features.append(arr)

        x_seq = np.stack(seq_features, axis=0).astype(np.float32)  # [2, T]
        x_skin = np.float32(row["skintone"])
        y = np.float32(row[self.target_col])

        if self.y_mean is not None and self.y_std is not None:
            y = (y - self.y_mean) / self.y_std

        return (
            torch.tensor(x_seq, dtype=torch.float32),
            torch.tensor(x_skin, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
        )


# =========================================================
# Original FP32 model
# =========================================================

class ConvTokenEncoder(nn.Module):
    def __init__(self, in_channels: int = 1, hidden_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=5, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Conv1d(32, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.net(x)
        x = x.transpose(1, 2)
        return x


class CrossAttentionBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, kv):
        attn_out, _ = self.attn(q, kv, kv)
        x = self.norm1(q + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return x


class TemporalSelfAttentionBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return x


class AttentionPooling(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(dim, 2 * dim),
            nn.Tanh(),
            nn.Linear(2 * dim, 1)
        )

    def forward(self, x):
        w = self.score(x)
        w = torch.softmax(w, dim=1)
        pooled = (x * w).sum(dim=1)
        return pooled


class SkinFiLM(nn.Module):
    def __init__(self, token_dim: int, hidden_dim: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 2 * hidden_dim),
            nn.ReLU(),
            nn.Linear(2 * hidden_dim, 4 * token_dim),
        )
        self.token_dim = token_dim

    def forward(self, x_skin):
        if x_skin.ndim == 1:
            x_skin = x_skin.unsqueeze(-1)

        params = self.net(x_skin)
        gamma_r, beta_r, gamma_i, beta_i = torch.chunk(params, 4, dim=-1)

        gamma_r = gamma_r.unsqueeze(1)
        beta_r = beta_r.unsqueeze(1)
        gamma_i = gamma_i.unsqueeze(1)
        beta_i = beta_i.unsqueeze(1)

        return gamma_r, beta_r, gamma_i, beta_i


class RawCrossAttentionSpO2Net(nn.Module):
    def __init__(self, hidden_dim: int = 64, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()

        self.red_encoder = ConvTokenEncoder(in_channels=1, hidden_dim=hidden_dim, dropout=dropout)
        self.ir_encoder = ConvTokenEncoder(in_channels=1, hidden_dim=hidden_dim, dropout=dropout)

        self.skin_film = SkinFiLM(token_dim=hidden_dim, hidden_dim=16)

        self.red_to_ir = CrossAttentionBlock(hidden_dim, num_heads=num_heads, dropout=dropout)
        self.ir_to_red = CrossAttentionBlock(hidden_dim, num_heads=num_heads, dropout=dropout)

        self.temporal_attn = TemporalSelfAttentionBlock(hidden_dim * 2, num_heads=num_heads, dropout=dropout)
        self.pool = AttentionPooling(hidden_dim * 2)

        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x_seq, x_skin):
        red = x_seq[:, 0:1, :]
        ir = x_seq[:, 1:2, :]

        red = self.red_encoder(red)
        ir = self.ir_encoder(ir)

        gamma_r, beta_r, gamma_i, beta_i = self.skin_film(x_skin)

        red = red * (1.0 + gamma_r) + beta_r
        # ir = ir * (1.0 + gamma_i) + beta_i

        red_cross = self.red_to_ir(red, ir)
        ir_cross = self.ir_to_red(ir, red)

        fused = torch.cat([red_cross, ir_cross], dim=-1)
        fused = self.temporal_attn(fused)
        fused = self.pool(fused)

        y_hat = self.regressor(fused).squeeze(-1)
        return y_hat


# =========================================================
# QAT-friendly model
# =========================================================

class QATConvTokenEncoder(nn.Module):
    def __init__(self, in_channels: int = 1, hidden_dim: int = 64, dropout: float = 0.1):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=5, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=False),
        )
        self.drop1 = nn.Dropout(dropout)

        self.block2 = nn.Sequential(
            nn.Conv1d(32, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=False),
        )
        self.drop2 = nn.Dropout(dropout)

        self.block3 = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=False),
        )

    def fuse_model(self):
        tq.fuse_modules(self.block1, ["0", "1", "2"], inplace=True)
        tq.fuse_modules(self.block2, ["0", "1", "2"], inplace=True)
        tq.fuse_modules(self.block3, ["0", "1", "2"], inplace=True)

    def forward(self, x):
        x = self.block1(x)
        x = self.drop1(x)
        x = self.block2(x)
        x = self.drop2(x)
        x = self.block3(x)
        x = x.transpose(1, 2)
        return x


class RawCrossAttentionSpO2NetQAT(nn.Module):
    def __init__(self, hidden_dim: int = 64, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()

        self.quant_in = tq.QuantStub()
        self.dequant_before_attn = tq.DeQuantStub()

        self.quant_before_regressor = tq.QuantStub()
        self.dequant_out = tq.DeQuantStub()

        self.red_encoder = QATConvTokenEncoder(in_channels=1, hidden_dim=hidden_dim, dropout=dropout)
        self.ir_encoder = QATConvTokenEncoder(in_channels=1, hidden_dim=hidden_dim, dropout=dropout)

        self.skin_film = SkinFiLM(token_dim=hidden_dim, hidden_dim=16)

        self.red_to_ir = CrossAttentionBlock(hidden_dim, num_heads=num_heads, dropout=dropout)
        self.ir_to_red = CrossAttentionBlock(hidden_dim, num_heads=num_heads, dropout=dropout)

        self.temporal_attn = TemporalSelfAttentionBlock(hidden_dim * 2, num_heads=num_heads, dropout=dropout)
        self.pool = AttentionPooling(hidden_dim * 2)

        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=False),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 16),
            nn.ReLU(inplace=False),
            nn.Linear(16, 1)
        )

    def fuse_model(self):
        self.red_encoder.fuse_model()
        self.ir_encoder.fuse_model()
        tq.fuse_modules(self.regressor, ["0", "1"], inplace=True)
        tq.fuse_modules(self.regressor, ["3", "4"], inplace=True)

    def forward(self, x_seq, x_skin):
        red = x_seq[:, 0:1, :]
        ir = x_seq[:, 1:2, :]

        red = self.quant_in(red)
        ir = self.quant_in(ir)

        red = self.red_encoder(red)
        ir = self.ir_encoder(ir)

        red = self.dequant_before_attn(red)
        ir = self.dequant_before_attn(ir)

        gamma_r, beta_r, gamma_i, beta_i = self.skin_film(x_skin)

        red = red * (1.0 + gamma_r) + beta_r
        # ir = ir * (1.0 + gamma_i) + beta_i

        red_cross = self.red_to_ir(red, ir)
        ir_cross = self.ir_to_red(ir, red)

        fused = torch.cat([red_cross, ir_cross], dim=-1)
        fused = self.temporal_attn(fused)
        fused = self.pool(fused)

        fused = self.quant_before_regressor(fused)
        y_hat = self.regressor(fused)
        y_hat = self.dequant_out(y_hat)

        return y_hat.squeeze(-1)


# =========================================================
# Weight transfer FP32 -> QAT architecture
# =========================================================

def copy_fp32_weights_to_qat(fp32_model: nn.Module, qat_model: nn.Module):
    with torch.no_grad():
        # red encoder
        qat_model.red_encoder.block1[0].weight.copy_(fp32_model.red_encoder.net[0].weight)
        qat_model.red_encoder.block1[0].bias.copy_(fp32_model.red_encoder.net[0].bias)
        qat_model.red_encoder.block1[1].weight.copy_(fp32_model.red_encoder.net[1].weight)
        qat_model.red_encoder.block1[1].bias.copy_(fp32_model.red_encoder.net[1].bias)
        qat_model.red_encoder.block1[1].running_mean.copy_(fp32_model.red_encoder.net[1].running_mean)
        qat_model.red_encoder.block1[1].running_var.copy_(fp32_model.red_encoder.net[1].running_var)
        qat_model.red_encoder.block1[1].num_batches_tracked.copy_(fp32_model.red_encoder.net[1].num_batches_tracked)

        qat_model.red_encoder.block2[0].weight.copy_(fp32_model.red_encoder.net[4].weight)
        qat_model.red_encoder.block2[0].bias.copy_(fp32_model.red_encoder.net[4].bias)
        qat_model.red_encoder.block2[1].weight.copy_(fp32_model.red_encoder.net[5].weight)
        qat_model.red_encoder.block2[1].bias.copy_(fp32_model.red_encoder.net[5].bias)
        qat_model.red_encoder.block2[1].running_mean.copy_(fp32_model.red_encoder.net[5].running_mean)
        qat_model.red_encoder.block2[1].running_var.copy_(fp32_model.red_encoder.net[5].running_var)
        qat_model.red_encoder.block2[1].num_batches_tracked.copy_(fp32_model.red_encoder.net[5].num_batches_tracked)

        qat_model.red_encoder.block3[0].weight.copy_(fp32_model.red_encoder.net[8].weight)
        qat_model.red_encoder.block3[0].bias.copy_(fp32_model.red_encoder.net[8].bias)
        qat_model.red_encoder.block3[1].weight.copy_(fp32_model.red_encoder.net[9].weight)
        qat_model.red_encoder.block3[1].bias.copy_(fp32_model.red_encoder.net[9].bias)
        qat_model.red_encoder.block3[1].running_mean.copy_(fp32_model.red_encoder.net[9].running_mean)
        qat_model.red_encoder.block3[1].running_var.copy_(fp32_model.red_encoder.net[9].running_var)
        qat_model.red_encoder.block3[1].num_batches_tracked.copy_(fp32_model.red_encoder.net[9].num_batches_tracked)

        # ir encoder
        qat_model.ir_encoder.block1[0].weight.copy_(fp32_model.ir_encoder.net[0].weight)
        qat_model.ir_encoder.block1[0].bias.copy_(fp32_model.ir_encoder.net[0].bias)
        qat_model.ir_encoder.block1[1].weight.copy_(fp32_model.ir_encoder.net[1].weight)
        qat_model.ir_encoder.block1[1].bias.copy_(fp32_model.ir_encoder.net[1].bias)
        qat_model.ir_encoder.block1[1].running_mean.copy_(fp32_model.ir_encoder.net[1].running_mean)
        qat_model.ir_encoder.block1[1].running_var.copy_(fp32_model.ir_encoder.net[1].running_var)
        qat_model.ir_encoder.block1[1].num_batches_tracked.copy_(fp32_model.ir_encoder.net[1].num_batches_tracked)

        qat_model.ir_encoder.block2[0].weight.copy_(fp32_model.ir_encoder.net[4].weight)
        qat_model.ir_encoder.block2[0].bias.copy_(fp32_model.ir_encoder.net[4].bias)
        qat_model.ir_encoder.block2[1].weight.copy_(fp32_model.ir_encoder.net[5].weight)
        qat_model.ir_encoder.block2[1].bias.copy_(fp32_model.ir_encoder.net[5].bias)
        qat_model.ir_encoder.block2[1].running_mean.copy_(fp32_model.ir_encoder.net[5].running_mean)
        qat_model.ir_encoder.block2[1].running_var.copy_(fp32_model.ir_encoder.net[5].running_var)
        qat_model.ir_encoder.block2[1].num_batches_tracked.copy_(fp32_model.ir_encoder.net[5].num_batches_tracked)

        qat_model.ir_encoder.block3[0].weight.copy_(fp32_model.ir_encoder.net[8].weight)
        qat_model.ir_encoder.block3[0].bias.copy_(fp32_model.ir_encoder.net[8].bias)
        qat_model.ir_encoder.block3[1].weight.copy_(fp32_model.ir_encoder.net[9].weight)
        qat_model.ir_encoder.block3[1].bias.copy_(fp32_model.ir_encoder.net[9].bias)
        qat_model.ir_encoder.block3[1].running_mean.copy_(fp32_model.ir_encoder.net[9].running_mean)
        qat_model.ir_encoder.block3[1].running_var.copy_(fp32_model.ir_encoder.net[9].running_var)
        qat_model.ir_encoder.block3[1].num_batches_tracked.copy_(fp32_model.ir_encoder.net[9].num_batches_tracked)

        # direct-copy modules
        qat_model.skin_film.load_state_dict(fp32_model.skin_film.state_dict())
        qat_model.red_to_ir.load_state_dict(fp32_model.red_to_ir.state_dict())
        qat_model.ir_to_red.load_state_dict(fp32_model.ir_to_red.state_dict())
        qat_model.temporal_attn.load_state_dict(fp32_model.temporal_attn.state_dict())
        qat_model.pool.load_state_dict(fp32_model.pool.state_dict())
        qat_model.regressor.load_state_dict(fp32_model.regressor.state_dict())


# =========================================================
# QAT helpers
# =========================================================

def prepare_model_for_qat(model: nn.Module):
    model = copy.deepcopy(model)

    torch.backends.quantized.engine = "fbgemm"

    model.eval()
    model.fuse_model()
    model.train()

    model.qconfig = tq.get_default_qat_qconfig("fbgemm")

    # Keep these modules in float
    model.skin_film.qconfig = None
    model.red_to_ir.qconfig = None
    model.ir_to_red.qconfig = None
    model.temporal_attn.qconfig = None
    model.pool.qconfig = None

    tq.prepare_qat(model, inplace=True)
    return model


def convert_qat_model(model: nn.Module):
    model = copy.deepcopy(model).cpu()
    model.eval()

    model.apply(tq.disable_observer)
    model.apply(tq.disable_fake_quant)

    for name, buf in model.named_buffers():
        if buf is not None and buf.device.type != "cpu":
            raise RuntimeError(f"Buffer still not on CPU before convert: {name} -> {buf.device}")

    for name, param in model.named_parameters():
        if param is not None and param.device.type != "cpu":
            raise RuntimeError(f"Parameter still not on CPU before convert: {name} -> {param.device}")

    quantized_model = tq.convert(model, inplace=False)
    return quantized_model


def qat_epoch_updates(model, epoch, num_epochs):
    if epoch == int(num_epochs * 0.6):
        model.apply(tq.disable_observer)
        print("Disabled QAT observers.")

    if epoch == int(num_epochs * 0.8):
        for m in model.modules():
            if isinstance(m, nn.BatchNorm1d):
                m.eval()
        print("Froze BatchNorm stats.")


# =========================================================
# Training helpers
# =========================================================

class RunningAverage:
    def __init__(self):
        self.sum = 0.0
        self.count = 0

    def update(self, value: float, n: int):
        self.sum += value * n
        self.count += n

    @property
    def avg(self):
        return self.sum / max(1, self.count)


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    return {"rmse": rmse, "mae": mae}


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()

    loss_meter = RunningAverage()
    preds_all = []
    targets_all = []

    for x_seq, x_skin, y in loader:
        x_seq = x_seq.to(device)
        x_skin = x_skin.to(device).float()
        y = y.to(device).float()

        optimizer.zero_grad()
        y_hat = model(x_seq, x_skin)
        loss = criterion(y_hat, y)
        loss.backward()
        optimizer.step()

        bs = y.size(0)
        loss_meter.update(loss.item(), bs)

        preds_all.append(y_hat.detach().cpu().numpy())
        targets_all.append(y.detach().cpu().numpy())

    preds_all = np.concatenate(preds_all)
    targets_all = np.concatenate(targets_all)

    metrics = compute_regression_metrics(targets_all, preds_all)
    metrics["loss"] = loss_meter.avg
    return metrics


@torch.no_grad()
def validate_one_epoch(model, loader, criterion, device, y_mean=None, y_std=None):
    model.eval()

    loss_meter = RunningAverage()
    preds_all = []
    targets_all = []

    for x_seq, x_skin, y in loader:
        x_seq = x_seq.to(device)
        x_skin = x_skin.to(device).float()
        y = y.to(device).float()

        y_hat = model(x_seq, x_skin)
        loss = criterion(y_hat, y)

        bs = y.size(0)
        loss_meter.update(loss.item(), bs)

        preds_all.append(y_hat.cpu().numpy())
        targets_all.append(y.cpu().numpy())

    preds_all = np.concatenate(preds_all)
    targets_all = np.concatenate(targets_all)

    if y_mean is not None and y_std is not None:
        preds_eval = preds_all * y_std + y_mean
        targets_eval = targets_all * y_std + y_mean
    else:
        preds_eval = preds_all
        targets_eval = targets_all

    metrics = compute_regression_metrics(targets_eval, preds_eval)
    metrics["loss"] = loss_meter.avg
    return metrics


# =========================================================
# Main
# =========================================================

if __name__ == "__main__":
    set_seed(30)
    device = get_device()

    # -----------------------------
    # Config
    # -----------------------------
    data_dir = "../../data"
    pattern = "*.h5"

    subject_col = "SubjectID"
    time_col = "timestamp"
    target_col = "SpO2_Rad"

    seq_feature_cols = ["red_win_filtered", "ir_win_filtered"]

    window_size = 400
    stride = 400
    val_size = 0.1
    batch_size = 32
    num_workers = 0
    normalize_seq = False

    hidden_dim = 64
    num_heads = 4
    dropout = 0.1

    lr = 1e-3
    weight_decay = 1e-4
    num_epochs = 10

    fp32_checkpoint_path = "best_raw_cross_attention_spo2_skinfilm.pt"
    save_path_qat = "best_raw_cross_attention_spo2_skinfilm_qat_fakequant.pt"
    save_path_quantized = "best_raw_cross_attention_spo2_skinfilm_qat_converted.pt"

    # -----------------------------
    # Load data
    # -----------------------------
    df = load_all_files(data_dir=data_dir, pattern=pattern)

    print("Raw dataframe shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print("Subjects:", df[subject_col].nunique())
    print("\nRaw target stats:")
    print(df[target_col].describe())

    windowed_df = create_windowed_dataset(
        df=df,
        subject_col=subject_col,
        time_col=time_col,
        target_col=target_col,
        seq_feature_cols=seq_feature_cols,
        window_size=window_size,
        stride=stride,
        drop_incomplete=True,
    )

    print("\nWindowed dataframe shape:", windowed_df.shape)
    print("Windowed target example:", windowed_df[target_col].iloc[0])
    print("Windowed skintone example:", windowed_df["skintone"].iloc[0])

    train_df, val_df = subject_wise_split(
        windowed_df=windowed_df,
        subject_col=subject_col,
        val_size=val_size,
        random_state=42,
    )

    print("\nTrain windows:", len(train_df))
    print("Validation windows:", len(val_df))

    # -----------------------------
    # Load normalization stats from saved FP32 checkpoint
    # -----------------------------
    if not os.path.exists(fp32_checkpoint_path):
        raise FileNotFoundError(f"Saved FP32 checkpoint not found: {fp32_checkpoint_path}")

    fp32_ckpt = torch.load(fp32_checkpoint_path, map_location="cpu")

    y_mean = float(fp32_ckpt["y_mean"])
    y_std = float(fp32_ckpt["y_std"])

    print("\nLoaded normalization stats from saved model:")
    print("y_mean:", y_mean)
    print("y_std :", y_std)

    # -----------------------------
    # Datasets / loaders
    # -----------------------------
    train_dataset = RawSpO2WindowDataset(
        windowed_df=train_df,
        seq_feature_cols=seq_feature_cols,
        target_col=target_col,
        seq_len=window_size,
        normalize_seq=normalize_seq,
        y_mean=y_mean,
        y_std=y_std,
    )

    val_dataset = RawSpO2WindowDataset(
        windowed_df=val_df,
        seq_feature_cols=seq_feature_cols,
        target_col=target_col,
        seq_len=window_size,
        normalize_seq=normalize_seq,
        y_mean=y_mean,
        y_std=y_std,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    x_seq_sample, x_skin_sample, y_sample = next(iter(train_loader))
    print("\nBatch shapes:")
    print("x_seq shape:", x_seq_sample.shape)
    print("x_skin shape:", x_skin_sample.shape)
    print("y shape:", y_sample.shape)

    # -----------------------------
    # Build and load original FP32 model
    # -----------------------------
    fp32_model = RawCrossAttentionSpO2Net(
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        dropout=dropout,
    )
    fp32_model.load_state_dict(fp32_ckpt["model_state_dict"])
    print("\nLoaded saved FP32 weights.")

    # -----------------------------
    # Build QAT model and transfer weights
    # -----------------------------
    qat_base_model = RawCrossAttentionSpO2NetQAT(
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        dropout=dropout,
    )
    copy_fp32_weights_to_qat(fp32_model, qat_base_model)
    print("Transferred FP32 weights into QAT model.")

    # -----------------------------
    # Prepare for QAT
    # -----------------------------
    model = prepare_model_for_qat(qat_base_model).to(device)

    criterion = nn.MSELoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
    )

    # -----------------------------
    # QAT fine-tuning
    # -----------------------------
    best_val_rmse = float("inf")
    best_qat_state = None

    for epoch in range(1, num_epochs + 1):
        train_metrics_norm = train_one_epoch(model, train_loader, optimizer, criterion, device)

        val_metrics = validate_one_epoch(
            model,
            val_loader,
            criterion,
            device,
            y_mean=y_mean,
            y_std=y_std,
        )

        scheduler.step(val_metrics["loss"])
        qat_epoch_updates(model, epoch, num_epochs)

        train_eval = validate_one_epoch(
            model,
            train_loader,
            criterion,
            device,
            y_mean=y_mean,
            y_std=y_std,
        )

        print(
            f"Epoch [{epoch:03d}/{num_epochs:03d}] | "
            f"Train Loss(norm): {train_metrics_norm['loss']:.4f} | "
            f"Train MAE: {train_eval['mae']:.4f} | "
            f"Train RMSE: {train_eval['rmse']:.4f} | "
            f"Val Loss(norm): {val_metrics['loss']:.4f} | "
            f"Val MAE: {val_metrics['mae']:.4f} | "
            f"Val RMSE: {val_metrics['rmse']:.4f}"
        )

        if val_metrics["rmse"] < best_val_rmse:
            best_val_rmse = val_metrics["rmse"]

            # Save CPU copy of best QAT state to avoid device mismatch later
            best_qat_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": best_qat_state,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_rmse": best_val_rmse,
                    "seq_feature_cols": seq_feature_cols,
                    "window_size": window_size,
                    "y_mean": y_mean,
                    "y_std": y_std,
                },
                save_path_qat,
            )
            print(f"Saved best fake-quant QAT model to {save_path_qat} (Val RMSE={best_val_rmse:.4f})")

    print(f"\nQAT fine-tuning complete. Best Val RMSE: {best_val_rmse:.4f}")

    # -----------------------------
    # Convert to quantized CPU model
    # -----------------------------
    if best_qat_state is None:
        raise RuntimeError("No best QAT state was saved.")

    # Rebuild a fresh CPU QAT model, then load CPU best state, then convert
    qat_base_model_cpu = RawCrossAttentionSpO2NetQAT(
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        dropout=dropout,
    )
    copy_fp32_weights_to_qat(fp32_model, qat_base_model_cpu)

    model_cpu_for_convert = prepare_model_for_qat(qat_base_model_cpu)
    model_cpu_for_convert.load_state_dict(best_qat_state, strict=True)
    model_cpu_for_convert = model_cpu_for_convert.cpu()
    model_cpu_for_convert.eval()

    quantized_model = convert_qat_model(model_cpu_for_convert)

    torch.save(
        {
            "model_state_dict": quantized_model.state_dict(),
            "seq_feature_cols": seq_feature_cols,
            "window_size": window_size,
            "y_mean": y_mean,
            "y_std": y_std,
            "hidden_dim": hidden_dim,
            "num_heads": num_heads,
            "dropout": dropout,
        },
        save_path_quantized,
    )
    print(f"Saved converted quantized model to {save_path_quantized}")