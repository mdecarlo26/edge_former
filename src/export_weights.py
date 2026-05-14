"""
export_weights.py
-----------------
Loads the trained FP32 checkpoint (best_raw_cross_attention_spo2_skinfilm.pt),
fuses BatchNorm into Conv1d weights offline, quantises every weight tensor to
INT8, and writes:

  weights/
    meta.h          -- C header: shapes, scales, zero-points, y_mean/y_std
    *.bin           -- raw INT8 weight blobs (one file per layer)

Run once on your PC after training:
    python export_weights.py --ckpt best_raw_cross_attention_spo2_skinfilm.pt

The generated weights/ folder is then copied into your Arduino/C++ project.
"""

import argparse
import os
import struct
import numpy as np
import torch
import torch.nn as nn


# ── Minimal model re-definition (must match test12.py exactly) ───────────────

class ConvTokenEncoder(nn.Module):
    def __init__(self, in_channels=1, hidden_dim=64, dropout=0.1):
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
        return self.net(x).transpose(1, 2)


class CrossAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(dim * 2, dim)
        )
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, kv):
        attn_out, _ = self.attn(q, kv, kv)
        x = self.norm1(q + self.dropout(attn_out))
        return self.norm2(x + self.dropout(self.ffn(x)))


class TemporalSelfAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(dim * 2, dim)
        )
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        return self.norm2(x + self.dropout(self.ffn(x)))


class AttentionPooling(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(dim, 2 * dim), nn.Tanh(), nn.Linear(2 * dim, 1))

    def forward(self, x):
        w = torch.softmax(self.score(x), dim=1)
        return (x * w).sum(dim=1)


class SkinFiLM(nn.Module):
    def __init__(self, token_dim, hidden_dim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 2 * hidden_dim), nn.ReLU(), nn.Linear(2 * hidden_dim, 4 * token_dim)
        )

    def forward(self, x_skin):
        if x_skin.ndim == 1:
            x_skin = x_skin.unsqueeze(-1)
        params = self.net(x_skin)
        gamma_r, beta_r, gamma_i, beta_i = torch.chunk(params, 4, dim=-1)
        return gamma_r.unsqueeze(1), beta_r.unsqueeze(1), gamma_i.unsqueeze(1), beta_i.unsqueeze(1)


class RawCrossAttentionSpO2Net(nn.Module):
    def __init__(self, hidden_dim=64, num_heads=4, dropout=0.1):
        super().__init__()
        self.red_encoder  = ConvTokenEncoder(1, hidden_dim, dropout)
        self.ir_encoder   = ConvTokenEncoder(1, hidden_dim, dropout)
        self.skin_film    = SkinFiLM(hidden_dim, 16)
        self.red_to_ir    = CrossAttentionBlock(hidden_dim, num_heads, dropout)
        self.ir_to_red    = CrossAttentionBlock(hidden_dim, num_heads, dropout)
        self.temporal_attn = TemporalSelfAttentionBlock(hidden_dim * 2, num_heads, dropout)
        self.pool         = AttentionPooling(hidden_dim * 2)
        self.regressor    = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, 16), nn.ReLU(), nn.Linear(16, 1)
        )

    def forward(self, x_seq, x_skin):
        red = self.red_encoder(x_seq[:, 0:1, :])
        ir  = self.ir_encoder(x_seq[:, 1:2, :])
        gamma_r, beta_r, _, _ = self.skin_film(x_skin)
        red = red * (1.0 + gamma_r) + beta_r
        fused = torch.cat([self.red_to_ir(red, ir), self.ir_to_red(ir, red)], dim=-1)
        fused = self.pool(self.temporal_attn(fused))
        return self.regressor(fused).squeeze(-1)


# ── BatchNorm fusion ──────────────────────────────────────────────────────────

def fuse_conv_bn(conv: nn.Conv1d, bn: nn.BatchNorm1d):
    """
    Returns new (weight, bias) with BN absorbed into the Conv.
    Formula:
        w_fused = w * (gamma / sqrt(var + eps))
        b_fused = (b - mean) * (gamma / sqrt(var + eps)) + beta
    """
    w = conv.weight.detach().float()           # [C_out, C_in, K]
    b = conv.bias.detach().float() if conv.bias is not None else torch.zeros(w.shape[0])

    gamma  = bn.weight.detach().float()
    beta   = bn.bias.detach().float()
    mean   = bn.running_mean.detach().float()
    var    = bn.running_var.detach().float()
    eps    = bn.eps

    scale  = gamma / torch.sqrt(var + eps)     # [C_out]
    w_fused = w * scale[:, None, None]
    b_fused = (b - mean) * scale + beta
    return w_fused.numpy(), b_fused.numpy()


# ── Per-tensor symmetric INT8 quantisation ────────────────────────────────────

def quantise(arr: np.ndarray, name: str):
    """
    Symmetric per-tensor quantisation to INT8.
    scale = max(|arr|) / 127
    No zero-point (always 0 for symmetric).
    Returns (int8_array, scale_float32).
    """
    amax = np.abs(arr).max()
    if amax < 1e-8:
        amax = 1e-8
    scale = amax / 127.0
    q = np.clip(np.round(arr / scale), -127, 127).astype(np.int8)
    actual_amax = np.abs(arr).max()
    print(f"  {name:50s}  shape={str(arr.shape):20s}  scale={scale:.6e}  "
          f"amax={actual_amax:.4f}")
    return q, float(scale)


# ── Write helpers ─────────────────────────────────────────────────────────────

def write_bin(path: str, arr: np.ndarray):
    arr.tofile(path)


def extract_mha_weights(mha: nn.MultiheadAttention, prefix: str, D: int):
    """
    nn.MultiheadAttention stores fused in_proj_weight [3D, D] and out_proj.
    Returns two separate dicts: one for weight matrices (INT8), one for biases (FP32).
    Keys are distinct: weight keys end in _Wq/_Wk/_Wv/_Wo,
                       bias keys end in   _bq/_bk/_bv/_bo.
    """
    W = mha.in_proj_weight.detach().float().numpy()   # [3D, D]
    b = mha.in_proj_bias.detach().float().numpy()     # [3D]
    W_q, W_k, W_v = W[:D], W[D:2*D], W[2*D:]
    b_q, b_k, b_v = b[:D], b[D:2*D], b[2*D:]
    W_o = mha.out_proj.weight.detach().float().numpy()  # [D, D]
    b_o = mha.out_proj.bias.detach().float().numpy()    # [D]
    weight_dict = {
        f"{prefix}_Wq": W_q,
        f"{prefix}_Wk": W_k,
        f"{prefix}_Wv": W_v,
        f"{prefix}_Wo": W_o,
    }
    bias_dict = {
        f"{prefix}_bq": b_q,
        f"{prefix}_bk": b_k,
        f"{prefix}_bv": b_v,
        f"{prefix}_bo": b_o,
    }
    return weight_dict, bias_dict


def extract_ffn_weights(ffn: nn.Sequential, prefix: str):
    # ffn = Linear, ReLU, Dropout, Linear
    # Returns (weight_dict, bias_dict) with distinct keys to avoid overwrite collisions.
    linears = [m for m in ffn if isinstance(m, nn.Linear)]
    weight_dict = {
        f"{prefix}_ffn0_w": linears[0].weight.detach().float().numpy(),
        f"{prefix}_ffn1_w": linears[1].weight.detach().float().numpy(),
    }
    bias_dict = {
        f"{prefix}_ffn0_b": linears[0].bias.detach().float().numpy(),
        f"{prefix}_ffn1_b": linears[1].bias.detach().float().numpy(),
    }
    return weight_dict, bias_dict


def extract_layernorm(ln: nn.LayerNorm, prefix: str):
    return {
        f"{prefix}_weight": ln.weight.detach().float().numpy(),
        f"{prefix}_bias":   ln.bias.detach().float().numpy(),
    }


# ── Main export ───────────────────────────────────────────────────────────────

def export(ckpt_path: str, out_dir: str, hidden_dim: int = 64, num_heads: int = 4):
    os.makedirs(out_dir, exist_ok=True)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = RawCrossAttentionSpO2Net(hidden_dim=hidden_dim, num_heads=num_heads, dropout=0.0)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    y_mean = float(ckpt["y_mean"])
    y_std  = float(ckpt["y_std"])
    D      = hidden_dim
    D2     = hidden_dim * 2

    weights   = {}   # name -> float32 np array (already fused where needed)
    scales    = {}   # name -> float32 scale
    # LayerNorm and bias terms stay FP32 (tiny, and LN is sensitive to quantisation)
    fp32_blobs = {}  # name -> float32 np array

    # ── 1. Conv encoders (BN fused) ──────────────────────────────────────────
    for enc_name, enc in [("red_enc", model.red_encoder), ("ir_enc", model.ir_encoder)]:
        net = enc.net
        # block 1: Conv(indices 0) + BN(1)
        w, b = fuse_conv_bn(net[0], net[1])
        weights[f"{enc_name}_conv0_w"] = w   # [32, 1, 5]
        weights[f"{enc_name}_conv0_b"] = b   # [32]
        # block 2
        w, b = fuse_conv_bn(net[4], net[5])
        weights[f"{enc_name}_conv1_w"] = w   # [64, 32, 3]
        weights[f"{enc_name}_conv1_b"] = b
        # block 3
        w, b = fuse_conv_bn(net[8], net[9])
        weights[f"{enc_name}_conv2_w"] = w   # [64, 64, 3]
        weights[f"{enc_name}_conv2_b"] = b

    # ── 2. SkinFiLM ──────────────────────────────────────────────────────────
    film_linears = [m for m in model.skin_film.net if isinstance(m, nn.Linear)]
    weights["film_fc0_w"] = film_linears[0].weight.detach().float().numpy()  # [32, 1]
    weights["film_fc0_b"] = film_linears[0].bias.detach().float().numpy()
    weights["film_fc1_w"] = film_linears[1].weight.detach().float().numpy()  # [256, 32]
    weights["film_fc1_b"] = film_linears[1].bias.detach().float().numpy()

    # ── 3. CrossAttention blocks (red_to_ir, ir_to_red) ──────────────────────
    for blk_name, blk in [("r2i", model.red_to_ir), ("i2r", model.ir_to_red)]:
        mha_w, mha_b = extract_mha_weights(blk.attn, blk_name, D)
        weights.update(mha_w)
        fp32_blobs.update(mha_b)
        ffn_w, ffn_b = extract_ffn_weights(blk.ffn, blk_name)
        weights.update(ffn_w)
        fp32_blobs.update(ffn_b)
        fp32_blobs.update(extract_layernorm(blk.norm1, f"{blk_name}_norm1"))
        fp32_blobs.update(extract_layernorm(blk.norm2, f"{blk_name}_norm2"))

    # ── 4. TemporalSelfAttention (dim = 2D = 128) ────────────────────────────
    blk = model.temporal_attn
    mha_w, mha_b = extract_mha_weights(blk.attn, "ta", D2)
    weights.update(mha_w)
    fp32_blobs.update(mha_b)
    ffn_w, ffn_b = extract_ffn_weights(blk.ffn, "ta")
    weights.update(ffn_w)
    fp32_blobs.update(ffn_b)
    fp32_blobs.update(extract_layernorm(blk.norm1, "ta_norm1"))
    fp32_blobs.update(extract_layernorm(blk.norm2, "ta_norm2"))

    # ── 5. AttentionPooling ───────────────────────────────────────────────────
    pool_linears = [m for m in model.pool.score if isinstance(m, nn.Linear)]
    weights["pool_fc0_w"] = pool_linears[0].weight.detach().float().numpy()  # [2D*2, 2D]
    weights["pool_fc0_b"] = pool_linears[0].bias.detach().float().numpy()
    weights["pool_fc1_w"] = pool_linears[1].weight.detach().float().numpy()  # [1, 2D*2]
    weights["pool_fc1_b"] = pool_linears[1].bias.detach().float().numpy()

    # ── 6. Regressor MLP ─────────────────────────────────────────────────────
    reg_linears = [m for m in model.regressor if isinstance(m, nn.Linear)]
    weights["reg_fc0_w"] = reg_linears[0].weight.detach().float().numpy()  # [64, 128]
    weights["reg_fc0_b"] = reg_linears[0].bias.detach().float().numpy()
    weights["reg_fc1_w"] = reg_linears[1].weight.detach().float().numpy()  # [16, 64]
    weights["reg_fc1_b"] = reg_linears[1].bias.detach().float().numpy()
    weights["reg_fc2_w"] = reg_linears[2].weight.detach().float().numpy()  # [1, 16]
    weights["reg_fc2_b"] = reg_linears[2].bias.detach().float().numpy()

    # ── Quantise weight matrices (not biases — biases stay FP32) ─────────────
    print("\n=== Quantising weight matrices ===")
    quant_weights = {}
    for name, arr in weights.items():
        if name.endswith("_b"):
            # biases stay FP32
            fp32_blobs[name] = arr
        else:
            q, s = quantise(arr, name)
            quant_weights[name] = q
            scales[name] = s

    # ── Write binary blobs ────────────────────────────────────────────────────
    print(f"\n=== Writing blobs to {out_dir}/ ===")
    for name, arr in quant_weights.items():
        path = os.path.join(out_dir, name + ".bin")
        write_bin(path, arr)
        print(f"  {path}  ({arr.nbytes} bytes)")

    for name, arr in fp32_blobs.items():
        path = os.path.join(out_dir, name + ".bin")
        write_bin(path, arr.astype(np.float32))
        print(f"  {path}  ({arr.nbytes*4} bytes, fp32)")

    # ── Generate meta.h ───────────────────────────────────────────────────────
    lines = [
        "// Auto-generated by export_weights.py — do not edit",
        "#pragma once",
        "#include <stdint.h>",
        "",
        "// ── Model hyper-parameters ──────────────────────────────",
        f"#define MODEL_HIDDEN_DIM   {D}",
        f"#define MODEL_HIDDEN_DIM2  {D2}",
        f"#define MODEL_NUM_HEADS    {num_heads}",
        f"#define MODEL_HEAD_DIM     {D // num_heads}",
        f"#define MODEL_HEAD_DIM2    {D2 // num_heads}",
        f"#define MODEL_SEQ_LEN      400",
        "",
        "// ── Target normalisation ────────────────────────────────",
        f"#define Y_MEAN  {y_mean}f",
        f"#define Y_STD   {y_std}f",
        "",
        "// ── INT8 weight scales (symmetric, zero_point = 0) ──────",
    ]
    for name, s in scales.items():
        macro = "SCALE_" + name.upper()
        lines.append(f"#define {macro:<55s} {s:.8e}f")

    lines += [
        "",
        "// ── Conv encoder output length after padding ────────────",
        "//  Conv1d(k=5, pad=1): T_out = T_in - 2  (loses 2 samples)",
        "//  Conv1d(k=3, pad=1): T_out = T_in       (same)",
        "//  So after 3 conv blocks: T_out = 400 - 2 = 398",
        "#define CONV_OUT_LEN  398",
        "",
    ]

    header_path = os.path.join(out_dir, "meta.h")
    with open(header_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nWrote {header_path}")

    # Summary
    total_int8  = sum(q.nbytes for q in quant_weights.values())
    total_fp32  = sum(a.nbytes * 4 for a in fp32_blobs.values() if a.dtype != np.float32)
    total_fp32 += sum(a.nbytes for a in fp32_blobs.values() if a.dtype == np.float32)
    print(f"\n=== Size summary ===")
    print(f"  INT8 weight blobs : {total_int8 / 1024:.1f} KB")
    print(f"  FP32 blobs (LN, biases): {total_fp32 / 1024:.1f} KB")
    print(f"  Total flash usage : {(total_int8 + total_fp32) / 1024:.1f} KB")
    print(f"  (nRF52840 on-chip flash: 1024 KB  +  QSPI: 2048 KB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="best_raw_cross_attention_spo2_skinfilm.pt")
    parser.add_argument("--out",  default="weights")
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_heads",  type=int, default=4)
    parser.add_argument("--diag", action="store_true",
                        help="Print all exported .bin shapes and exit (for debugging).")
    args = parser.parse_args()

    if args.diag:
        import glob
        bins = sorted(glob.glob(os.path.join(args.out, "*.bin")))
        if not bins:
            print(f"No .bin files found in {args.out}/  (run without --diag first)")
        else:
            print(f"{'File':<45}  {'Bytes':>8}  {'Elements':>10}  {'Sqrt':>8}")
            print("-" * 78)
            for b in bins:
                sz = os.path.getsize(b)
                name = os.path.basename(b)
                is_fp32 = (name.endswith("_b.bin") or "norm" in name or
                           name.endswith("_bq.bin") or name.endswith("_bk.bin") or
                           name.endswith("_bv.bin") or name.endswith("_bo.bin"))
                elems = sz // 4 if is_fp32 else sz
                sq = int(elems**0.5)
                print(f"{name:<45}  {sz:>8}  {elems:>10}  {sq:>8}")
    else:
        export(args.ckpt, args.out, args.hidden_dim, args.num_heads)