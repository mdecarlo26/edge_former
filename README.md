# edge_former

Real-time SpO2 (blood oxygen) inference on a **Seeed XIAO nRF52840** microcontroller. A cross-attention transformer is trained in PyTorch, quantised to INT8, exported to plain C, and flashed to the device — no runtime dependencies, no cloud.

---

## Table of Contents

1. [Overview](#overview)
2. [Model Architecture](#model-architecture)
3. [Repository Layout](#repository-layout)
4. [Requirements](#requirements)
5. [End-to-End Workflow](#end-to-end-workflow)
   - [Step 1 — Train the FP32 model](#step-1--train-the-fp32-model)
   - [Step 2 — (Optional) QAT fine-tuning](#step-2--optional-qat-fine-tuning)
   - [Step 3 — Export weights](#step-3--export-weights)
   - [Step 4 — Generate C array headers](#step-4--generate-c-array-headers)
   - [Step 5 — Validate on desktop](#step-5--validate-on-desktop)
   - [Step 6 — Generate embedded test data](#step-6--generate-embedded-test-data)
   - [Step 7 — Flash to device](#step-7--flash-to-device)
6. [Memory Budget](#memory-budget)
7. [Validation Chain](#validation-chain)
8. [Configuration Reference](#configuration-reference)

---

## Overview

The model takes two raw PPG signal windows (red + IR channels, length T) and a scalar skin tone value, and predicts SpO2 in percent. It is designed to be skin-tone-aware via a FiLM conditioning module, and targets the FDA/ISO ±2% SpO2 accuracy tolerance after INT8 quantisation.

**Data:** DOVE Hypoxia Study `.h5` files (10 subjects), each containing `red_win_filtered`, `ir_win_filtered`, `SpO2_Rad`, `skintone`, and `timestamp` columns.

---

## Model Architecture

```
  red[T]  ──► ConvEncoder ──► red_tok [T_out, 64]
                                      │
  skintone ─► SkinFiLM ──► γ, β ──► FiLM ──► red_tok_film [T_out, 64]
                                                     │                │
  ir[T]  ───► ConvEncoder ──► ir_tok [T_out, 64]    │                │
                                      │              ▼                ▼
                               ┌──────┴──────────────────────────────┐
                               │  Cross-Attention (bidirectional)     │
                               │                                      │
                               │  r2i: q=red_film, kv=ir  ──► red'   │
                               │  i2r: q=ir,       kv=red_film ► ir' │
                               └──────┬───────────────────────┬──────┘
                                      │                       │
                                      └──────── cat ──────────┘
                                               │
                                        fused [T_out, 128]
                                               │
                                   TemporalSelfAttention
                                               │
                                      fused [T_out, 128]
                                               │
                                    AttentionPooling
                                               │
                                          vec [128]
                                               │
                                      Regressor MLP
                                               │
                                          SpO2 (%)
```

**ConvEncoder** — 3× (Conv1d → BatchNorm → ReLU). First block (k=5, pad=1) reduces sequence length by 2; subsequent blocks (k=3, pad=1) preserve length. BN is fused into Conv weights offline before export.

**SkinFiLM** — Two-layer MLP mapping scalar skin tone → (γ, β) vectors of dim 64. Applied as `red_tok = red_tok * (1 + γ) + β`.

**Cross-attention** — Standard multi-head attention (4 heads, dim=64). Both directions use a saved copy of the pre-cross-attended tokens as key/value to avoid a data dependency bug.

**Temporal self-attention** — MHA over the concatenated [T, 128] stream (4 heads, dim=128).

**Attention pooling** — Learned scalar score per token → softmax → weighted sum → [128].

**Regressor** — Linear(128→64) → ReLU → Linear(64→16) → ReLU → Linear(16→1), output de-normalised with training-set y_mean / y_std.

---

## Repository Layout

```
edge_former/
├── data/
│   └── Subject*_DOVE_Hypoxia_Study_RValue_RawPPG.h5
└── src/
    ├── python/
    │   ├── test12.py                  # FP32 training script
    │   └── Qtest.py                   # QAT fine-tuning script
    ├── C/
    │   └── spo2_model.h               # Single-file C inference engine
    ├── export_weights.py              # PyTorch → INT8 .bin + meta.h
    ├── generate_weight_headers.py     # .bin → weights_flash.h / weights_init.h
    ├── gen_test_data_header.py        # reference.bin → test_data.h
    ├── validate.py                    # NumPy vs PyTorch correctness check
    ├── test_desktop.cpp               # x86 C++ validation harness
    ├── weights/                       # Generated weight blobs (gitignored in practice)
    │   ├── meta.h
    │   ├── weights_flash.h
    │   ├── weights_init.h
    │   └── *.bin
    └── spo2_inference/                # Arduino sketch
        ├── spo2_inference.ino
        ├── spo2_model.h
        ├── test_data.h
        └── weights/
            ├── meta.h
            ├── weights_flash.h
            └── weights_init.h
```

---

## Requirements

**Python (training & export)**
```
torch >= 2.0
numpy
pandas
tables          # for pd.read_hdf
scikit-learn
```

**C++ desktop validation**
```
g++ with C++11 support
```

**Arduino / embedded**
- Board: Seeed XIAO nRF52840 Sense
- Board manager: *Seeed nRF52 Boards* (search in Arduino Board Manager)
- Serial monitor: 115200 baud

---

## End-to-End Workflow

### Step 1 — Train the FP32 model

```bash
cd src/python
python test12.py
```

Trains `RawCrossAttentionSpO2Net` for 100 epochs with subject-wise train/val split (90/10). Saves the best checkpoint by validation RMSE to:

```
best_raw_cross_attention_spo2_skinfilm.pt
```

Key hyperparameters (edit at the top of `test12.py`):

| Parameter | Default | Notes |
|---|---|---|
| `window_size` | 400 | Reduce to 50 to fit nRF52840 RAM |
| `stride` | 400 | Non-overlapping windows |
| `hidden_dim` | 64 | Transformer width |
| `num_epochs` | 100 | |
| `lr` | 1e-3 | AdamW |

---

### Step 2 — (Optional) QAT fine-tuning

Quantisation-aware training recovers accuracy lost from INT8 quantisation. Requires a trained FP32 checkpoint from Step 1.

```bash
python Qtest.py
```

Runs for 10 epochs. Saves:
- `best_raw_cross_attention_spo2_skinfilm_qat_fakequant.pt` — best fake-quant state
- `best_raw_cross_attention_spo2_skinfilm_qat_converted.pt` — fully converted INT8 model

If skipping QAT, use the FP32 checkpoint directly in Step 3 (INT8 quantisation is still applied at export time).

---

### Step 3 — Export weights

Fuses BatchNorm into Conv weights, quantises all weight matrices to symmetric INT8, and writes binary blobs plus a C header with scales and normalisation constants.

```bash
cd src
python export_weights.py \
    --ckpt python/best_raw_cross_attention_spo2_skinfilm.pt \
    --out weights
```

Outputs in `weights/`:
- `*.bin` — INT8 weight matrices and FP32 bias / LayerNorm blobs
- `meta.h` — `#define` macros for shapes, per-tensor scales, `Y_MEAN`, `Y_STD`

To inspect exported shapes without re-running export:
```bash
python export_weights.py --diag --out weights
```

---

### Step 4 — Generate C array headers

Converts the `.bin` files into C `static const` arrays suitable for flash storage on the microcontroller.

```bash
python generate_weight_headers.py --weights_dir weights
```

Outputs:
- `weights/weights_flash.h` — all weight arrays as `static const int8_t` / `static const float`
- `weights/weights_init.h` — `spo2_weights_init()` function that fills the `SpO2ModelWeights` struct

---

### Step 5 — Validate on desktop

Two-level validation before touching the device.

**Level 1 — NumPy vs PyTorch** (catches logic bugs in the export):
```bash
python validate.py \
    --ckpt python/best_raw_cross_attention_spo2_skinfilm.pt \
    --weights_dir weights \
    --h5_dir ../data \
    --n_windows 20 \
    --dump_ref reference.bin
```

Expected: max error < 2.0 SpO2 % (INT8 quantisation noise).
Run with `--fp32_test` to verify the NumPy logic is exact before quantisation (expected < 0.01%).

**Level 2 — C++ x86 vs PyTorch** (catches bugs in `spo2_model.h`):
```bash
g++ -O2 -std=c++11 -lm -o test_desktop test_desktop.cpp
./test_desktop reference.bin weights
```

Expected: max error < 0.15 SpO2 % (pass threshold hardcoded in `test_desktop.cpp`).

---

### Step 6 — Generate embedded test data

Embeds a small number of real PPG windows and their PyTorch reference outputs directly into the Arduino sketch for on-device self-test.

```bash
python gen_test_data_header.py \
    --ref reference.bin \
    --out spo2_inference/test_data.h \
    --seq_len 400 \
    --max_windows 5 \
    --tolerance 2.0
```

Flash cost: ~3.2 KB per window at T=400; ~0.4 KB per window at T=50.

---

### Step 7 — Flash to device

1. Copy the generated headers into the sketch folder:
   ```
   spo2_inference/weights/meta.h
   spo2_inference/weights/weights_flash.h
   spo2_inference/weights/weights_init.h
   spo2_inference/test_data.h
   ```
2. Open `spo2_inference/spo2_inference.ino` in the Arduino IDE.
3. Set **Tools → Board → Seeed XIAO nRF52840 Sense**.
4. Upload. Open Serial Monitor at 115200 baud.

On boot the sketch replays all embedded test windows and prints PASS/FAIL per window. If any window fails, the sketch halts — do not deploy a failed device. After all tests pass it enters the live inference loop (replace the `read_ppg_window()` stub with your actual sensor driver, e.g. MAX30102 over I2C).

---

## Memory Budget

| Buffer | Size at T=400, D=64 |
|---|---|
| Red encoder tokens | 398 × 64 × 4 B = **102 KB** |
| IR encoder tokens | 398 × 64 × 4 B = **102 KB** |
| Fused tokens | 398 × 128 × 4 B = **204 KB** |
| MHA scratch (proj Q/K/V) | 398 × 128 × 4 B × 3 = **611 KB** |
| INT8 weight flash | ~**180 KB** |
| FP32 bias / LN flash | ~**30 KB** |

> ⚠️ **T=400 exceeds the nRF52840's 256 KB RAM.** Retrain with `window_size=50` (T_out=48): encoder buffers drop to ~24 KB total and the full model fits comfortably. Update `MODEL_SEQ_LEN` in `spo2_inference.ino` and `test_desktop.cpp` to match.

---

## Validation Chain

```
PyTorch training
      │
      ▼
export_weights.py  (BN fusion + INT8 quantisation)
      │
      ▼
validate.py  ──── Level 1: NumPy vs PyTorch  ────  max error < 2.0%
      │
      ▼
test_desktop.cpp ── Level 2: C++ x86 vs PyTorch ── max error < 0.15%
      │
      ▼
spo2_inference.ino ─ Level 3: on-device self-test ─ PASS / HALT
```

---

## Configuration Reference

| `#define` / argument | Where | Default | Description |
|---|---|---|---|
| `MODEL_SEQ_LEN` | sketch / `.cpp` | 400 | Window length T — must match training |
| `CHUNKED_ATTN_SIZE` | `spo2_model.h` | 8 | Query rows per attention chunk; reduce to save RAM |
| `NORMALIZE_INPUT` | sketch | `false` | Set `true` only if trained with `normalize_seq=True` |
| `--window_size` | `test12.py` | 400 | Sequence length for training |
| `--hidden_dim` | `test12.py` | 64 | Transformer hidden dimension |
| `--val_size` | `test12.py` | 0.1 | Fraction of subjects held out for validation |
| `--num_epochs` | `test12.py` | 100 | Training epochs (QAT uses 10) |