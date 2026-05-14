// test_desktop.cpp
// -----------------
// Level 2 validation: compile spo2_model.h on x86 and compare
// its output against reference values produced by validate_export.py.
//
// Build:
//   g++ -O2 -std=c++11 -lm -o test_desktop test_desktop.cpp
//
// Run:
//   ./test_desktop reference.bin
//
// reference.bin is produced by validate_export.py --dump_ref reference.bin
// (see instructions below).
//
// The test harness:
//   1. Loads weight arrays from  weights/*.bin  at runtime (no flash headers
//      needed on desktop — we read the files directly).
//   2. Reads test cases from reference.bin (red[T], ir[T], skintone, y_pt).
//   3. Runs spo2_infer() and reports max absolute error vs PyTorch.
//
// ── How to generate reference.bin ────────────────────────────────────────────
// Add  --dump_ref reference.bin  to your validate_export.py call:
//
//   python validate_export.py \
//       --ckpt best_raw_cross_attention_spo2_skinfilm.pt \
//       --weights_dir weights \
//       --h5_dir ../../data \
//       --dump_ref reference.bin
//
// Format (little-endian float32):
//   [n_windows : int32]
//   for each window:
//     [T floats: red] [T floats: ir] [1 float: skintone] [1 float: y_pytorch]

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

// ── Sequence length: must match what you trained with ────────────────────────
#define MODEL_SEQ_LEN      40    // change to 50 if retraining with window_size=50
#define MODEL_HIDDEN_DIM   64
#define MODEL_HIDDEN_DIM2  128
#define MODEL_NUM_HEADS    4
#define MODEL_HEAD_DIM     16
#define MODEL_HEAD_DIM2    32
#define CHUNKED_ATTN_SIZE  8

// ── Include the inference engine ─────────────────────────────────────────────
#include "spo2_model.h"

// ── Runtime weight loader (desktop only — MCU uses flash arrays) ──────────────
static int8_t*  load_i8 (const char* dir, const char* name, size_t n_bytes) {
    char path[512]; snprintf(path, sizeof(path), "%s/%s.bin", dir, name);
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); exit(1); }
    int8_t* buf = (int8_t*)malloc(n_bytes);
    fread(buf, 1, n_bytes, f); fclose(f); return buf;
}
static float*   load_f32(const char* dir, const char* name, size_t n_floats) {
    char path[512]; snprintf(path, sizeof(path), "%s/%s.bin", dir, name);
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); exit(1); }
    float* buf = (float*)malloc(n_floats * sizeof(float));
    fread(buf, sizeof(float), n_floats, f); fclose(f); return buf;
}

// ── Parse meta.h scales at runtime ───────────────────────────────────────────
// Simpler than templating: we just hardcode the scale variable names and
// read them from meta.h using a tiny parser.
#include <map>
#include <string>

static std::map<std::string, float> parse_meta(const char* meta_path) {
    std::map<std::string, float> m;
    FILE* f = fopen(meta_path, "r");
    if (!f) { fprintf(stderr, "Cannot open %s\n", meta_path); exit(1); }
    char line[512];
    while (fgets(line, sizeof(line), f)) {
        if (strncmp(line, "#define SCALE_", 14) == 0 ||
            strncmp(line, "#define Y_MEAN", 14) == 0 ||
            strncmp(line, "#define Y_STD",  13) == 0) {
            char key[256]; float val;
            if (sscanf(line, "#define %s %f", key, &val) == 2) {
                m[std::string(key)] = val;
            }
        }
    }
    fclose(f);
    return m;
}

// ── Build SpO2ModelWeights from runtime-loaded arrays ────────────────────────
static SpO2ModelWeights build_weights(const char* wdir,
                                       std::map<std::string,float>& scales,
                                       float y_mean, float y_std)
{
    int D = MODEL_HIDDEN_DIM, D2 = MODEL_HIDDEN_DIM2;
    SpO2ModelWeights W;
    memset(&W, 0, sizeof(W));

#define S(k)   scales["SCALE_" k]
#define I8(nm, n)   load_i8(wdir, nm, n)
#define F32(nm, n)  load_f32(wdir, nm, n)

    // Conv encoders
    W.red_conv0_w = I8("red_enc_conv0_w", 32*1*5);
    W.red_conv0_b = F32("red_enc_conv0_b", 32);   W.red_conv0_s = S("RED_ENC_CONV0_W");
    W.red_conv1_w = I8("red_enc_conv1_w", D*32*3);
    W.red_conv1_b = F32("red_enc_conv1_b", D);    W.red_conv1_s = S("RED_ENC_CONV1_W");
    W.red_conv2_w = I8("red_enc_conv2_w", D*D*3);
    W.red_conv2_b = F32("red_enc_conv2_b", D);    W.red_conv2_s = S("RED_ENC_CONV2_W");

    W.ir_conv0_w  = I8("ir_enc_conv0_w",  32*1*5);
    W.ir_conv0_b  = F32("ir_enc_conv0_b", 32);    W.ir_conv0_s  = S("IR_ENC_CONV0_W");
    W.ir_conv1_w  = I8("ir_enc_conv1_w",  D*32*3);
    W.ir_conv1_b  = F32("ir_enc_conv1_b", D);     W.ir_conv1_s  = S("IR_ENC_CONV1_W");
    W.ir_conv2_w  = I8("ir_enc_conv2_w",  D*D*3);
    W.ir_conv2_b  = F32("ir_enc_conv2_b", D);     W.ir_conv2_s  = S("IR_ENC_CONV2_W");

    // SkinFiLM
    W.film_fc0_w = I8("film_fc0_w",  32*1);
    W.film_fc0_b = F32("film_fc0_b", 32);   W.film_fc0_s = S("FILM_FC0_W");
    W.film_fc1_w = I8("film_fc1_w",  4*D*32);
    W.film_fc1_b = F32("film_fc1_b", 4*D);  W.film_fc1_s = S("FILM_FC1_W");

    // red_to_ir
    W.r2i_Wq = I8("r2i_Wq",D*D); W.r2i_bq=F32("r2i_bq",D); W.r2i_sq=S("R2I_WQ");
    W.r2i_Wk = I8("r2i_Wk",D*D); W.r2i_bk=F32("r2i_bk",D); W.r2i_sk=S("R2I_WK");
    W.r2i_Wv = I8("r2i_Wv",D*D); W.r2i_bv=F32("r2i_bv",D); W.r2i_sv=S("R2I_WV");
    W.r2i_Wo = I8("r2i_Wo",D*D); W.r2i_bo=F32("r2i_bo",D); W.r2i_so=S("R2I_WO");
    W.r2i_ffn0_w=I8("r2i_ffn0_w",D*2*D); W.r2i_ffn0_b=F32("r2i_ffn0_b",D*2); W.r2i_sf0=S("R2I_FFN0_W");
    W.r2i_ffn1_w=I8("r2i_ffn1_w",D*D*2); W.r2i_ffn1_b=F32("r2i_ffn1_b",D);   W.r2i_sf1=S("R2I_FFN1_W");
    W.r2i_norm1_w=F32("r2i_norm1_weight",D); W.r2i_norm1_b=F32("r2i_norm1_bias",D);
    W.r2i_norm2_w=F32("r2i_norm2_weight",D); W.r2i_norm2_b=F32("r2i_norm2_bias",D);

    // ir_to_red
    W.i2r_Wq = I8("i2r_Wq",D*D); W.i2r_bq=F32("i2r_bq",D); W.i2r_sq=S("I2R_WQ");
    W.i2r_Wk = I8("i2r_Wk",D*D); W.i2r_bk=F32("i2r_bk",D); W.i2r_sk=S("I2R_WK");
    W.i2r_Wv = I8("i2r_Wv",D*D); W.i2r_bv=F32("i2r_bv",D); W.i2r_sv=S("I2R_WV");
    W.i2r_Wo = I8("i2r_Wo",D*D); W.i2r_bo=F32("i2r_bo",D); W.i2r_so=S("I2R_WO");
    W.i2r_ffn0_w=I8("i2r_ffn0_w",D*2*D); W.i2r_ffn0_b=F32("i2r_ffn0_b",D*2); W.i2r_sf0=S("I2R_FFN0_W");
    W.i2r_ffn1_w=I8("i2r_ffn1_w",D*D*2); W.i2r_ffn1_b=F32("i2r_ffn1_b",D);   W.i2r_sf1=S("I2R_FFN1_W");
    W.i2r_norm1_w=F32("i2r_norm1_weight",D); W.i2r_norm1_b=F32("i2r_norm1_bias",D);
    W.i2r_norm2_w=F32("i2r_norm2_weight",D); W.i2r_norm2_b=F32("i2r_norm2_bias",D);

    // temporal self-attention (dim = 2D)
    W.ta_Wq = I8("ta_Wq",D2*D2); W.ta_bq=F32("ta_bq",D2); W.ta_sq=S("TA_WQ");
    W.ta_Wk = I8("ta_Wk",D2*D2); W.ta_bk=F32("ta_bk",D2); W.ta_sk=S("TA_WK");
    W.ta_Wv = I8("ta_Wv",D2*D2); W.ta_bv=F32("ta_bv",D2); W.ta_sv=S("TA_WV");
    W.ta_Wo = I8("ta_Wo",D2*D2); W.ta_bo=F32("ta_bo",D2); W.ta_so=S("TA_WO");
    W.ta_ffn0_w=I8("ta_ffn0_w",D2*2*D2); W.ta_ffn0_b=F32("ta_ffn0_b",D2*2); W.ta_sf0=S("TA_FFN0_W");
    W.ta_ffn1_w=I8("ta_ffn1_w",D2*D2*2); W.ta_ffn1_b=F32("ta_ffn1_b",D2);   W.ta_sf1=S("TA_FFN1_W");
    W.ta_norm1_w=F32("ta_norm1_weight",D2); W.ta_norm1_b=F32("ta_norm1_bias",D2);
    W.ta_norm2_w=F32("ta_norm2_weight",D2); W.ta_norm2_b=F32("ta_norm2_bias",D2);

    // attention pooling
    W.pool_fc0_w=I8("pool_fc0_w",D2*2*D2); W.pool_fc0_b=F32("pool_fc0_b",D2*2); W.pool_fc0_s=S("POOL_FC0_W");
    W.pool_fc1_w=I8("pool_fc1_w",1*D2*2);  W.pool_fc1_b=F32("pool_fc1_b",1);    W.pool_fc1_s=S("POOL_FC1_W");

    // regressor
    W.reg_fc0_w=I8("reg_fc0_w",D*D2);  W.reg_fc0_b=F32("reg_fc0_b",D);  W.reg_fc0_s=S("REG_FC0_W");
    W.reg_fc1_w=I8("reg_fc1_w",16*D);  W.reg_fc1_b=F32("reg_fc1_b",16); W.reg_fc1_s=S("REG_FC1_W");
    W.reg_fc2_w=I8("reg_fc2_w",1*16);  W.reg_fc2_b=F32("reg_fc2_b",1);  W.reg_fc2_s=S("REG_FC2_W");

    W.y_mean = y_mean;
    W.y_std  = y_std;

    return W;
}

int main(int argc, char** argv) {
    const char* ref_path  = argc > 1 ? argv[1] : "reference.bin";
    const char* wdir      = argc > 2 ? argv[2] : "weights";
    const char* meta_path_buf[512];
    char meta_path[512];
    snprintf(meta_path, sizeof(meta_path), "%s/meta.h", wdir);

    // Load scales and normalisation constants
    auto scales = parse_meta(meta_path);
    float y_mean = scales["Y_MEAN"];
    float y_std  = scales["Y_STD"];

    // Build weight struct
    SpO2ModelWeights W = build_weights(wdir, scales, y_mean, y_std);

    // Load reference test cases
    FILE* rf = fopen(ref_path, "rb");
    if (!rf) {
        fprintf(stderr,
            "Cannot open %s.\n"
            "Generate it with:\n"
            "  python validate_export.py --dump_ref reference.bin ...\n", ref_path);
        return 1;
    }

    int32_t n_windows = 0;
    fread(&n_windows, sizeof(int32_t), 1, rf);
    printf("Testing %d windows from %s\n\n", n_windows, ref_path);
    printf("%6s  %10s  %10s  %10s\n", "Window", "PyTorch", "C++", "Diff");
    printf("----------------------------------------------\n");

    float max_diff  = 0.0f;
    float mean_diff = 0.0f;
    int   T = MODEL_SEQ_LEN;

    float* red_buf = (float*)malloc(T * sizeof(float));
    float* ir_buf  = (float*)malloc(T * sizeof(float));

    for (int i = 0; i < n_windows; i++) {
        float skintone, y_ref;
        fread(red_buf,  sizeof(float), T, rf);
        fread(ir_buf,   sizeof(float), T, rf);
        fread(&skintone, sizeof(float), 1, rf);
        fread(&y_ref,    sizeof(float), 1, rf);

        float y_cpp = spo2_infer(red_buf, ir_buf, skintone, &W);
        float diff  = fabsf(y_ref - y_cpp);

        printf("%6d  %10.4f  %10.4f  %10.6f\n", i, y_ref, y_cpp, diff);
        if (diff > max_diff)  max_diff  = diff;
        mean_diff += diff;
    }

    mean_diff /= n_windows;
    printf("----------------------------------------------\n");
    printf("Max error  : %.6f SpO2 %%\n", max_diff);
    printf("Mean error : %.6f SpO2 %%\n", mean_diff);

    float threshold = 0.15f;
    if (max_diff < threshold) {
        printf("\nPASS — max error < %.2f SpO2 %%. Safe to flash to device.\n", threshold);
    } else {
        printf("\nFAIL — max error >= %.2f SpO2 %%.\n", threshold);
        printf("Check spo2_model.h for indexing bugs (conv padding, MHA head splitting).\n");
    }

    fclose(rf);
    free(red_buf); free(ir_buf);
    return (max_diff < threshold) ? 0 : 1;
}