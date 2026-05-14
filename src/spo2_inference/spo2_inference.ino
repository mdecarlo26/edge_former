// spo2_inference.ino
// ------------------
// Arduino sketch for Seeed XIAO nRF52840 Sense.
//
// On startup: replays real PPG windows from test_data.h, compares the C++
// inference output against the PyTorch reference values, and prints PASS/FAIL
// for each window over Serial.  If any window fails, the LED turns red and
// the sketch halts — do not trust a failed device.
//
// After all tests pass: enters the live sensor loop (replace read_ppg_window()
// with your actual sensor driver).
//
// ── File layout ──────────────────────────────────────────────────────────────
//   spo2_inference.ino
//   spo2_model.h
//   test_data.h              <- from gen_test_data_header.py
//   weights/
//     meta.h                 <- from export_weights.py
//     weights_flash.h        <- from gen_weight_headers.py
//     weights_init.h         <- from gen_weight_headers.py
//
// ── Generate test_data.h on your PC ──────────────────────────────────────────
//   python validate_export.py --ckpt ... --h5_dir ... --dump_ref reference.bin
//   python gen_test_data_header.py --ref reference.bin --seq_len 400 --max_windows 5
//
// ── Board setup ──────────────────────────────────────────────────────────────
//   Board Manager: search "Seeed nRF52", install "Seeed nRF52 Boards"
//   Tools -> Board -> Seeed XIAO nRF52840 Sense
//   Serial monitor: 115200 baud

// ── Model config — must match your training window_size ──────────────────────
#define MODEL_SEQ_LEN     40   // change to 50 if retrained with window_size=50
#define CHUNKED_ATTN_SIZE 8

#include "spo2_model.h"
#include "weights/weights_init.h"   // pulls in meta.h + weights_flash.h
#include "test_data.h"              // embedded real PPG windows + expected SpO2

// ── Sanity check: seq lens must agree ────────────────────────────────────────
#if MODEL_SEQ_LEN != TEST_SEQ_LEN
#  error "MODEL_SEQ_LEN and TEST_SEQ_LEN differ. Re-run gen_test_data_header.py with --seq_len matching MODEL_SEQ_LEN."
#endif

// ── Global weight struct ──────────────────────────────────────────────────────
static SpO2ModelWeights g_weights;

// ── Working input buffers ─────────────────────────────────────────────────────
static float g_red[MODEL_SEQ_LEN];
static float g_ir [MODEL_SEQ_LEN];

// ── Optional per-window normalisation ────────────────────────────────────────
// Set true only if normalize_seq=True was used during training.
#define NORMALIZE_INPUT false

static void normalize_window(float* buf, int len) {
    float mean = 0.0f;
    for (int i = 0; i < len; i++) mean += buf[i];
    mean /= (float)len;
    float var = 0.0f;
    for (int i = 0; i < len; i++) { float d = buf[i] - mean; var += d * d; }
    var /= (float)len;
    float inv_std = 1.0f / sqrtf(var + 1e-8f);
    for (int i = 0; i < len; i++) buf[i] = (buf[i] - mean) * inv_std;
}

// ── Self-test: run all embedded windows and verify against PyTorch refs ───────
// Returns true if every window passes within TEST_TOLERANCE SpO2 %.
static bool run_self_test() {
    Serial.println("========================================");
    Serial.println("  Self-test: replaying embedded windows");
    Serial.println("========================================");
    Serial.print  ("  Windows   : "); Serial.println(TEST_N_WINDOWS);
    Serial.print  ("  Tolerance : +/-"); Serial.print(TEST_TOLERANCE, 1);
    Serial.println(" SpO2 %");
    Serial.println();
    Serial.println("  Win   Expected     Got        Diff   Result");
    Serial.println("  ---   --------   --------   ------   ------");

    bool all_pass = true;
    float max_diff = 0.0f;

    for (int w = 0; w < TEST_N_WINDOWS; w++) {
        // Copy window from flash into RAM working buffers
        memcpy(g_red, test_red_windows[w], MODEL_SEQ_LEN * sizeof(float));
        memcpy(g_ir,  test_ir_windows[w],  MODEL_SEQ_LEN * sizeof(float));

        if (NORMALIZE_INPUT) {
            normalize_window(g_red, MODEL_SEQ_LEN);
            normalize_window(g_ir,  MODEL_SEQ_LEN);
        }

        float skintone = test_skintone[w];
        float expected = test_expected[w];

        uint32_t t0 = micros();
        float got = spo2_infer(g_red, g_ir, skintone, &g_weights);
        uint32_t elapsed_us = micros() - t0;

        float diff = got - expected;
        if (diff < 0.0f) diff = -diff;
        if (diff > max_diff) max_diff = diff;

        bool pass = (diff <= TEST_TOLERANCE);
        if (!pass) all_pass = false;

        Serial.print("  ");
        Serial.print(w);
        Serial.print("   ");
        Serial.print(expected, 2);
        Serial.print("   ");
        Serial.print(got, 2);
        Serial.print("   ");
        Serial.print(diff, 4);
        Serial.print("   ");
        Serial.print(pass ? "PASS" : "FAIL");
        Serial.print("  (");
        Serial.print(elapsed_us / 1000.0f, 1);
        Serial.println(" ms)");
    }

    Serial.println();
    Serial.print("  Max error : "); Serial.print(max_diff, 4); Serial.println(" SpO2 %");
    Serial.println(all_pass ? "  Result   : ALL PASS" : "  Result   : FAIL -- halting");
    Serial.println("========================================");
    Serial.println();

    return all_pass;
}

// ── Sensor stub — replace with your actual PPG driver ────────────────────────
// Return true when a full window of MODEL_SEQ_LEN samples is ready.
static bool read_ppg_window(float* red, float* ir) {
    // TODO: replace with real sensor driver, e.g. MAX30102 over I2C:
    //
    // static int idx = 0;
    // while (particleSensor.available()) {
    //     red[idx] = (float)particleSensor.getRed();
    //     ir [idx] = (float)particleSensor.getIR();
    //     particleSensor.nextSample();
    //     if (++idx >= MODEL_SEQ_LEN) { idx = 0; return true; }
    // }
    // return false;

    // Placeholder: sine wave so the sketch compiles without a sensor attached.
    for (int i = 0; i < MODEL_SEQ_LEN; i++) {
        red[i] = 0.5f + 0.4f * sinf(2.0f * 3.14159f * (float)i / MODEL_SEQ_LEN);
        ir [i] = 0.6f + 0.3f * sinf(2.0f * 3.14159f * (float)i / MODEL_SEQ_LEN + 0.2f);
    }
    return true;
}

// ── setup ─────────────────────────────────────────────────────────────────────
void setup() {
    Serial.begin(115200);
    while (!Serial && millis() < 4000);

    Serial.println();
    Serial.println("SpO2 inference engine");
    Serial.print  ("  MODEL_SEQ_LEN  : "); Serial.println(MODEL_SEQ_LEN);
    Serial.print  ("  HIDDEN_DIM     : "); Serial.println(MODEL_HIDDEN_DIM);
    Serial.print  ("  CHUNK_SIZE     : "); Serial.println(CHUNKED_ATTN_SIZE);

    spo2_weights_init(&g_weights);
    Serial.print  ("  y_mean         : "); Serial.println(g_weights.y_mean, 4);
    Serial.print  ("  y_std          : "); Serial.println(g_weights.y_std,  4);
    Serial.println();

    bool ok = run_self_test();

    if (!ok) {
        Serial.println("HALTED -- fix the model or weight export before deploying.");
        while (true) { delay(1000); }
    }

    Serial.println("Self-test passed. Entering live inference loop.");
    Serial.println();
}

// ── loop ──────────────────────────────────────────────────────────────────────
void loop() {
    if (!read_ppg_window(g_red, g_ir)) {
        delay(1);
        return;
    }

    if (NORMALIZE_INPUT) {
        normalize_window(g_red, MODEL_SEQ_LEN);
        normalize_window(g_ir,  MODEL_SEQ_LEN);
    }

    float skintone = 3.0f;   // TODO: replace with actual measurement

    uint32_t t0 = micros();
    float spo2 = spo2_infer(g_red, g_ir, skintone, &g_weights);
    uint32_t elapsed_us = micros() - t0;

    Serial.print("SpO2: ");
    Serial.print(spo2, 2);
    Serial.print(" %   (");
    Serial.print(elapsed_us / 1000.0f, 1);
    Serial.println(" ms)");

    delay(1000);
}
