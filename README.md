# Quantization and Edge Deployment of a ML Bearing Fault Detector

Compressing and deploying a PyTorch autoencoder across multiple formats — ONNX, TFLite Float32, and TFLite INT8 — with a live browser-based inference app powered by ONNX Runtime Web.

---

## Overview

This project takes the bearing fault detector from Project 6 and answers the edge deployment question: how small and fast can this model get while keeping 100% fault detection? The model was exported, quantized, and benchmarked across 4 formats, then deployed as a fully client-side web application with no backend required.

---

## Model

Same autoencoder architecture from Project 6, retrained for 100 epochs for improved accuracy.

```
Input (9 features)
→ Encoder: 9 → 16 → 8 → 4
→ Decoder: 4 → 8 → 16 → 9
→ Output (9 reconstructed features)
```

| Training | Value |
|----------|-------|
| Epochs | 100 (vs 50 in Project 6) |
| Final Loss | 0.2989 (vs 0.6079 in Project 6) |
| Optimizer | Adam (lr=0.0001) |
| Loss Function | MSELoss |

---

## Export Pipeline

**PyTorch → ONNX → TFLite**

### ONNX Export

A key lesson in this project: the standard `torch.onnx.export` call failed to load in the browser due to shape inference issues. The fix required explicit opset version, named inputs/outputs, dynamic axes, and forcing weights to be embedded in the file:

```python
torch.onnx.export(
    model,
    dummy_input,
    "autoencoder.onnx",
    verbose=False,
    opset_version=17,
    input_names=["x"],
    output_names=["output"],
    dynamic_axes={
        "x": {0: "batch_size"},
        "output": {0: "batch_size"}
    },
    export_params=True,
    external_data=False
)
```

`export_params=True` and `external_data=False` ensure all trained weights are embedded directly in the `.onnx` file — essential for browser loading where there is no filesystem to reference external weight files.

### ONNX Quantization

ONNX Runtime's `quantize_dynamic` required a preprocessing step before quantization due to shape inference errors:

```python
quant_pre_process(input_model_path="autoencoder.onnx",
                  output_model_path="autoencoder_preprocessed.onnx")

quantize_dynamic(model_input="autoencoder_preprocessed.onnx",
                 model_output="autoencoder_quantized.onnx",
                 weight_type=QuantType.QInt8)
```

### TFLite Conversion

ONNX-to-TFLite direct conversion failed due to library compatibility issues with the current ONNX version. The solution was to rebuild the same architecture in Keras and transfer weights manually from PyTorch, then convert using TFLite's native converter.

---

## Benchmark Results

| Format | Size | Latency | Detection Rate |
|--------|------|---------|----------------|
| ONNX Float32 | 2.82 KB | 0.0155 ms | 100% |
| ONNX INT8 Quantized | 7.16 KB | 0.0344 ms | 100% |
| TFLite Float32 | 5.59 KB | 0.0021 ms | 100% |
| TFLite INT8 Quantized | 6.07 KB | 0.0032 ms | 100% |

**Key Finding: Quantization hurt, not helped.**

On a model this small, quantization overhead — zero points, scale factors, quantization parameters per layer — exceeds the weight savings from float32 → int8 conversion. Both ONNX and TFLite quantized versions ended up larger and slower than their float32 counterparts. Detection rate held at 100% across every format.

This is a genuine engineering insight: quantization benefits large models. For tiny models the overhead is counterproductive.

**Best deployment candidate: TFLite Float32** — smallest practical size at 5.59 KB and fastest inference at 0.0021 ms.

---

## Web Application

A fully client-side browser app (`https://olanle.github.io/Quantization-and-Edge-Deployment-of-a-ML-Bearing-Fault-Detector/app.html`) that runs the ONNX model in-browser using ONNX Runtime Web — no server, no backend, no Python.

**Features:**
- Sensor Sliders mode — 9 individual parameter sliders with hover tooltips explaining each feature and its normal range
- Quick Simulation mode — single slider interpolating from Normal to Fault, all 9 parameters update automatically — designed for non-technical users
- CSV Upload mode — upload a CSV file with one row of 9 values and run inference
- Live results — prediction label, reconstruction error, confidence bar, inference latency
- Run log — last 20 predictions with timestamp, verdict, error and latency
- Dark hardware-themed UI, two-column layout, GitHub Pages ready

**To run locally:**
```bash
python -m http.server 8000
# Open http://localhost:8000/app.html
```

**To deploy on GitHub Pages:**
Push `app.html` and `autoencoder.onnx` to a GitHub repo and enable Pages. The app works with no backend.

---

## Key Learnings

- Quantization increases model size on tiny models — overhead exceeds weight savings
- `export_params=True` and `external_data=False` are required for browser-compatible ONNX export
- Dynamic axes must be specified for ONNX models to handle variable batch sizes in the browser
- ONNX quantization requires a preprocessing step (`quant_pre_process`) to resolve shape inference errors
- Direct ONNX-to-TFLite conversion failed — rebuilding in Keras and transferring weights is the reliable path
- PyTorch tensors must be converted to numpy before use in TFLite's representative dataset generator
- TFLite Float32 outperformed INT8 in both size and speed for this model size

---

## Files

```
project7_quantization/
├── app.html                     — Browser inference app
├── autoencoder.onnx             — Exported model (weights embedded)
├── autoencoder_quantized.onnx   — ONNX INT8 quantized
├── autoencoder.tflite           — TFLite Float32
├── autoencoder_int8.tflite      — TFLite INT8 quantized
└── project7_quantization.ipynb  — Full pipeline notebook
```

---

## Stack
`PyTorch` `ONNX` `ONNX Runtime` `TensorFlow Lite` `ONNX Runtime Web` `JavaScript` `HTML/CSS`

---

*Part of a 10-project ML engineering curriculum targeting Edge/Embedded ML roles.*
