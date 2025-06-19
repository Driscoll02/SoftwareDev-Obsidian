# Precision in Large Language Models

> [!Tip]
> For a detailed explanation of how floating point numbers work in computers, I recommend reading the [[Floating Point Numbers]] notes.

## Why Precision Matters in LLMs

Large Language Models (LLMs) contain billions of parameters (weights), each stored as a floating point number. The precision format used for these numbers directly impacts:

1. **Model Size**: Higher precision = larger file size
2. **Memory Usage**: Higher precision = more RAM/VRAM required
3. **Inference Speed**: Lower precision = faster calculations
4. **Model Quality**: Higher precision = potentially better outputs

## Common Precision Formats in LLMs

🟦 float32 (FP32)

- **Bits**: 32
- **Usage**: Standard precision for training and storing original models.
- **Pros**: High precision, stable gradients during training
- **Cons**: Large memory footprint, slower inference
- **Use Case**: Full training pipelines, high-accuracy evaluation

🟪 float16 (FP16)

- **Bits**: 16
- **Usage**: Common for mixed-precision training and faster inference
- **Pros**: Reduces memory and compute cost by ~50% compared to FP32
- **Cons**: Can suffer from underflow/overflow without special handling
- **Use Case**: Training with mixed-precision (FP16 + loss scaling), or inference when GPU supports it

🟨 bfloat16 (Brain Float 16)

- **Bits**: 16
- **Usage**: Developed by Google, often used in TPUs
- **Pros**: Same dynamic range as FP32 with reduced memory
- **Cons**: Slightly less precise mantissa compared to FP16
- **Use Case**: Training on TPUs, and models like PaLM or T5

🟥 int8 (8-bit Integer)

- **Bits**: 8
- **Usage**: Post-training quantisation format for inference
- **Pros**: Massive reduction in model size and inference cost
- **Cons**: Must be carefully quantised (e.g., using calibration or quantisation-aware training) to avoid quality loss
- **Use Case**: Edge devices, low-latency inference, large models on limited hardware

🟩 int4 and int3 (4-bit / 3-bit Integer)

- **Bits**: 4 / 3
- **Usage**: Ultra-low-precision inference (popular with LoRA and quantised GGUF models)
- **Pros**: Even smaller size, enables running LLMs on laptops or mobile devices
- **Cons**: Higher risk of performance degradation — quality depends on quantisation method
- **Use Case**: [[Quantised Models]] like LLaMA using `GGML/GGUF, q4_0, q4_K_M, q3_K_S` etc.

🔷 Mixed Precision (e.g. FP16 + FP32, Int8 + FP16)

- **Usage**: A strategy, not a format — different layers or parts of the model use different precisions
- **Pros**: Balances memory usage and accuracy
- **Cons**: Needs careful implementation to avoid numerical instability
- **Use Case**: Common in both training and inference (e.g., using NVIDIA’s AMP or DeepSpeed)