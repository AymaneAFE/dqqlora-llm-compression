# LLM Compression Pipeline

A complete pipeline for compressing Large Language Models using **QLoRA**, **Knowledge Distillation**, and **AWQ Quantization**.

## Overview

This project implements a three-stage compression pipeline that reduces a 7B parameter model to a highly efficient ~0.5GB deployable model:

```
Mistral-7B (14 GB FP16)
        │
        ▼ QLoRA Fine-tuning
Mistral-7B + QLoRA (Teacher)
        │
        ▼ Knowledge Distillation
TinyLlama-1.1B Distilled (2.2 GB FP16)  ← 6-7x compression
        │
        ▼ AWQ Quantization (INT4)
TinyLlama-1.1B AWQ (~0.55 GB)           ← 4x additional compression
```

**Total compression: ~25-28x** with minimal quality degradation.

## Project Structure

```
compression/
├── mistralv01-qlora.ipynb          # Stage 1: QLoRA fine-tuning
├── distillation-tinyllama.ipynb    # Stage 2: Knowledge distillation
├── awq-quantization-llmawq-mithanlab.ipynb  # Stage 3a: AWQ (MIT-HAN-LAB)
├── autoAWQ-llmcompressor.ipynb     # Stage 3b: AWQ (LLM Compressor)
└── notebooks-test/
    ├── overall_llm_testing.ipynb   # Comprehensive LLM evaluation
    └── qlora_distillation_testing.ipynb  # Comparative evaluation
```

## Pipeline Stages

### Stage 1: QLoRA Fine-Tuning

Fine-tune Mistral-7B-v0.1 using QLoRA (Quantized Low-Rank Adaptation):

- **Base Model**: `mistralai/Mistral-7B-v0.1`
- **Quantization**: 4-bit NF4 with double quantization
- **LoRA Config**: r=64, alpha=16, dropout=0.05
- **Target Modules**: Attention (q, k, v, o) + MLP (gate, up, down)
- **Dataset**: Alpaca (70%) + GSM8K (30%)

### Stage 2: Knowledge Distillation

Transfer knowledge from the fine-tuned teacher to a smaller student:

- **Teacher**: Mistral-7B + QLoRA adapters
- **Student**: TinyLlama-1.1B
- **Loss**: KL Divergence with temperature scaling (T=4.0)
- **Dataset**: OpenHermes/Alpaca + GSM8K

### Stage 3: AWQ Quantization

Apply Activation-aware Weight Quantization for efficient deployment:

- **Method**: AWQ (INT4 weights, FP16 activations)
- **Group Size**: 128
- **Calibration**: WikiText-2 (128 samples)
- **Output**: Compatible with vLLM, transformers

## Requirements

```bash
pip install torch transformers accelerate peft bitsandbytes
pip install datasets sentencepiece
pip install llmcompressor  # For LLM Compressor variant
```

## Hardware Requirements

- **Minimum**: NVIDIA GPU with 15GB VRAM (Tesla T4)
- **Recommended**: NVIDIA GPU with 24GB+ VRAM
- **Platform**: Kaggle/Colab compatible

## Quick Start

### 1. QLoRA Fine-Tuning

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

# Load model in 4-bit
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-v0.1",
    quantization_config=bnb_config,
    device_map="auto",
)

# Attach LoRA adapters
lora_config = LoraConfig(r=64, lora_alpha=16, ...)
model = get_peft_model(model, lora_config)
```

### 2. Knowledge Distillation

```python
# Distillation loss
class DistillationLoss(nn.Module):
    def __init__(self, temperature=4.0):
        self.temperature = temperature
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
    
    def forward(self, student_logits, teacher_logits):
        student_log_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=-1)
        return self.kl_div(student_log_probs, teacher_probs) * (self.temperature ** 2)
```

### 3. AWQ Quantization

```python
from llmcompressor import oneshot
from llmcompressor.modifiers.awq import AWQModifier

recipe = [AWQModifier(scheme="W4A16", targets=["Linear"], ignore=["lm_head"])]
oneshot(model=model, dataset=calibration_data, recipe=recipe)
```

## Evaluation Metrics

| Metric | FP16 | AWQ INT4 | Change |
|--------|------|----------|--------|
| Perplexity | ~15 | ~16 | +5-7% |
| Speed (tokens/s) | ~30 | ~80 | +2-3x |
| Model Size | 2.2 GB | 0.55 GB | -75% |

## Notebooks Description

| Notebook | Description |
|----------|-------------|
| `mistralv01-qlora.ipynb` | QLoRA fine-tuning on Alpaca + GSM8K |
| `distillation-tinyllama.ipynb` | Knowledge distillation pipeline |
| `awq-quantization-llmawq-mithanlab.ipynb` | AWQ using MIT-HAN-LAB implementation |
| `autoAWQ-llmcompressor.ipynb` | AWQ using vLLM's LLM Compressor |
| `overall_llm_testing.ipynb` | Comprehensive model testing suite |
| `qlora_distillation_testing.ipynb` | Before/after comparison evaluation |

## References

- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314) - Dettmers et al., 2023
- [AWQ: Activation-aware Weight Quantization](https://arxiv.org/abs/2306.00978) - Lin et al., 2023
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) - Hu et al., 2021
- [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531) - Hinton et al., 2015

## License

MIT License

## Acknowledgments

- [MIT-HAN-LAB/llm-awq](https://github.com/mit-han-lab/llm-awq) for AWQ implementation
- [Hugging Face](https://huggingface.co/) for transformers and PEFT
- [vLLM](https://github.com/vllm-project/vllm) for LLM Compressor
