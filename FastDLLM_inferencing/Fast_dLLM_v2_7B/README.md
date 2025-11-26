---
license: apache-2.0
language:
- en
base_model:
- Qwen/Qwen2.5-7B-Instruct
---

# Fast-dLLM v2 (7B) — Efficient Block-Diffusion LLM

## 📖 Introduction

Autoregressive (AR) large language models (LLMs) have achieved remarkable performance across a wide range of natural language tasks, yet their **inherent sequential decoding limits inference efficiency**.

We present **Fast-dLLM v2** — a carefully designed **block diffusion language model (dLLM)** that efficiently adapts a pretrained AR model (**Qwen2.5-7B-Instruct**) into a diffusion-style decoder for **parallel text generation**.

### ✨ Key Innovations
- **Block Diffusion Mechanism + Complementary Attention Mask**  
  Enables **blockwise bidirectional context modeling** without sacrificing AR objectives.
- **Hierarchical Caching**  
  - **Block-level cache**: Stores historical context representations across blocks.
  - **Sub-block cache**: Parallel decoding within partially generated blocks.
- **Token Shift Mechanism**  
  Retains autoregressive characteristics while supporting bidirectional context within blocks.
- **Parallel Decoding Pipeline**  
  Achieves up to **2.5× speedup** over standard AR decoding **without compromising quality**.

> 🚀 Fast-dLLM v2 uses **only ~1B tokens** for fine-tuning — a **500× reduction** vs. full-attention diffusion LLMs (Dream: 580B tokens) — while **matching or surpassing AR baselines** in accuracy.

![Generation Process](assets/visualization_animation.gif)

---

## 🛠 Model Overview
- **Type**: Block Diffusion Language Model (dLLM)
- **Base Model**: `Qwen/Qwen2.5-7B-Instruct`
- **Architecture**: Transformer w/ RoPE, SwiGLU activation, RMSNorm, Attention QKV bias
- **Params**: ~7B  
- **Layers**: 28  
- **Attention Heads**: 28 (Q), 4 (KV, GQA)  
- **Block Diffusion Size**: 32 tokens  
- **Key Feature**: Parallel **block-wise decoding** + **hierarchical caching (block-level & sub-block)**

---

## 📦 Installation
You will need `transformers`, `torch`, and our **custom generation function**:

```bash
pip install transformers torch numpy
```

---

## 🚀 Quickstart

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Efficient-Large-Model/Fast_dLLM_7B"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

prompt = "Give me a short introduction to large language model."
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompt}
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
inputs = tokenizer([text], return_tensors="pt").to(model.device)

# Fast-dLLM v2 parallel decoding
gen_ids = model.generate(
    inputs["input_ids"],
    tokenizer=tokenizer,
    max_new_tokens=512,
    small_block_size=8,
    threshold=0.9,
)

response = tokenizer.decode(
    gen_ids[0][inputs["input_ids"].shape[1]:], 
    skip_special_tokens=True
)
print(response)
```

---

## 📊 Performance & Benchmarks

### ▶ Real-time Throughput
Fast-dLLM v2 offers **up to 2.54× higher throughput** than Qwen2.5-7B-Instruct, **without loss in quality**.

![Throughput Comparison](assets/throughput.png)

---

### 🏆 Benchmark Results
We compare Fast-dLLM v2 against AR baselines and previous diffusion LLMs on diverse tasks:  
HumanEval, MBPP (code), GSM8K, Math (reasoning), IFEval (instruction), MMLU, GPQA (knowledge QA).

- **1B group**: Fast-dLLM v2 (7B) achieves **best average score: 45.0**.
- **7B group**: Fast-dLLM v2 (7B) achieves **best average score: 60.3**, surpassing LLaDA and Dream models.

![Benchmark Results](assets/benchmark_results.png)

---

## 📜 Citation

If you use Fast-dLLM v2 in your research or products, please cite:

```bibtex
@misc{wu2025fastdllmv2efficientblockdiffusion,
      title={Fast-dLLM v2: Efficient Block-Diffusion LLM}, 
      author={Chengyue Wu and Hao Zhang and Shuchen Xue and Shizhe Diao and Yonggan Fu and Zhijian Liu and Pavlo Molchanov and Ping Luo and Song Han and Enze Xie},
      year={2025},
      eprint={2509.26328},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2509.26328}, 
}
```

---

## 📄 License
Released under **Apache 2.0**, following the base Qwen2.5 license.

---

## 🔗 Resources
- 📄 [Paper](https://arxiv.org/abs/2509.26328)  
- 💻 [Code](https://github.com/NVlabs/Fast-dLLM)  
- 🤗 [HuggingFace Model](https://huggingface.co/Efficient-Large-Model/Fast_dLLM_7B)