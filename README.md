<div align="center">

<img src="torchleet-header.png" alt="TorchLeet" width="700">

**75 PyTorch problems from real ML/AI interviews at Google, Meta, Anthropic, and more.**

[![GitHub stars](https://img.shields.io/github/stars/Exorust/TorchLeet?style=social)](https://github.com/Exorust/TorchLeet)
[![Website](https://img.shields.io/badge/website-torch--leet.vercel.app-8b5cf6)](https://torch-leet.vercel.app)

[Follow me on Twitter](https://twitter.com/charoori_ai) | [Try the Terminal](https://torch-leet.vercel.app) | [AI Tutor](https://torch-leet.vercel.app/ai-tutor) | [Send Feedback](mailto:chandrahas.aroori@gmail.com?subject=TorchLeet%20Feedback)

</div>

---

I struggled to grind for ML/AI interviews so I went back to the basics and created a list after careful research. These are real problems from first person reports from real engineer interviews.

> [!IMPORTANT]
> **Don't use GPT.** The whole point is to struggle through these yourself. If you paste these into ChatGPT you're wasting your time. The goal is to deeply understand PyTorch, not to get an answer. I used GPT to help write some of the initial code, but I tested and solved every problem myself. That's where the learning happens.

## AI Tutor (NEW)

Turn any AI assistant into your PyTorch interview coach. The TorchLeet MCP server gives your AI access to all 90 problems, progressive hints, company prep plans, and learning paths, while enforcing a no-spoilers teaching style.

```bash
# Clone the repo first
git clone https://github.com/Exorust/TorchLeet.git
cd TorchLeet

# Then connect the AI Tutor (pick your client)

# Claude Code
claude mcp add torchleet -- npx -y torchleet-mcp

# Codex
codex mcp add torchleet -- npx -y torchleet-mcp
```

<details>
<summary>Claude Desktop / Cursor / VS Code</summary>

Add this to your MCP config:

```json
{
  "mcpServers": {
    "torchleet": {
      "command": "npx",
      "args": ["-y", "torchleet-mcp"]
    }
  }
}
```
</details>

**Four learning guides:**

| Guide | What it does |
|-------|-------------|
| `torchleet-tutor` | Guides you through problems with progressive hints |
| `torchleet-interview-prep` | Timed mock interviews for specific companies |
| `torchleet-review` | Senior ML engineer reviews your code |
| `torchleet-explain` | Deep-dives from intuition to math to code |

[Set up the AI Tutor](https://torch-leet.vercel.app/ai-tutor) | [torchleet-mcp on npm](https://www.npmjs.com/package/torchleet-mcp)

---

75 problems across three tracks:

| Track | Focus | Questions |
|-------|-------|-----------|
| **Basics** | Core PyTorch, classical ML, fundamentals | 24 |
| **LLM Learning Path** | Build an LLM from scratch in order | 23 |
| **Advanced** | Systems, kernels, modern architectures, alignment | 48 |

Questions overlap between tracks. Company-tagged questions tell you exactly what Google, Anthropic, Meta, and others ask.

---

## Quick Start

```bash
# Install PyTorch
# https://pytorch.org/get-started/locally/

# Pick a problem, fill in the TODOs, compare with the solution
jupyter notebook torch/basic/lin-regression/lin-regression.ipynb
```

Each problem has a question file and a `_SOLN` solution file. Fill in the `...` and `#TODO` blocks, then check your work.

---

## LLM Learning Path

Build an LLM from scratch, one question at a time. Recommended order:

### 1. Foundations
| Problem | Links |
|---------|-------|
| Implement Byte Pair Encoding from Scratch | [Q](llm/Byte-Pair-Encoder/BPE-q3.ipynb) |
| Implement Sinusoidal Embeddings | [Q](llm/Sinusoidal-Positional-Embedding/sinusoidal-q7-Question.ipynb) / [S](llm/Sinusoidal-Positional-Embedding/sinusoidal-q7.ipynb) |
| Implement ROPE Embeddings | [Q](llm/Rotary-Positional-Embedding/rope-q8-Question.ipynb) / [S](llm/Rotary-Positional-Embedding/rope-q8.ipynb) |
| Implement RMS Norm | |
| Implement Attention from Scratch | [Q](llm/Implement-Attention-from-Scratch/attention-q4-Question.ipynb) / [S](llm/Implement-Attention-from-Scratch/attention-q4.ipynb) |

### 2. Core Transformer
| Problem | Links |
|---------|-------|
| Implement Multi-Head Attention | [Q](llm/Multi-Head-Attention/multi-head-attention-q5-Question.ipynb) / [S](llm/Multi-Head-Attention/multi-head-attention-q5.ipynb) |
| Implement Grouped Query Attention | [Q](llm/Grouped-Query-Attention/grouped-query-attention-Question.ipynb) / [S](llm/Grouped-Query-Attention/grouped-query-attention.ipynb) |
| Implement KV Cache | [Q](v3/llm-inference/kv-cache/kv-cache.ipynb) / [S](v3/llm-inference/kv-cache/kv-cache_SOLN.ipynb) |
| Implement Sliding Window Attention | [Q](v3/modern-architectures/sliding-window-attention/sliding-window-attention.ipynb) / [S](v3/modern-architectures/sliding-window-attention/sliding-window-attention_SOLN.ipynb) |

### 3. Full Model
| Problem | Links |
|---------|-------|
| Implement SmolLM from Scratch | [Q](llm/SmolLM/smollm-q12-Question.ipynb) / [S](llm/SmolLM/smollm-q12.ipynb) |

### 4. Alignment & Fine-Tuning
| Problem | Companies | Links |
|---------|-----------|-------|
| Implement KL Divergence Loss | | |
| Implement LoRA | Meta, Google, Anthropic, OpenAI | [Q](v3/alignment-training/lora/lora.ipynb) / [S](v3/alignment-training/lora/lora_SOLN.ipynb) |
| Apply SFT on SmolLM | | |
| Implement DPO Loss | Anthropic, OpenAI, DeepMind, Meta | [Q](v3/alignment-training/dpo/dpo.ipynb) / [S](v3/alignment-training/dpo/dpo_SOLN.ipynb) |
| Implement PPO for RLHF | Anthropic, OpenAI, DeepMind, Meta | [Q](v3/alignment-training/ppo-rlhf/ppo-rlhf.ipynb) / [S](v3/alignment-training/ppo-rlhf/ppo-rlhf_SOLN.ipynb) |
| Implement GRPO (DeepSeek-R1) | DeepMind, Anthropic, OpenAI | [Q](v3/alignment-training/grpo/grpo.ipynb) / [S](v3/alignment-training/grpo/grpo_SOLN.ipynb) |

### 5. Decoding & Inference
| Problem | Companies | Links |
|---------|-----------|-------|
| Temperature Sampling | OpenAI, Anthropic, Cohere | [Q](v3/llm-decoding/temperature-sampling/temperature-sampling.ipynb) / [S](v3/llm-decoding/temperature-sampling/temperature-sampling_SOLN.ipynb) |
| Top-k Sampling | Anthropic, OpenAI, DeepMind | [Q](v3/llm-decoding/top-k-sampling/top-k-sampling.ipynb) / [S](v3/llm-decoding/top-k-sampling/top-k-sampling_SOLN.ipynb) |
| Top-p (Nucleus) Sampling | Anthropic, OpenAI, DeepMind | [Q](v3/llm-decoding/top-p-sampling/top-p-sampling.ipynb) / [S](v3/llm-decoding/top-p-sampling/top-p-sampling_SOLN.ipynb) |
| Speculative Decoding | Google, DeepMind, Anthropic | [Q](v3/llm-inference/speculative-decoding/speculative-decoding.ipynb) / [S](v3/llm-inference/speculative-decoding/speculative-decoding_SOLN.ipynb) |
| Continuous Batching | Perplexity, Together AI, Meta | [Q](v3/llm-inference/continuous-batching/continuous-batching.ipynb) / [S](v3/llm-inference/continuous-batching/continuous-batching_SOLN.ipynb) |
| Build a Complete LLM Inference Engine | Perplexity, Together AI, Fireworks AI | [Q](v3/llm-inference/inference-engine/inference-engine.ipynb) / [S](v3/llm-inference/inference-engine/inference-engine_SOLN.ipynb) |

### 6. Systems
| Problem | Companies | Links |
|---------|-----------|-------|
| Mixture of Experts Layer | Google, DeepMind, Mistral, xAI | [Q](v3/modern-architectures/mixture-of-experts/mixture-of-experts.ipynb) / [S](v3/modern-architectures/mixture-of-experts/mixture-of-experts_SOLN.ipynb) |

---

## Basics

Core PyTorch and classical ML fundamentals.

| Problem | Difficulty | Links |
|---------|------------|-------|
| Implement Linear Regression | Basic | [Q](torch/basic/lin-regression/lin-regression.ipynb) / [S](torch/basic/lin-regression/lin-regression_SOLN.ipynb) |
| Custom Dataset and DataLoader | Basic | [Q](torch/basic/custom-dataset/custom-dataset.ipynb) / [S](torch/basic/custom-dataset/custom-dataset_SOLN.ipynb) |
| Custom Activation Function | Basic | [Q](torch/basic/custom-activation/custom-activation.ipynb) / [S](torch/basic/custom-activation/custom-activation_SOLN.ipynb) |
| Custom Loss Function (Huber Loss) | Basic | [Q](torch/basic/custom-loss/custom-loss.ipynb) / [S](torch/basic/custom-loss/custom-loss_SOLN.ipynb) |
| Implement a Deep Neural Network | Basic | [Q](torch/basic/custom-DNN/custon-DNN.ipynb) / [S](torch/basic/custom-DNN/custon-DNN_SOLN.ipynb) |
| Visualize Training with TensorBoard | Basic | [Q](torch/basic/tensorboard/tensorboard.ipynb) / [S](torch/basic/tensorboard/tensorboard_SOLN.ipynb) |
| Save and Load PyTorch Model | Basic | [Q](torch/basic/save-model/save_model.ipynb) / [S](torch/basic/save-model/save_model_SOLN.ipynb) |
| Implement a CNN on CIFAR-10 | Easy | [Q](torch/easy/cnn/CNN.ipynb) / [S](torch/easy/cnn/CNN_SOLN.ipynb) |
| Implement an RNN from Scratch | Easy | [Q](torch/easy/rnn/RNN.ipynb) / [S](torch/easy/rnn/RNN_SOLN.ipynb) |
| Data Augmentation with torchvision | Easy | [Q](torch/easy/augmentation/augmentation.ipynb) / [S](torch/easy/augmentation/augmentation_SOLN.ipynb) |
| Add Benchmarking to PyTorch Code | Easy | [Q](torch/easy/benchmark/bench.ipynb) / [S](torch/easy/benchmark/bench_SOLN.ipynb) |
| Train an Autoencoder for Anomaly Detection | Easy | [Q](torch/easy/autoencoder/autoencoder.ipynb) / [S](torch/easy/autoencoder/autoencoder_SOLN.ipynb) |
| Quantize Your Language Model | Easy | [Q](torch/easy/quantize-lm/quantize-language-model.ipynb) / [S](torch/easy/quantize-lm/quantize-language-model_SOLN.ipynb) |
| Mixed Precision Training | Easy | [Q](torch/easy/cuda-amp/cuda-amp.ipynb) / [S](torch/easy/cuda-amp/cuda-amp_SOLN.ipynb) |
| Implement Softmax (numerically stable) | Easy | [Q](v3/classical-ml/softmax/softmax.ipynb) / [S](v3/classical-ml/softmax/softmax_SOLN.ipynb) |
| Implement K-Means Clustering | Easy | [Q](v3/classical-ml/kmeans/kmeans.ipynb) / [S](v3/classical-ml/kmeans/kmeans_SOLN.ipynb) |
| Implement KNN in PyTorch | Easy | [Q](v3/classical-ml/knn/knn.ipynb) / [S](v3/classical-ml/knn/knn_SOLN.ipynb) |
| Implement Logistic Regression | Easy | [Q](v3/classical-ml/logistic-regression/logistic-regression.ipynb) / [S](v3/classical-ml/logistic-regression/logistic-regression_SOLN.ipynb) |
| KL Divergence Loss | Easy | |
| RMS Norm | Easy | |
| Byte Pair Encoding | Easy | [Q](llm/Byte-Pair-Encoder/BPE-q3.ipynb) |
| CNN Parameter Initialization | Medium | [Q](torch/medium/cnn-param-init/CNN_ParamInit.ipynb) / [S](torch/medium/cnn-param-init/CNN_ParamInit_SOLN.ipynb) |
| Implement a CNN from Scratch | Medium | [Q](torch/medium/cnn-scratch/CNN_scratch.ipynb) / [S](torch/medium/cnn-scratch/CNN_scratch_SOLN.ipynb) |
| Implement an LSTM from Scratch | Medium | [Q](torch/medium/lstm/LSTM.ipynb) / [S](torch/medium/lstm/LSTM_SOLN.ipynb) |

---

## Advanced

Company-tagged questions from real ML/AI interviews. Sorted by topic.

### Modern Architectures
| Problem | Difficulty | Companies | Links |
|---------|------------|-----------|-------|
| Contrastive Loss (InfoNCE) + CLIP | Medium | OpenAI, Anthropic, DeepMind, Midjourney | [Q](v3/modern-architectures/contrastive-loss-clip/contrastive-loss-clip.ipynb) / [S](v3/modern-architectures/contrastive-loss-clip/contrastive-loss-clip_SOLN.ipynb) |
| 2D Positional Embeddings | Medium | Anthropic, DeepMind, Midjourney, Runway | [Q](v3/modern-architectures/2d-positional-embeddings/2d-positional-embeddings.ipynb) / [S](v3/modern-architectures/2d-positional-embeddings/2d-positional-embeddings_SOLN.ipynb) |
| Sliding Window Attention | Medium | Mistral, Anthropic, Google, DeepMind | [Q](v3/modern-architectures/sliding-window-attention/sliding-window-attention.ipynb) / [S](v3/modern-architectures/sliding-window-attention/sliding-window-attention_SOLN.ipynb) |
| Knowledge Distillation | Medium | Google, Apple, Meta, Qualcomm, Tesla | [Q](v3/modern-architectures/knowledge-distillation/knowledge-distillation.ipynb) / [S](v3/modern-architectures/knowledge-distillation/knowledge-distillation_SOLN.ipynb) |
| Mixture of Experts Layer | Hard | Google, DeepMind, Mistral, Databricks, xAI | [Q](v3/modern-architectures/mixture-of-experts/mixture-of-experts.ipynb) / [S](v3/modern-architectures/mixture-of-experts/mixture-of-experts_SOLN.ipynb) |
| DDPM (Denoising Diffusion) | Hard | Midjourney, Runway, Stability AI, Adobe, Google | [Q](v3/modern-architectures/ddpm/ddpm.ipynb) / [S](v3/modern-architectures/ddpm/ddpm_SOLN.ipynb) |
| DDIM Sampling + Classifier-Free Guidance | Hard | Midjourney, Runway, Stability AI, Adobe | [Q](v3/modern-architectures/ddim-cfg/ddim-cfg.ipynb) / [S](v3/modern-architectures/ddim-cfg/ddim-cfg_SOLN.ipynb) |
| Selective State Space Model (Mamba) | Hard | DeepMind, Google, Anthropic | [Q](v3/modern-architectures/mamba/mamba.ipynb) / [S](v3/modern-architectures/mamba/mamba_SOLN.ipynb) |
| Vision Transformer + MAE Pretraining | Hard | Meta, Google, Apple, Tesla, Waymo | [Q](v3/modern-architectures/vit-mae/vit-mae.ipynb) / [S](v3/modern-architectures/vit-mae/vit-mae_SOLN.ipynb) |

### Alignment & Training
| Problem | Difficulty | Companies | Links |
|---------|------------|-----------|-------|
| Implement LoRA | Medium | Meta, Google, Anthropic, OpenAI, Databricks | [Q](v3/alignment-training/lora/lora.ipynb) / [S](v3/alignment-training/lora/lora_SOLN.ipynb) |
| Implement DPO Loss | Hard | Anthropic, OpenAI, DeepMind, Meta | [Q](v3/alignment-training/dpo/dpo.ipynb) / [S](v3/alignment-training/dpo/dpo_SOLN.ipynb) |
| Implement PPO for RLHF | Hard | Anthropic, OpenAI, DeepMind, Meta | [Q](v3/alignment-training/ppo-rlhf/ppo-rlhf.ipynb) / [S](v3/alignment-training/ppo-rlhf/ppo-rlhf_SOLN.ipynb) |
| Gradient Checkpointing | Hard | Meta, Google, NVIDIA, Tesla | [Q](v3/alignment-training/gradient-checkpointing/gradient-checkpointing.ipynb) / [S](v3/alignment-training/gradient-checkpointing/gradient-checkpointing_SOLN.ipynb) |
| Implement GRPO (DeepSeek-R1) | Expert | DeepMind, Anthropic, OpenAI | [Q](v3/alignment-training/grpo/grpo.ipynb) / [S](v3/alignment-training/grpo/grpo_SOLN.ipynb) |
| Apply SFT on SmolLM | Hard | | |

### LLM Inference & Systems
| Problem | Difficulty | Companies | Links |
|---------|------------|-----------|-------|
| Implement KV Cache | Medium | Anthropic, OpenAI, Meta, Perplexity | [Q](v3/llm-inference/kv-cache/kv-cache.ipynb) / [S](v3/llm-inference/kv-cache/kv-cache_SOLN.ipynb) |
| Speculative Decoding | Hard | Google, DeepMind, Anthropic, Apple | [Q](v3/llm-inference/speculative-decoding/speculative-decoding.ipynb) / [S](v3/llm-inference/speculative-decoding/speculative-decoding_SOLN.ipynb) |
| Continuous Batching | Hard | Perplexity, Together AI, Anyscale, Meta | [Q](v3/llm-inference/continuous-batching/continuous-batching.ipynb) / [S](v3/llm-inference/continuous-batching/continuous-batching_SOLN.ipynb) |
| GPTQ Quantization | Hard | | |
| RAG Search of Embeddings | Medium | | |
| Build a Complete LLM Inference Engine | Expert | Perplexity, Together AI, Anyscale, Fireworks AI | [Q](v3/llm-inference/inference-engine/inference-engine.ipynb) / [S](v3/llm-inference/inference-engine/inference-engine_SOLN.ipynb) |

### GPU Systems & Kernels
| Problem | Difficulty | Companies | Links |
|---------|------------|-----------|-------|
| Fused Softmax Kernel in Triton | Expert | NVIDIA, Meta, Google, xAI, Tesla | [Q](v3/gpu-systems/triton-fused-softmax/triton-fused-softmax.ipynb) / [S](v3/gpu-systems/triton-fused-softmax/triton-fused-softmax_SOLN.ipynb) |
| FlashAttention-2 in Triton | Expert | NVIDIA, Meta, Together AI, xAI | [Q](v3/gpu-systems/flash-attention-triton/flash-attention-triton.ipynb) / [S](v3/gpu-systems/flash-attention-triton/flash-attention-triton_SOLN.ipynb) |
| FSDP (Fully Sharded Data Parallel) | Expert | Meta, Google, NVIDIA, Anthropic, xAI | [Q](v3/gpu-systems/fsdp/fsdp.ipynb) / [S](v3/gpu-systems/fsdp/fsdp_SOLN.ipynb) |
| Ring Attention for Long Contexts | Expert | Anthropic, Google, Meta, xAI | [Q](v3/gpu-systems/ring-attention/ring-attention.ipynb) / [S](v3/gpu-systems/ring-attention/ring-attention_SOLN.ipynb) |

### Hard Foundations
| Problem | Difficulty | Links |
|---------|------------|-------|
| Custom Autograd Function (SILU) | Hard | [Q](torch/hard/custom-autograd/custom-autgrad-function.ipynb) / [S](torch/hard/custom-autograd/custom-autgrad-function_SOLN.ipynb) |
| Write a Transformer from Scratch | Hard | [Q](torch/hard/transformer/transformer.ipynb) / [S](torch/hard/transformer/transformer_SOLN.ipynb) |
| Write a GAN | Hard | [Q](torch/hard/GAN/GAN.ipynb) / [S](torch/hard/GAN/GAN_SOLN.ipynb) |
| Sequence-to-Sequence with Attention | Hard | [Q](torch/hard/seq-seq/seq-to-seq-with-Attention.ipynb) / [S](torch/hard/seq-seq/seq-to-seq-with-Attention_SOLN.ipynb) |
| Explainable AI (GradCAM/SHAP) | Hard | [Q](torch/hard/xai/xai.ipynb) / [S](torch/hard/xai/xai_SOLN.ipynb) |

---

## Company Quick-Reference

*"If I'm interviewing at X, which questions should I prioritize?"* Numbers reference the v3-tagged questions above.

| Company | Priority Questions |
|---------|--------------------|
| **Anthropic** | [5](v3/modern-architectures/contrastive-loss-clip/contrastive-loss-clip.ipynb), [6](v3/modern-architectures/2d-positional-embeddings/2d-positional-embeddings.ipynb), [7](v3/llm-decoding/top-p-sampling/top-p-sampling.ipynb), [8](v3/llm-decoding/top-k-sampling/top-k-sampling.ipynb), [10](v3/llm-decoding/temperature-sampling/temperature-sampling.ipynb), [12](v3/llm-inference/kv-cache/kv-cache.ipynb), [13](v3/modern-architectures/sliding-window-attention/sliding-window-attention.ipynb), [14](v3/alignment-training/dpo/dpo.ipynb), [15](v3/alignment-training/ppo-rlhf/ppo-rlhf.ipynb), [18](v3/llm-inference/speculative-decoding/speculative-decoding.ipynb), [22](v3/modern-architectures/mamba/mamba.ipynb), [26](v3/gpu-systems/fsdp/fsdp.ipynb), [27](v3/alignment-training/grpo/grpo.ipynb), [30](v3/gpu-systems/ring-attention/ring-attention.ipynb) |
| **OpenAI** | [5](v3/modern-architectures/contrastive-loss-clip/contrastive-loss-clip.ipynb), [7](v3/llm-decoding/top-p-sampling/top-p-sampling.ipynb), [8](v3/llm-decoding/top-k-sampling/top-k-sampling.ipynb), [10](v3/llm-decoding/temperature-sampling/temperature-sampling.ipynb), [11](v3/alignment-training/lora/lora.ipynb), [12](v3/llm-inference/kv-cache/kv-cache.ipynb), [14](v3/alignment-training/dpo/dpo.ipynb), [15](v3/alignment-training/ppo-rlhf/ppo-rlhf.ipynb), [27](v3/alignment-training/grpo/grpo.ipynb) |
| **DeepMind** | [5](v3/modern-architectures/contrastive-loss-clip/contrastive-loss-clip.ipynb), [6](v3/modern-architectures/2d-positional-embeddings/2d-positional-embeddings.ipynb), [7](v3/llm-decoding/top-p-sampling/top-p-sampling.ipynb), [8](v3/llm-decoding/top-k-sampling/top-k-sampling.ipynb), [9](v3/llm-decoding/beam-search/beam-search.ipynb), [13](v3/modern-architectures/sliding-window-attention/sliding-window-attention.ipynb), [14](v3/alignment-training/dpo/dpo.ipynb), [15](v3/alignment-training/ppo-rlhf/ppo-rlhf.ipynb), [17](v3/modern-architectures/mixture-of-experts/mixture-of-experts.ipynb), [18](v3/llm-inference/speculative-decoding/speculative-decoding.ipynb), [22](v3/modern-architectures/mamba/mamba.ipynb), [27](v3/alignment-training/grpo/grpo.ipynb) |
| **Meta** | [1](v3/classical-ml/softmax/softmax.ipynb), [2](v3/classical-ml/kmeans/kmeans.ipynb), [3](v3/classical-ml/knn/knn.ipynb), [4](v3/classical-ml/logistic-regression/logistic-regression.ipynb), [9](v3/llm-decoding/beam-search/beam-search.ipynb), [11](v3/alignment-training/lora/lora.ipynb), [12](v3/llm-inference/kv-cache/kv-cache.ipynb), [14](v3/alignment-training/dpo/dpo.ipynb), [15](v3/alignment-training/ppo-rlhf/ppo-rlhf.ipynb), [16](v3/alignment-training/gradient-checkpointing/gradient-checkpointing.ipynb), [19](v3/llm-inference/continuous-batching/continuous-batching.ipynb), [23](v3/modern-architectures/vit-mae/vit-mae.ipynb), [24](v3/gpu-systems/triton-fused-softmax/triton-fused-softmax.ipynb), [25](v3/gpu-systems/flash-attention-triton/flash-attention-triton.ipynb), [26](v3/gpu-systems/fsdp/fsdp.ipynb), [30](v3/gpu-systems/ring-attention/ring-attention.ipynb) |
| **Google** | [1](v3/classical-ml/softmax/softmax.ipynb), [2](v3/classical-ml/kmeans/kmeans.ipynb), [4](v3/classical-ml/logistic-regression/logistic-regression.ipynb), [9](v3/llm-decoding/beam-search/beam-search.ipynb), [11](v3/alignment-training/lora/lora.ipynb), [13](v3/modern-architectures/sliding-window-attention/sliding-window-attention.ipynb), [16](v3/alignment-training/gradient-checkpointing/gradient-checkpointing.ipynb), [17](v3/modern-architectures/mixture-of-experts/mixture-of-experts.ipynb), [18](v3/llm-inference/speculative-decoding/speculative-decoding.ipynb), [20](v3/modern-architectures/ddpm/ddpm.ipynb), [22](v3/modern-architectures/mamba/mamba.ipynb), [23](v3/modern-architectures/vit-mae/vit-mae.ipynb), [24](v3/gpu-systems/triton-fused-softmax/triton-fused-softmax.ipynb), [26](v3/gpu-systems/fsdp/fsdp.ipynb), [29](v3/modern-architectures/knowledge-distillation/knowledge-distillation.ipynb), [30](v3/gpu-systems/ring-attention/ring-attention.ipynb) |
| **Apple** | [1](v3/classical-ml/softmax/softmax.ipynb), [5](v3/modern-architectures/contrastive-loss-clip/contrastive-loss-clip.ipynb), [9](v3/llm-decoding/beam-search/beam-search.ipynb), [18](v3/llm-inference/speculative-decoding/speculative-decoding.ipynb), [23](v3/modern-architectures/vit-mae/vit-mae.ipynb), [29](v3/modern-architectures/knowledge-distillation/knowledge-distillation.ipynb) |
| **NVIDIA** | [16](v3/alignment-training/gradient-checkpointing/gradient-checkpointing.ipynb), [24](v3/gpu-systems/triton-fused-softmax/triton-fused-softmax.ipynb), [25](v3/gpu-systems/flash-attention-triton/flash-attention-triton.ipynb), [26](v3/gpu-systems/fsdp/fsdp.ipynb) |
| **Midjourney / Runway / Stability AI** | [5](v3/modern-architectures/contrastive-loss-clip/contrastive-loss-clip.ipynb), [6](v3/modern-architectures/2d-positional-embeddings/2d-positional-embeddings.ipynb), [20](v3/modern-architectures/ddpm/ddpm.ipynb), [21](v3/modern-architectures/ddim-cfg/ddim-cfg.ipynb) |
| **Perplexity / Together AI / Anyscale** | [7](v3/llm-decoding/top-p-sampling/top-p-sampling.ipynb), [10](v3/llm-decoding/temperature-sampling/temperature-sampling.ipynb), [12](v3/llm-inference/kv-cache/kv-cache.ipynb), [19](v3/llm-inference/continuous-batching/continuous-batching.ipynb), [25](v3/gpu-systems/flash-attention-triton/flash-attention-triton.ipynb), [28](v3/llm-inference/inference-engine/inference-engine.ipynb) |
| **Tesla / Waymo** | [16](v3/alignment-training/gradient-checkpointing/gradient-checkpointing.ipynb), [23](v3/modern-architectures/vit-mae/vit-mae.ipynb), [24](v3/gpu-systems/triton-fused-softmax/triton-fused-softmax.ipynb), [29](v3/modern-architectures/knowledge-distillation/knowledge-distillation.ipynb) |
| **xAI** | [17](v3/modern-architectures/mixture-of-experts/mixture-of-experts.ipynb), [24](v3/gpu-systems/triton-fused-softmax/triton-fused-softmax.ipynb), [25](v3/gpu-systems/flash-attention-triton/flash-attention-triton.ipynb), [26](v3/gpu-systems/fsdp/fsdp.ipynb), [30](v3/gpu-systems/ring-attention/ring-attention.ipynb) |
| **Mistral / Cohere** | [7](v3/llm-decoding/top-p-sampling/top-p-sampling.ipynb), [8](v3/llm-decoding/top-k-sampling/top-k-sampling.ipynb), [10](v3/llm-decoding/temperature-sampling/temperature-sampling.ipynb), [13](v3/modern-architectures/sliding-window-attention/sliding-window-attention.ipynb), [17](v3/modern-architectures/mixture-of-experts/mixture-of-experts.ipynb) |

---

## Contributing

Found a bug? Have a question from your own interview? PRs are welcome. Follow the notebook structure (question file + `_SOLN` file) and tag the authors.

If you found this helpful, [follow me on Twitter](https://twitter.com/charoori_ai). I post about ML interviews, PyTorch tips, and what I'm building next. Or just [send me feedback](mailto:chandrahas.aroori@gmail.com?subject=TorchLeet%20Feedback), I read everything.

---

## Authors

<div align="center">
  <table>
    <tr>
      <td align="center">
        <a href="https://github.com/Exorust">
          <img src="https://avatars.githubusercontent.com/u/20578676?v=4" width="100px;" alt="Chandrahas Aroori"/>
          <br />
          <b>Chandrahas Aroori</b>
        </a>
        <br />
        <a href="https://twitter.com/charoori_ai">Twitter</a> |
        <a href="https://www.linkedin.com/in/chandrahas-aroori/">LinkedIn</a> |
        <a href="mailto:chandrahas.aroori@gmail.com">Email</a>
      </td>
      <td align="center">
        <a href="https://github.com/CaslowChien">
          <img src="https://avatars.githubusercontent.com/u/99608452?v=4" width="100px;" alt="Caslow Chien"/>
          <br />
          <b>Caslow Chien</b>
        </a>
        <br />
        <a href="https://caslowchien.github.io/caslow.github.io/">Website</a> |
        <a href="https://www.linkedin.com/in/caslow/">LinkedIn</a> |
        <a href="mailto:caslow@bu.edu">Email</a>
      </td>
    </tr>
  </table>
</div>

## Stargazers

[![Stargazers over time](https://starchart.cc/Exorust/TorchLeet.svg?variant=adaptive)](https://starchart.cc/Exorust/TorchLeet)
