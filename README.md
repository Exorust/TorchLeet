<div align="center">

```
████████╗ ██████╗ ██████╗  ██████╗██╗  ██╗██╗     ███████╗███████╗████████╗
╚══██╔══╝██╔═══██╗██╔══██╗██╔════╝██║  ██║██║     ██╔════╝██╔════╝╚══██╔══╝
   ██║   ██║   ██║██████╔╝██║     ███████║██║     █████╗  █████╗     ██║
   ██║   ██║   ██║██╔══██╗██║     ██╔══██║██║     ██╔══╝  ██╔══╝     ██║
   ██║   ╚██████╔╝██║  ██║╚██████╗██║  ██║███████╗███████╗███████╗   ██║
   ╚═╝    ╚═════╝ ╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝╚══════╝╚══════╝╚══════╝   ╚═╝
```

**90 PyTorch problems from real ML/AI interviews at Google, Meta, Anthropic, and more.**

[![GitHub stars](https://img.shields.io/github/stars/Exorust/TorchLeet?style=social)](https://github.com/Exorust/TorchLeet)
[![Website](https://img.shields.io/badge/website-torch--leet.vercel.app-8b5cf6)](https://torch-leet.vercel.app)

[Website](https://torch-leet.vercel.app) | [Follow on Twitter](https://twitter.com/charoori_ai) | [Send Feedback](mailto:chandrahas.aroori@gmail.com?subject=TorchLeet)

</div>

---

## What is TorchLeet?

TorchLeet is a collection of PyTorch practice problems sourced from real engineer interviews. Each problem is a Jupyter notebook with incomplete code blocks (`#TODO`) and a corresponding solution.

Three question sets, 90 problems total:

| Set | Focus | Questions | Difficulty |
|-----|-------|-----------|------------|
| **v1** | Core PyTorch | 35 | Basic to Hard |
| **v2** | LLMs from Scratch | 25 | Easy to Hard |
| **v3** | Advanced ML Systems | 30 | Easy to Expert |

> v3 questions are tagged with the companies that actually ask them, compiled from Glassdoor, Blind, Reddit, and first-person interview reports.

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

## v1: Question Set

### Basic
| # | Problem | Links |
|---|---------|-------|
| 1 | Implement linear regression | [Question](torch/basic/lin-regression/lin-regression.ipynb) / [Solution](torch/basic/lin-regression/lin-regression_SOLN.ipynb) |
| 2 | Write a custom Dataset and Dataloader | [Question](torch/basic/custom-dataset/custom-dataset.ipynb) / [Solution](torch/basic/custom-dataset/custom-dataset_SOLN.ipynb) |
| 3 | Write a custom activation function | [Question](torch/basic/custom-activation/custom-activation.ipynb) / [Solution](torch/basic/custom-activation/custom-activation_SOLN.ipynb) |
| 4 | Implement Custom Loss Function (Huber Loss) | [Question](torch/basic/custom-loss/custom-loss.ipynb) / [Solution](torch/basic/custom-loss/custom-loss_SOLN.ipynb) |
| 5 | Implement a Deep Neural Network | [Question](torch/basic/custom-DNN/custon-DNN.ipynb) / [Solution](torch/basic/custom-DNN/custon-DNN_SOLN.ipynb) |
| 6 | Visualize Training Progress with TensorBoard | [Question](torch/basic/tensorboard/tensorboard.ipynb) / [Solution](torch/basic/tensorboard/tensorboard_SOLN.ipynb) |
| 7 | Save and Load Your PyTorch Model | [Question](torch/basic/save-model/save_model.ipynb) / [Solution](torch/basic/save-model/save_model_SOLN.ipynb) |
| 10 | Implement Softmax function from scratch | |

### Easy
| # | Problem | Links |
|---|---------|-------|
| 1 | Implement a CNN on CIFAR-10 | [Question](torch/easy/cnn/CNN.ipynb) / [Solution](torch/easy/cnn/CNN_SOLN.ipynb) |
| 2 | Implement an RNN from Scratch | [Question](torch/easy/rnn/RNN.ipynb) / [Solution](torch/easy/rnn/RNN_SOLN.ipynb) |
| 3 | Use torchvision.transforms for data augmentation | [Question](torch/easy/augmentation/augmentation.ipynb) / [Solution](torch/easy/augmentation/augmentation_SOLN.ipynb) |
| 4 | Add a benchmark to your PyTorch code | [Question](torch/easy/benchmark/bench.ipynb) / [Solution](torch/easy/benchmark/bench_SOLN.ipynb) |
| 5 | Train an autoencoder for anomaly detection | [Question](torch/easy/autoencoder/autoencoder.ipynb) / [Solution](torch/easy/autoencoder/autoencoder_SOLN.ipynb) |
| 6 | Quantize your language model | [Question](torch/easy/quantize-lm/quantize-language-model.ipynb) / [Solution](torch/easy/quantize-lm/quantize-language-model_SOLN.ipynb) |
| 7 | Implement Mixed Precision Training | [Question](torch/easy/cuda-amp/cuda-amp.ipynb) / [Solution](torch/easy/cuda-amp/cuda-amp_SOLN.ipynb) |

### Medium
| # | Problem | Links |
|---|---------|-------|
| 1 | Implement parameter initialization for a CNN | [Question](torch/medium/cnn-param-init/CNN_ParamInit.ipynb) / [Solution](torch/medium/cnn-param-init/CNN_ParamInit_SOLN.ipynb) |
| 2 | Implement a CNN from Scratch | [Question](torch/medium/cnn-scratch/CNN_scratch.ipynb) / [Solution](torch/medium/cnn-scratch/CNN_scratch_SOLN.ipynb) |
| 3 | Implement an LSTM from Scratch | [Question](torch/medium/lstm/LSTM.ipynb) / [Solution](torch/medium/lstm/LSTM_SOLN.ipynb) |
| 4 | Implement AlexNet from scratch | |
| 5 | Build a Dense Retrieval System | |
| 6 | Implement KNN from scratch in PyTorch | |
| 7 | Train a 3D CNN for segmenting CT images | [Question](torch/medium/3dcnn/3DCNN.ipynb) / [Solution](torch/medium/3dcnn/3DCNN_SOLN.ipynb) |

### Hard
| # | Problem | Links |
|---|---------|-------|
| 1 | Write a custom Autograd function (SILU) | [Question](torch/hard/custom-autograd/custom-autgrad-function.ipynb) / [Solution](torch/hard/custom-autograd/custom-autgrad-function_SOLN.ipynb) |
| 2 | Write a Neural Style Transfer | |
| 3 | Build a Graph Neural Network (GNN) from scratch | |
| 4 | Build a Graph Convolutional Network (GCN) | |
| 5 | Write a Transformer | [Question](torch/hard/transformer/transformer.ipynb) / [Solution](torch/hard/transformer/transformer_SOLN.ipynb) |
| 6 | Write a GAN | [Question](torch/hard/GAN/GAN.ipynb) / [Solution](torch/hard/GAN/GAN_SOLN.ipynb) |
| 7 | Write Sequence-to-Sequence with Attention | [Question](torch/hard/seq-seq/seq-to-seq-with-Attention.ipynb) / [Solution](torch/hard/seq-seq/seq-to-seq-with-Attention_SOLN.ipynb) |
| 8 | Enable distributed training (DDP) | |
| 9 | Work with Sparse Tensors | |
| 10 | Add GradCam/SHAP to explain the model | [Question](torch/hard/xai/xai.ipynb) / [Solution](torch/hard/xai/xai_SOLN.ipynb) |
| 11 | Linear Probe on CLIP Features | |
| 12 | Cross Modal Embedding Visualization (t-SNE/UMAP) | |
| 13 | Implement a Vision Transformer | |
| 14 | Implement a Variational Autoencoder | |

---

## v2: LLM Set

Build a Large Language Model from scratch, one question at a time.

| # | Problem | Links |
|---|---------|-------|
| 1 | Implement KL Divergence Loss | |
| 2 | Implement RMS Norm | |
| 3 | Implement Byte Pair Encoding from Scratch | [Question](llm/Byte-Pair-Encoder/BPE-q3.ipynb) / [Solution](llm/Byte-Pair-Encoder/BPE-q3.ipynb) |
| 4 | Create a RAG Search of Embeddings | |
| 5 | Implement Speculative Decoding | |
| 6 | Implement Attention from Scratch | [Question](llm/Implement-Attention-from-Scratch/attention-q4-Question.ipynb) / [Solution](llm/Implement-Attention-from-Scratch/attention-q4.ipynb) |
| 7 | Implement Multi-Head Attention | [Question](llm/Multi-Head-Attention/multi-head-attention-q5-Question.ipynb) / [Solution](llm/Multi-Head-Attention/multi-head-attention-q5.ipynb) |
| 8 | Implement Grouped Query Attention | [Question](llm/Grouped-Query-Attention/grouped-query-attention-Question.ipynb) / [Solution](llm/Grouped-Query-Attention/grouped-query-attention.ipynb) |
| 9 | Implement KV Cache in Multi-Head Attention | |
| 10 | Implement Sinusoidal Embeddings | [Question](llm/Sinusoidal-Positional-Embedding/sinusoidal-q7-Question.ipynb) / [Solution](llm/Sinusoidal-Positional-Embedding/sinusoidal-q7.ipynb) |
| 11 | Implement ROPE Embeddings | [Question](llm/Rotary-Positional-Embedding/rope-q8-Question.ipynb) / [Solution](llm/Rotary-Positional-Embedding/rope-q8.ipynb) |
| 12 | Implement SmolLM from Scratch | [Question](llm/SmolLM/smollm-q12-Question.ipynb) / [Solution](llm/SmolLM/smollm-q12.ipynb) |
| 13 | Implement Quantization (GPTQ) | |
| 14 | Implement Beam Search for decoding | |
| 15 | Implement Top K Sampling | |
| 16 | Implement Top p Sampling | |
| 17 | Implement Temperature Sampling | |
| 18 | Implement LoRA / QLoRA | |
| 19 | Mix two models into a Mixture of Experts | |
| 20 | Apply SFT on SmolLM | |
| 21 | Apply RLHF on SmolLM | |
| 22 | Implement DPO based RLHF | |
| 23 | Add continuous batching to your LLM | |
| 24 | Chunk Data for Dense Passage Retrieval | |
| 25 | Implement 5D Parallelism for Large Scale Training | |

---

## v3: Advanced ML Systems (NEW)

**30 questions covering what top AI companies actually ask in 2024-2025 interviews.** Tagged with real companies.

### Classical ML from Scratch
| # | Problem | Difficulty | Companies | Links |
|---|---------|------------|-----------|-------|
| 1 | Implement Softmax (numerically stable) | Easy | Apple, Meta, Google, Amazon | [Q](v3/classical-ml/softmax/softmax.ipynb) / [S](v3/classical-ml/softmax/softmax_SOLN.ipynb) |
| 2 | Implement K-Means Clustering | Easy | Uber, LinkedIn, Google, Amazon | [Q](v3/classical-ml/kmeans/kmeans.ipynb) / [S](v3/classical-ml/kmeans/kmeans_SOLN.ipynb) |
| 3 | Implement KNN in PyTorch | Easy | Uber, LinkedIn, Meta | [Q](v3/classical-ml/knn/knn.ipynb) / [S](v3/classical-ml/knn/knn_SOLN.ipynb) |
| 4 | Implement Logistic Regression | Easy | Google, Meta, Amazon | [Q](v3/classical-ml/logistic-regression/logistic-regression.ipynb) / [S](v3/classical-ml/logistic-regression/logistic-regression_SOLN.ipynb) |

### LLM Decoding & Architectures
| # | Problem | Difficulty | Companies | Links |
|---|---------|------------|-----------|-------|
| 5 | Contrastive Loss (InfoNCE) + CLIP | Medium | OpenAI, Anthropic, DeepMind, Midjourney | [Q](v3/modern-architectures/contrastive-loss-clip/contrastive-loss-clip.ipynb) / [S](v3/modern-architectures/contrastive-loss-clip/contrastive-loss-clip_SOLN.ipynb) |
| 6 | 2D Positional Embeddings | Medium | Anthropic, DeepMind, Midjourney, Runway | [Q](v3/modern-architectures/2d-positional-embeddings/2d-positional-embeddings.ipynb) / [S](v3/modern-architectures/2d-positional-embeddings/2d-positional-embeddings_SOLN.ipynb) |
| 7 | Top-p (Nucleus) Sampling | Medium | Anthropic, OpenAI, DeepMind, Perplexity | [Q](v3/llm-decoding/top-p-sampling/top-p-sampling.ipynb) / [S](v3/llm-decoding/top-p-sampling/top-p-sampling_SOLN.ipynb) |
| 8 | Top-k Sampling | Medium | Anthropic, OpenAI, DeepMind, Cohere | [Q](v3/llm-decoding/top-k-sampling/top-k-sampling.ipynb) / [S](v3/llm-decoding/top-k-sampling/top-k-sampling_SOLN.ipynb) |
| 9 | Beam Search for LLM Decoding | Medium | Google, DeepMind, Meta, Apple | [Q](v3/llm-decoding/beam-search/beam-search.ipynb) / [S](v3/llm-decoding/beam-search/beam-search_SOLN.ipynb) |
| 10 | Temperature Sampling | Easy | OpenAI, Anthropic, Cohere, Perplexity | [Q](v3/llm-decoding/temperature-sampling/temperature-sampling.ipynb) / [S](v3/llm-decoding/temperature-sampling/temperature-sampling_SOLN.ipynb) |

### Alignment & Training
| # | Problem | Difficulty | Companies | Links |
|---|---------|------------|-----------|-------|
| 11 | Implement LoRA on a Linear Layer | Medium | Meta, Google, Anthropic, OpenAI, Databricks | [Q](v3/alignment-training/lora/lora.ipynb) / [S](v3/alignment-training/lora/lora_SOLN.ipynb) |
| 14 | Implement DPO Loss from Scratch | Hard | Anthropic, OpenAI, DeepMind, Meta | [Q](v3/alignment-training/dpo/dpo.ipynb) / [S](v3/alignment-training/dpo/dpo_SOLN.ipynb) |
| 15 | Implement PPO for RLHF | Hard | Anthropic, OpenAI, DeepMind, Meta | [Q](v3/alignment-training/ppo-rlhf/ppo-rlhf.ipynb) / [S](v3/alignment-training/ppo-rlhf/ppo-rlhf_SOLN.ipynb) |
| 16 | Implement Gradient Checkpointing | Hard | Meta, Google, NVIDIA, Tesla | [Q](v3/alignment-training/gradient-checkpointing/gradient-checkpointing.ipynb) / [S](v3/alignment-training/gradient-checkpointing/gradient-checkpointing_SOLN.ipynb) |
| 27 | Implement GRPO (DeepSeek-R1) | Expert | DeepMind, Anthropic, OpenAI | [Q](v3/alignment-training/grpo/grpo.ipynb) / [S](v3/alignment-training/grpo/grpo_SOLN.ipynb) |

### LLM Inference & Systems
| # | Problem | Difficulty | Companies | Links |
|---|---------|------------|-----------|-------|
| 12 | Implement KV Cache | Medium | Anthropic, OpenAI, Meta, Perplexity | [Q](v3/llm-inference/kv-cache/kv-cache.ipynb) / [S](v3/llm-inference/kv-cache/kv-cache_SOLN.ipynb) |
| 18 | Implement Speculative Decoding | Hard | Google, DeepMind, Anthropic, Apple | [Q](v3/llm-inference/speculative-decoding/speculative-decoding.ipynb) / [S](v3/llm-inference/speculative-decoding/speculative-decoding_SOLN.ipynb) |
| 19 | Implement Continuous Batching | Hard | Perplexity, Together AI, Anyscale, Meta | [Q](v3/llm-inference/continuous-batching/continuous-batching.ipynb) / [S](v3/llm-inference/continuous-batching/continuous-batching_SOLN.ipynb) |
| 28 | Build a Complete LLM Inference Engine | Expert | Perplexity, Together AI, Anyscale, Fireworks AI | [Q](v3/llm-inference/inference-engine/inference-engine.ipynb) / [S](v3/llm-inference/inference-engine/inference-engine_SOLN.ipynb) |

### Modern Architectures
| # | Problem | Difficulty | Companies | Links |
|---|---------|------------|-----------|-------|
| 13 | Sliding Window Attention | Medium | Mistral, Anthropic, Google, DeepMind | [Q](v3/modern-architectures/sliding-window-attention/sliding-window-attention.ipynb) / [S](v3/modern-architectures/sliding-window-attention/sliding-window-attention_SOLN.ipynb) |
| 17 | Mixture of Experts Layer | Hard | Google, DeepMind, Mistral, Databricks, xAI | [Q](v3/modern-architectures/mixture-of-experts/mixture-of-experts.ipynb) / [S](v3/modern-architectures/mixture-of-experts/mixture-of-experts_SOLN.ipynb) |
| 20 | DDPM (Denoising Diffusion) | Hard | Midjourney, Runway, Stability AI, Adobe, Google | [Q](v3/modern-architectures/ddpm/ddpm.ipynb) / [S](v3/modern-architectures/ddpm/ddpm_SOLN.ipynb) |
| 21 | DDIM Sampling + Classifier-Free Guidance | Hard | Midjourney, Runway, Stability AI, Adobe | [Q](v3/modern-architectures/ddim-cfg/ddim-cfg.ipynb) / [S](v3/modern-architectures/ddim-cfg/ddim-cfg_SOLN.ipynb) |
| 22 | Selective State Space Model (Mamba) | Hard | DeepMind, Google, Anthropic | [Q](v3/modern-architectures/mamba/mamba.ipynb) / [S](v3/modern-architectures/mamba/mamba_SOLN.ipynb) |
| 23 | Vision Transformer + MAE Pretraining | Hard | Meta, Google, Apple, Tesla, Waymo | [Q](v3/modern-architectures/vit-mae/vit-mae.ipynb) / [S](v3/modern-architectures/vit-mae/vit-mae_SOLN.ipynb) |
| 29 | Knowledge Distillation | Medium | Google, Apple, Meta, Qualcomm, Tesla | [Q](v3/modern-architectures/knowledge-distillation/knowledge-distillation.ipynb) / [S](v3/modern-architectures/knowledge-distillation/knowledge-distillation_SOLN.ipynb) |

### GPU Systems & Kernels
| # | Problem | Difficulty | Companies | Links |
|---|---------|------------|-----------|-------|
| 24 | Fused Softmax Kernel in Triton | Expert | NVIDIA, Meta, Google, xAI, Tesla | [Q](v3/gpu-systems/triton-fused-softmax/triton-fused-softmax.ipynb) / [S](v3/gpu-systems/triton-fused-softmax/triton-fused-softmax_SOLN.ipynb) |
| 25 | FlashAttention-2 in Triton | Expert | NVIDIA, Meta, Together AI, xAI | [Q](v3/gpu-systems/flash-attention-triton/flash-attention-triton.ipynb) / [S](v3/gpu-systems/flash-attention-triton/flash-attention-triton_SOLN.ipynb) |
| 26 | FSDP (Fully Sharded Data Parallel) | Expert | Meta, Google, NVIDIA, Anthropic, xAI | [Q](v3/gpu-systems/fsdp/fsdp.ipynb) / [S](v3/gpu-systems/fsdp/fsdp_SOLN.ipynb) |
| 30 | Ring Attention for Long Contexts | Expert | Anthropic, Google, Meta, xAI | [Q](v3/gpu-systems/ring-attention/ring-attention.ipynb) / [S](v3/gpu-systems/ring-attention/ring-attention_SOLN.ipynb) |

---

## Company Quick-Reference

*"If I'm interviewing at X, which v3 questions should I do?"*

| Company | Priority Questions |
|---------|--------------------|
| **Anthropic** | 5, 6, 7, 8, 10, 12, 13, 14, 15, 18, 22, 26, 27, 30 |
| **OpenAI** | 5, 7, 8, 10, 11, 12, 14, 15, 27 |
| **DeepMind** | 5, 6, 7, 8, 9, 13, 14, 15, 17, 18, 22, 27 |
| **Meta** | 1, 2, 3, 4, 9, 11, 12, 14, 15, 16, 19, 23, 24, 25, 26, 30 |
| **Google** | 1, 2, 4, 9, 11, 13, 16, 17, 18, 20, 22, 23, 24, 26, 29, 30 |
| **Apple** | 1, 5, 9, 18, 23, 29 |
| **NVIDIA** | 16, 24, 25, 26 |
| **Midjourney / Runway / Stability AI** | 5, 6, 20, 21 |
| **Perplexity / Together AI / Anyscale** | 7, 10, 12, 19, 25, 28 |
| **Tesla / Waymo** | 16, 23, 24, 29 |
| **xAI** | 17, 24, 25, 26, 30 |
| **Mistral / Cohere** | 7, 8, 10, 13, 17 |

---

## Contributing

Add new questions or improve existing ones. Follow the notebook structure (question file + `_SOLN` file). Submit a PR and tag the authors.

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
