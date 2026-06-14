export type QuestionSet = "v1" | "v2" | "v3";
export type Difficulty = "basic" | "easy" | "medium" | "hard" | "expert";
export type V3Category =
  | "classical-ml"
  | "llm-decoding"
  | "llm-inference"
  | "modern-architectures"
  | "alignment-training"
  | "gpu-systems";

export interface Question {
  id: string;
  set: QuestionSet;
  number: number;
  title: string;
  description: string;
  difficulty: Difficulty;
  category?: string;
  companies: string[];
  questionPath: string | null;
  solutionPath: string | null;
  hasNotebook: boolean;
  // New for revamped structure
  tracks: ("basics" | "advanced" | "llm-path")[];
  llmPathOrder?: number;
  llmPathStage?: string;
}

// ---------------------------------------------------------------------------
// V1 Questions (torch/ directory)
// ---------------------------------------------------------------------------

const v1Questions: Omit<Question, "description" | "tracks" | "llmPathOrder" | "llmPathStage">[] = [
  // Basic (1-7)
  {
    id: "v1-1",
    set: "v1",
    number: 1,
    title: "Implement Linear Regression",
    difficulty: "basic",
    companies: [],
    questionPath: "torch/basic/lin-regression/lin-regression.ipynb",
    solutionPath: "torch/basic/lin-regression/lin-regression_SOLN.ipynb",
    hasNotebook: true,
  },
  {
    id: "v1-2",
    set: "v1",
    number: 2,
    title: "Write a Custom Dataset and DataLoader",
    difficulty: "basic",
    companies: [],
    questionPath: "torch/basic/custom-dataset/custom-dataset.ipynb",
    solutionPath: "torch/basic/custom-dataset/custom-dataset_SOLN.ipynb",
    hasNotebook: true,
  },
  {
    id: "v1-3",
    set: "v1",
    number: 3,
    title: "Write a Custom Activation Function",
    difficulty: "basic",
    companies: [],
    questionPath: "torch/basic/custom-activation/custom-activation.ipynb",
    solutionPath: "torch/basic/custom-activation/custom-activation_SOLN.ipynb",
    hasNotebook: true,
  },
  {
    id: "v1-4",
    set: "v1",
    number: 4,
    title: "Implement Custom Loss Function (Huber Loss)",
    difficulty: "basic",
    companies: [],
    questionPath: "torch/basic/custom-loss/custom-loss.ipynb",
    solutionPath: "torch/basic/custom-loss/custom-loss_SOLN.ipynb",
    hasNotebook: true,
  },
  {
    id: "v1-5",
    set: "v1",
    number: 5,
    title: "Implement a Deep Neural Network",
    difficulty: "basic",
    companies: [],
    questionPath: "torch/basic/custom-DNN/custon-DNN.ipynb",
    solutionPath: "torch/basic/custom-DNN/custon-DNN_SOLN.ipynb",
    hasNotebook: true,
  },
  {
    id: "v1-6",
    set: "v1",
    number: 6,
    title: "Visualize Training with TensorBoard",
    difficulty: "basic",
    companies: [],
    questionPath: "torch/basic/tensorboard/tensorboard.ipynb",
    solutionPath: "torch/basic/tensorboard/tensorboard_SOLN.ipynb",
    hasNotebook: true,
  },
  {
    id: "v1-7",
    set: "v1",
    number: 7,
    title: "Save and Load PyTorch Model",
    difficulty: "basic",
    companies: [],
    questionPath: "torch/basic/save-model/save_model.ipynb",
    solutionPath: "torch/basic/save-model/save_model_SOLN.ipynb",
    hasNotebook: true,
  },

  // Easy (8-14)
  {
    id: "v1-8",
    set: "v1",
    number: 8,
    title: "Implement a CNN on CIFAR-10",
    difficulty: "easy",
    companies: [],
    questionPath: "torch/easy/cnn/CNN.ipynb",
    solutionPath: "torch/easy/cnn/CNN_SOLN.ipynb",
    hasNotebook: true,
  },
  {
    id: "v1-9",
    set: "v1",
    number: 9,
    title: "Implement an RNN from Scratch",
    difficulty: "easy",
    companies: [],
    questionPath: "torch/easy/rnn/RNN.ipynb",
    solutionPath: "torch/easy/rnn/RNN_SOLN.ipynb",
    hasNotebook: true,
  },
  {
    id: "v1-10",
    set: "v1",
    number: 10,
    title: "Data Augmentation with torchvision.transforms",
    difficulty: "easy",
    companies: [],
    questionPath: "torch/easy/augmentation/augmentation.ipynb",
    solutionPath: "torch/easy/augmentation/augmentation_SOLN.ipynb",
    hasNotebook: true,
  },
  {
    id: "v1-11",
    set: "v1",
    number: 11,
    title: "Add Benchmarking to PyTorch Code",
    difficulty: "easy",
    companies: [],
    questionPath: "torch/easy/benchmark/bench.ipynb",
    solutionPath: "torch/easy/benchmark/bench_SOLN.ipynb",
    hasNotebook: true,
  },
  {
    id: "v1-12",
    set: "v1",
    number: 12,
    title: "Train an Autoencoder for Anomaly Detection",
    difficulty: "easy",
    companies: [],
    questionPath: "torch/easy/autoencoder/autoencoder.ipynb",
    solutionPath: "torch/easy/autoencoder/autoencoder_SOLN.ipynb",
    hasNotebook: true,
  },
  {
    id: "v1-13",
    set: "v1",
    number: 13,
    title: "Quantize Your Language Model",
    difficulty: "easy",
    companies: [],
    questionPath: "torch/easy/quantize-lm/quantize-language-model.ipynb",
    solutionPath: "torch/easy/quantize-lm/quantize-language-model_SOLN.ipynb",
    hasNotebook: true,
  },
  {
    id: "v1-14",
    set: "v1",
    number: 14,
    title: "Mixed Precision Training with torch.cuda.amp",
    difficulty: "easy",
    companies: [],
    questionPath: "torch/easy/cuda-amp/cuda-amp.ipynb",
    solutionPath: "torch/easy/cuda-amp/cuda-amp_SOLN.ipynb",
    hasNotebook: true,
  },

  // Medium (15-21)
  {
    id: "v1-15",
    set: "v1",
    number: 15,
    title: "CNN Parameter Initialization",
    difficulty: "medium",
    companies: [],
    questionPath: "torch/medium/cnn-param-init/CNN_ParamInit.ipynb",
    solutionPath: "torch/medium/cnn-param-init/CNN_ParamInit_SOLN.ipynb",
    hasNotebook: true,
  },
  {
    id: "v1-16",
    set: "v1",
    number: 16,
    title: "Implement a CNN from Scratch",
    difficulty: "medium",
    companies: [],
    questionPath: "torch/medium/cnn-scratch/CNN_scratch.ipynb",
    solutionPath: "torch/medium/cnn-scratch/CNN_scratch_SOLN.ipynb",
    hasNotebook: true,
  },
  {
    id: "v1-17",
    set: "v1",
    number: 17,
    title: "Implement an LSTM from Scratch",
    difficulty: "medium",
    companies: [],
    questionPath: "torch/medium/lstm/LSTM.ipynb",
    solutionPath: "torch/medium/lstm/LSTM_SOLN.ipynb",
    hasNotebook: true,
  },
  {
    id: "v1-18",
    set: "v1",
    number: 18,
    title: "Implement AlexNet from Scratch",
    difficulty: "medium",
    companies: [],
    questionPath: null,
    solutionPath: null,
    hasNotebook: false,
  },
  {
    id: "v1-19",
    set: "v1",
    number: 19,
    title: "Build a Dense Retrieval System",
    difficulty: "medium",
    companies: [],
    questionPath: null,
    solutionPath: null,
    hasNotebook: false,
  },
  {
    id: "v1-20",
    set: "v1",
    number: 20,
    title: "Implement KNN from Scratch in PyTorch",
    difficulty: "medium",
    companies: [],
    questionPath: null,
    solutionPath: null,
    hasNotebook: false,
  },
  {
    id: "v1-21",
    set: "v1",
    number: 21,
    title: "Train a 3D CNN for CT Image Segmentation",
    difficulty: "medium",
    companies: [],
    questionPath: "torch/medium/3dcnn/3DCNN.ipynb",
    solutionPath: "torch/medium/3dcnn/3DCNN_SOLN.ipynb",
    hasNotebook: true,
  },

  // Hard (22-35)
  {
    id: "v1-22",
    set: "v1",
    number: 22,
    title: "Write a Custom Autograd Function (SILU)",
    difficulty: "hard",
    companies: [],
    questionPath: "torch/hard/custom-autograd/custom-autgrad-function.ipynb",
    solutionPath:
      "torch/hard/custom-autograd/custom-autgrad-function_SOLN.ipynb",
    hasNotebook: true,
  },
  {
    id: "v1-23",
    set: "v1",
    number: 23,
    title: "Neural Style Transfer",
    difficulty: "hard",
    companies: [],
    questionPath: null,
    solutionPath: null,
    hasNotebook: false,
  },
  {
    id: "v1-24",
    set: "v1",
    number: 24,
    title: "Build a Graph Neural Network from Scratch",
    difficulty: "hard",
    companies: [],
    questionPath: null,
    solutionPath: null,
    hasNotebook: false,
  },
  {
    id: "v1-25",
    set: "v1",
    number: 25,
    title: "Build a Graph Convolutional Network from Scratch",
    difficulty: "hard",
    companies: [],
    questionPath: null,
    solutionPath: null,
    hasNotebook: false,
  },
  {
    id: "v1-26",
    set: "v1",
    number: 26,
    title: "Write a Transformer from Scratch",
    difficulty: "hard",
    companies: [],
    questionPath: "torch/hard/transformer/transformer.ipynb",
    solutionPath: "torch/hard/transformer/transformer_SOLN.ipynb",
    hasNotebook: true,
  },
  {
    id: "v1-27",
    set: "v1",
    number: 27,
    title: "Write a GAN",
    difficulty: "hard",
    companies: [],
    questionPath: "torch/hard/GAN/GAN.ipynb",
    solutionPath: "torch/hard/GAN/GAN_SOLN.ipynb",
    hasNotebook: true,
  },
  {
    id: "v1-28",
    set: "v1",
    number: 28,
    title: "Sequence-to-Sequence with Attention",
    difficulty: "hard",
    companies: [],
    questionPath: "torch/hard/seq-seq/seq-to-seq-with-Attention.ipynb",
    solutionPath: "torch/hard/seq-seq/seq-to-seq-with-Attention_SOLN.ipynb",
    hasNotebook: true,
  },
  {
    id: "v1-29",
    set: "v1",
    number: 29,
    title: "Distributed Training (DistributedDataParallel)",
    difficulty: "hard",
    companies: [],
    questionPath: null,
    solutionPath: null,
    hasNotebook: false,
  },
  {
    id: "v1-30",
    set: "v1",
    number: 30,
    title: "Work with Sparse Tensors",
    difficulty: "hard",
    companies: [],
    questionPath: null,
    solutionPath: null,
    hasNotebook: false,
  },
  {
    id: "v1-31",
    set: "v1",
    number: 31,
    title: "Explainable AI (GradCAM/SHAP)",
    difficulty: "hard",
    companies: [],
    questionPath: "torch/hard/xai/xai.ipynb",
    solutionPath: "torch/hard/xai/xai_SOLN.ipynb",
    hasNotebook: true,
  },
  {
    id: "v1-32",
    set: "v1",
    number: 32,
    title: "Linear Probe on CLIP Features",
    difficulty: "hard",
    companies: [],
    questionPath: null,
    solutionPath: null,
    hasNotebook: false,
  },
  {
    id: "v1-33",
    set: "v1",
    number: 33,
    title: "Cross Modal Embedding Visualization (t-SNE/UMAP)",
    difficulty: "hard",
    companies: [],
    questionPath: null,
    solutionPath: null,
    hasNotebook: false,
  },
  {
    id: "v1-34",
    set: "v1",
    number: 34,
    title: "Implement a Vision Transformer",
    difficulty: "hard",
    companies: [],
    questionPath: null,
    solutionPath: null,
    hasNotebook: false,
  },
  {
    id: "v1-35",
    set: "v1",
    number: 35,
    title: "Implement a Variational Autoencoder",
    difficulty: "hard",
    companies: [],
    questionPath: null,
    solutionPath: null,
    hasNotebook: false,
  },
];

// ---------------------------------------------------------------------------
// V2 Questions (llm/ directory)
// ---------------------------------------------------------------------------

const v2Questions: Omit<Question, "description" | "tracks" | "llmPathOrder" | "llmPathStage">[] = [
  {
    id: "v2-1",
    set: "v2",
    number: 1,
    title: "Implement KL Divergence Loss",
    difficulty: "easy",
    companies: [],
    questionPath: null,
    solutionPath: null,
    hasNotebook: false,
  },
  {
    id: "v2-2",
    set: "v2",
    number: 2,
    title: "Implement RMS Norm",
    difficulty: "easy",
    companies: [],
    questionPath: null,
    solutionPath: null,
    hasNotebook: false,
  },
  {
    id: "v2-3",
    set: "v2",
    number: 3,
    title: "Implement Byte Pair Encoding from Scratch",
    difficulty: "easy",
    companies: [],
    questionPath: "llm/Byte-Pair-Encoder/BPE-q3.ipynb",
    solutionPath: "llm/Byte-Pair-Encoder/BPE-q3.ipynb",
    hasNotebook: true,
  },
  {
    id: "v2-4",
    set: "v2",
    number: 4,
    title: "Create a RAG Search of Embeddings",
    difficulty: "medium",
    companies: [],
    questionPath: null,
    solutionPath: null,
    hasNotebook: false,
  },
  {
    id: "v2-5",
    set: "v2",
    number: 5,
    title: "Implement Speculative Decoding with Predictive Prefill",
    difficulty: "hard",
    companies: [],
    questionPath: null,
    solutionPath: null,
    hasNotebook: false,
  },
  {
    id: "v2-6",
    set: "v2",
    number: 6,
    title: "Implement Attention from Scratch",
    difficulty: "medium",
    companies: [],
    questionPath:
      "llm/Implement-Attention-from-Scratch/attention-q4-Question.ipynb",
    solutionPath: "llm/Implement-Attention-from-Scratch/attention-q4.ipynb",
    hasNotebook: true,
  },
  {
    id: "v2-7",
    set: "v2",
    number: 7,
    title: "Implement Multi-Head Attention from Scratch",
    difficulty: "medium",
    companies: [],
    questionPath:
      "llm/Multi-Head-Attention/multi-head-attention-q5-Question.ipynb",
    solutionPath: "llm/Multi-Head-Attention/multi-head-attention-q5.ipynb",
    hasNotebook: true,
  },
  {
    id: "v2-8",
    set: "v2",
    number: 8,
    title: "Implement Grouped Query Attention from Scratch",
    difficulty: "medium",
    companies: [],
    questionPath:
      "llm/Grouped-Query-Attention/grouped-query-attention-Question.ipynb",
    solutionPath: "llm/Grouped-Query-Attention/grouped-query-attention.ipynb",
    hasNotebook: true,
  },
  {
    id: "v2-9",
    set: "v2",
    number: 9,
    title: "Implement KV Cache in Multi-Head Attention",
    difficulty: "medium",
    companies: [],
    questionPath: null,
    solutionPath: null,
    hasNotebook: false,
  },
  {
    id: "v2-10",
    set: "v2",
    number: 10,
    title: "Implement Sinusoidal Embeddings",
    difficulty: "medium",
    companies: [],
    questionPath:
      "llm/Sinusoidal-Positional-Embedding/sinusoidal-q7-Question.ipynb",
    solutionPath: "llm/Sinusoidal-Positional-Embedding/sinusoidal-q7.ipynb",
    hasNotebook: true,
  },
  {
    id: "v2-11",
    set: "v2",
    number: 11,
    title: "Implement ROPE Embeddings",
    difficulty: "medium",
    companies: [],
    questionPath:
      "llm/Rotary-Positional-Embedding/rope-q8-Question.ipynb",
    solutionPath: "llm/Rotary-Positional-Embedding/rope-q8.ipynb",
    hasNotebook: true,
  },
  {
    id: "v2-12",
    set: "v2",
    number: 12,
    title: "Implement SmolLM from Scratch",
    difficulty: "hard",
    companies: [],
    questionPath: "llm/SmolLM/smollm-q12-Question.ipynb",
    solutionPath: "llm/SmolLM/smollm-q12.ipynb",
    hasNotebook: true,
  },
  {
    id: "v2-13",
    set: "v2",
    number: 13,
    title: "Implement Quantization (GPTQ)",
    difficulty: "hard",
    companies: [],
    questionPath: null,
    solutionPath: null,
    hasNotebook: false,
  },
  {
    id: "v2-14",
    set: "v2",
    number: 14,
    title: "Implement Beam Search for LLM Decoding",
    difficulty: "medium",
    companies: [],
    questionPath: null,
    solutionPath: null,
    hasNotebook: false,
  },
  {
    id: "v2-15",
    set: "v2",
    number: 15,
    title: "Implement Top-K Sampling for LLM Decoding",
    difficulty: "medium",
    companies: [],
    questionPath: null,
    solutionPath: null,
    hasNotebook: false,
  },
  {
    id: "v2-16",
    set: "v2",
    number: 16,
    title: "Implement Top-p Sampling for LLM Decoding",
    difficulty: "medium",
    companies: [],
    questionPath: null,
    solutionPath: null,
    hasNotebook: false,
  },
  {
    id: "v2-17",
    set: "v2",
    number: 17,
    title: "Implement Temperature Sampling for LLM Decoding",
    difficulty: "easy",
    companies: [],
    questionPath: null,
    solutionPath: null,
    hasNotebook: false,
  },
  {
    id: "v2-18",
    set: "v2",
    number: 18,
    title: "Implement LoRA on an LLM Layer",
    difficulty: "medium",
    companies: [],
    questionPath: null,
    solutionPath: null,
    hasNotebook: false,
  },
  {
    id: "v2-19",
    set: "v2",
    number: 19,
    title: "Mix Models to Create Mixture of Experts",
    difficulty: "hard",
    companies: [],
    questionPath: null,
    solutionPath: null,
    hasNotebook: false,
  },
  {
    id: "v2-20",
    set: "v2",
    number: 20,
    title: "Apply SFT on SmolLM",
    difficulty: "hard",
    companies: [],
    questionPath: null,
    solutionPath: null,
    hasNotebook: false,
  },
  {
    id: "v2-21",
    set: "v2",
    number: 21,
    title: "Apply RLHF on SmolLM",
    difficulty: "hard",
    companies: [],
    questionPath: null,
    solutionPath: null,
    hasNotebook: false,
  },
  {
    id: "v2-22",
    set: "v2",
    number: 22,
    title: "Implement DPO based RLHF",
    difficulty: "hard",
    companies: [],
    questionPath: null,
    solutionPath: null,
    hasNotebook: false,
  },
  {
    id: "v2-23",
    set: "v2",
    number: 23,
    title: "Add Continuous Batching to LLM",
    difficulty: "hard",
    companies: [],
    questionPath: null,
    solutionPath: null,
    hasNotebook: false,
  },
  {
    id: "v2-24",
    set: "v2",
    number: 24,
    title: "Chunk Textual Data for Dense Passage Retrieval",
    difficulty: "medium",
    companies: [],
    questionPath: null,
    solutionPath: null,
    hasNotebook: false,
  },
  {
    id: "v2-25",
    set: "v2",
    number: 25,
    title: "Implement Large Scale Training (5D Parallelism)",
    difficulty: "expert",
    companies: [],
    questionPath: null,
    solutionPath: null,
    hasNotebook: false,
  },
];

// ---------------------------------------------------------------------------
// V3 Questions (v3/ directory)
// ---------------------------------------------------------------------------

const v3Questions: Omit<Question, "description" | "tracks" | "llmPathOrder" | "llmPathStage">[] = [
  // classical-ml (1-4)
  {
    id: "v3-1",
    set: "v3",
    number: 1,
    title: "Implement Softmax from Scratch",
    difficulty: "easy",
    category: "classical-ml",
    companies: ["Apple", "Meta", "Google", "Amazon"],
    questionPath: "v3/classical-ml/softmax/softmax.ipynb",
    solutionPath: "v3/classical-ml/softmax/softmax_SOLN.ipynb",
    hasNotebook: true,
  },
  {
    id: "v3-2",
    set: "v3",
    number: 2,
    title: "Implement K-Means Clustering in PyTorch",
    difficulty: "easy",
    category: "classical-ml",
    companies: ["Uber", "LinkedIn", "Google", "Amazon"],
    questionPath: "v3/classical-ml/kmeans/kmeans.ipynb",
    solutionPath: "v3/classical-ml/kmeans/kmeans_SOLN.ipynb",
    hasNotebook: true,
  },
  {
    id: "v3-3",
    set: "v3",
    number: 3,
    title: "Implement KNN in PyTorch",
    difficulty: "easy",
    category: "classical-ml",
    companies: ["Uber", "LinkedIn", "Meta"],
    questionPath: "v3/classical-ml/knn/knn.ipynb",
    solutionPath: "v3/classical-ml/knn/knn_SOLN.ipynb",
    hasNotebook: true,
  },
  {
    id: "v3-4",
    set: "v3",
    number: 4,
    title: "Implement Logistic Regression with Gradient Descent",
    difficulty: "easy",
    category: "classical-ml",
    companies: ["Google", "Meta", "Amazon"],
    questionPath: "v3/classical-ml/logistic-regression/logistic-regression.ipynb",
    solutionPath:
      "v3/classical-ml/logistic-regression/logistic-regression_SOLN.ipynb",
    hasNotebook: true,
  },

  // modern-architectures (5-6)
  {
    id: "v3-5",
    set: "v3",
    number: 5,
    title: "Implement Contrastive Loss (InfoNCE) + CLIP Training Loop",
    difficulty: "medium",
    category: "modern-architectures",
    companies: ["OpenAI", "Anthropic", "DeepMind", "Midjourney", "Apple"],
    questionPath:
      "v3/modern-architectures/contrastive-loss-clip/contrastive-loss-clip.ipynb",
    solutionPath:
      "v3/modern-architectures/contrastive-loss-clip/contrastive-loss-clip_SOLN.ipynb",
    hasNotebook: true,
  },
  {
    id: "v3-6",
    set: "v3",
    number: 6,
    title: "Implement 2D Positional Embeddings",
    difficulty: "medium",
    category: "modern-architectures",
    companies: ["Anthropic", "DeepMind", "Midjourney", "Runway"],
    questionPath:
      "v3/modern-architectures/2d-positional-embeddings/2d-positional-embeddings.ipynb",
    solutionPath:
      "v3/modern-architectures/2d-positional-embeddings/2d-positional-embeddings_SOLN.ipynb",
    hasNotebook: true,
  },

  // llm-decoding (7-10)
  {
    id: "v3-7",
    set: "v3",
    number: 7,
    title: "Implement Top-p (Nucleus) Sampling",
    difficulty: "medium",
    category: "llm-decoding",
    companies: ["Anthropic", "OpenAI", "DeepMind", "Perplexity", "Cohere"],
    questionPath: "v3/llm-decoding/top-p-sampling/top-p-sampling.ipynb",
    solutionPath: "v3/llm-decoding/top-p-sampling/top-p-sampling_SOLN.ipynb",
    hasNotebook: true,
  },
  {
    id: "v3-8",
    set: "v3",
    number: 8,
    title: "Implement Top-k Sampling",
    difficulty: "medium",
    category: "llm-decoding",
    companies: ["Anthropic", "OpenAI", "DeepMind", "Cohere"],
    questionPath: "v3/llm-decoding/top-k-sampling/top-k-sampling.ipynb",
    solutionPath: "v3/llm-decoding/top-k-sampling/top-k-sampling_SOLN.ipynb",
    hasNotebook: true,
  },
  {
    id: "v3-9",
    set: "v3",
    number: 9,
    title: "Implement Beam Search for LLM Decoding",
    difficulty: "medium",
    category: "llm-decoding",
    companies: ["Google", "DeepMind", "Meta", "Apple"],
    questionPath: "v3/llm-decoding/beam-search/beam-search.ipynb",
    solutionPath: "v3/llm-decoding/beam-search/beam-search_SOLN.ipynb",
    hasNotebook: true,
  },
  {
    id: "v3-10",
    set: "v3",
    number: 10,
    title: "Implement Temperature Sampling",
    difficulty: "easy",
    category: "llm-decoding",
    companies: ["OpenAI", "Anthropic", "Cohere", "Perplexity"],
    questionPath:
      "v3/llm-decoding/temperature-sampling/temperature-sampling.ipynb",
    solutionPath:
      "v3/llm-decoding/temperature-sampling/temperature-sampling_SOLN.ipynb",
    hasNotebook: true,
  },

  // alignment-training (11)
  {
    id: "v3-11",
    set: "v3",
    number: 11,
    title: "Implement LoRA on a Linear Layer",
    difficulty: "medium",
    category: "alignment-training",
    companies: ["Meta", "Google", "Anthropic", "OpenAI", "Databricks"],
    questionPath: "v3/alignment-training/lora/lora.ipynb",
    solutionPath: "v3/alignment-training/lora/lora_SOLN.ipynb",
    hasNotebook: true,
  },

  // llm-inference (12)
  {
    id: "v3-12",
    set: "v3",
    number: 12,
    title: "Implement KV Cache for Autoregressive Generation",
    difficulty: "medium",
    category: "llm-inference",
    companies: ["Anthropic", "OpenAI", "Meta", "Perplexity", "Together AI"],
    questionPath: "v3/llm-inference/kv-cache/kv-cache.ipynb",
    solutionPath: "v3/llm-inference/kv-cache/kv-cache_SOLN.ipynb",
    hasNotebook: true,
  },

  // modern-architectures (13)
  {
    id: "v3-13",
    set: "v3",
    number: 13,
    title: "Implement Sliding Window Attention",
    difficulty: "medium",
    category: "modern-architectures",
    companies: ["Mistral", "Anthropic", "Google", "DeepMind"],
    questionPath:
      "v3/modern-architectures/sliding-window-attention/sliding-window-attention.ipynb",
    solutionPath:
      "v3/modern-architectures/sliding-window-attention/sliding-window-attention_SOLN.ipynb",
    hasNotebook: true,
  },

  // alignment-training (14-16)
  {
    id: "v3-14",
    set: "v3",
    number: 14,
    title: "Implement DPO Loss from Scratch",
    difficulty: "hard",
    category: "alignment-training",
    companies: ["Anthropic", "OpenAI", "DeepMind", "Meta"],
    questionPath: "v3/alignment-training/dpo/dpo.ipynb",
    solutionPath: "v3/alignment-training/dpo/dpo_SOLN.ipynb",
    hasNotebook: true,
  },
  {
    id: "v3-15",
    set: "v3",
    number: 15,
    title: "Implement PPO for RLHF",
    difficulty: "hard",
    category: "alignment-training",
    companies: ["Anthropic", "OpenAI", "DeepMind", "Meta"],
    questionPath: "v3/alignment-training/ppo-rlhf/ppo-rlhf.ipynb",
    solutionPath: "v3/alignment-training/ppo-rlhf/ppo-rlhf_SOLN.ipynb",
    hasNotebook: true,
  },
  {
    id: "v3-16",
    set: "v3",
    number: 16,
    title: "Implement Gradient Checkpointing",
    difficulty: "hard",
    category: "alignment-training",
    companies: ["Meta", "Google", "NVIDIA", "Tesla"],
    questionPath:
      "v3/alignment-training/gradient-checkpointing/gradient-checkpointing.ipynb",
    solutionPath:
      "v3/alignment-training/gradient-checkpointing/gradient-checkpointing_SOLN.ipynb",
    hasNotebook: true,
  },

  // modern-architectures (17, 20-23, 29)
  {
    id: "v3-17",
    set: "v3",
    number: 17,
    title: "Implement Mixture of Experts Layer",
    difficulty: "hard",
    category: "modern-architectures",
    companies: ["Google", "DeepMind", "Mistral", "Databricks", "xAI"],
    questionPath:
      "v3/modern-architectures/mixture-of-experts/mixture-of-experts.ipynb",
    solutionPath:
      "v3/modern-architectures/mixture-of-experts/mixture-of-experts_SOLN.ipynb",
    hasNotebook: true,
  },

  // llm-inference (18-19)
  {
    id: "v3-18",
    set: "v3",
    number: 18,
    title: "Implement Speculative Decoding",
    difficulty: "hard",
    category: "llm-inference",
    companies: ["Google", "DeepMind", "Anthropic", "Apple"],
    questionPath:
      "v3/llm-inference/speculative-decoding/speculative-decoding.ipynb",
    solutionPath:
      "v3/llm-inference/speculative-decoding/speculative-decoding_SOLN.ipynb",
    hasNotebook: true,
  },
  {
    id: "v3-19",
    set: "v3",
    number: 19,
    title: "Implement Continuous Batching for LLM Inference",
    difficulty: "hard",
    category: "llm-inference",
    companies: ["Perplexity", "Together AI", "Anyscale", "Meta"],
    questionPath:
      "v3/llm-inference/continuous-batching/continuous-batching.ipynb",
    solutionPath:
      "v3/llm-inference/continuous-batching/continuous-batching_SOLN.ipynb",
    hasNotebook: true,
  },

  // modern-architectures (20-23)
  {
    id: "v3-20",
    set: "v3",
    number: 20,
    title: "Implement DDPM from Scratch",
    difficulty: "hard",
    category: "modern-architectures",
    companies: ["Midjourney", "Runway", "Stability AI", "Adobe", "Google"],
    questionPath: "v3/modern-architectures/ddpm/ddpm.ipynb",
    solutionPath: "v3/modern-architectures/ddpm/ddpm_SOLN.ipynb",
    hasNotebook: true,
  },
  {
    id: "v3-21",
    set: "v3",
    number: 21,
    title: "Implement DDIM Sampling + Classifier-Free Guidance",
    difficulty: "hard",
    category: "modern-architectures",
    companies: ["Midjourney", "Runway", "Stability AI", "Adobe"],
    questionPath: "v3/modern-architectures/ddim-cfg/ddim-cfg.ipynb",
    solutionPath: "v3/modern-architectures/ddim-cfg/ddim-cfg_SOLN.ipynb",
    hasNotebook: true,
  },
  {
    id: "v3-22",
    set: "v3",
    number: 22,
    title: "Implement Selective State Space Model (Mamba)",
    difficulty: "hard",
    category: "modern-architectures",
    companies: ["DeepMind", "Google", "Anthropic"],
    questionPath: "v3/modern-architectures/mamba/mamba.ipynb",
    solutionPath: "v3/modern-architectures/mamba/mamba_SOLN.ipynb",
    hasNotebook: true,
  },
  {
    id: "v3-23",
    set: "v3",
    number: 23,
    title: "Implement Vision Transformer + MAE Pretraining",
    difficulty: "hard",
    category: "modern-architectures",
    companies: ["Meta", "Google", "Apple", "Tesla", "Waymo"],
    questionPath: "v3/modern-architectures/vit-mae/vit-mae.ipynb",
    solutionPath: "v3/modern-architectures/vit-mae/vit-mae_SOLN.ipynb",
    hasNotebook: true,
  },

  // gpu-systems (24-26)
  {
    id: "v3-24",
    set: "v3",
    number: 24,
    title: "Write a Fused Softmax Kernel in Triton",
    difficulty: "expert",
    category: "gpu-systems",
    companies: ["NVIDIA", "Meta", "Google", "xAI", "Tesla"],
    questionPath:
      "v3/gpu-systems/triton-fused-softmax/triton-fused-softmax.ipynb",
    solutionPath:
      "v3/gpu-systems/triton-fused-softmax/triton-fused-softmax_SOLN.ipynb",
    hasNotebook: true,
  },
  {
    id: "v3-25",
    set: "v3",
    number: 25,
    title: "Implement FlashAttention-2 in Triton",
    difficulty: "expert",
    category: "gpu-systems",
    companies: ["NVIDIA", "Meta", "Together AI", "xAI"],
    questionPath:
      "v3/gpu-systems/flash-attention-triton/flash-attention-triton.ipynb",
    solutionPath:
      "v3/gpu-systems/flash-attention-triton/flash-attention-triton_SOLN.ipynb",
    hasNotebook: true,
  },
  {
    id: "v3-26",
    set: "v3",
    number: 26,
    title: "Implement FSDP from Scratch",
    difficulty: "expert",
    category: "gpu-systems",
    companies: ["Meta", "Google", "NVIDIA", "Anthropic", "xAI"],
    questionPath: "v3/gpu-systems/fsdp/fsdp.ipynb",
    solutionPath: "v3/gpu-systems/fsdp/fsdp_SOLN.ipynb",
    hasNotebook: true,
  },

  // alignment-training (27)
  {
    id: "v3-27",
    set: "v3",
    number: 27,
    title: "Implement GRPO (DeepSeek-R1 Algorithm)",
    difficulty: "expert",
    category: "alignment-training",
    companies: ["DeepMind", "Anthropic", "OpenAI"],
    questionPath: "v3/alignment-training/grpo/grpo.ipynb",
    solutionPath: "v3/alignment-training/grpo/grpo_SOLN.ipynb",
    hasNotebook: true,
  },

  // llm-inference (28)
  {
    id: "v3-28",
    set: "v3",
    number: 28,
    title: "Build a Complete LLM Inference Engine",
    difficulty: "expert",
    category: "llm-inference",
    companies: ["Perplexity", "Together AI", "Anyscale", "Fireworks AI"],
    questionPath: "v3/llm-inference/inference-engine/inference-engine.ipynb",
    solutionPath:
      "v3/llm-inference/inference-engine/inference-engine_SOLN.ipynb",
    hasNotebook: true,
  },

  // modern-architectures (29)
  {
    id: "v3-29",
    set: "v3",
    number: 29,
    title: "Implement Knowledge Distillation",
    difficulty: "medium",
    category: "modern-architectures",
    companies: ["Google", "Apple", "Meta", "Qualcomm", "Tesla"],
    questionPath:
      "v3/modern-architectures/knowledge-distillation/knowledge-distillation.ipynb",
    solutionPath:
      "v3/modern-architectures/knowledge-distillation/knowledge-distillation_SOLN.ipynb",
    hasNotebook: true,
  },

  // gpu-systems (30)
  {
    id: "v3-30",
    set: "v3",
    number: 30,
    title: "Implement Ring Attention for Long Contexts",
    difficulty: "expert",
    category: "gpu-systems",
    companies: ["Anthropic", "Google", "Meta", "xAI"],
    questionPath: "v3/gpu-systems/ring-attention/ring-attention.ipynb",
    solutionPath: "v3/gpu-systems/ring-attention/ring-attention_SOLN.ipynb",
    hasNotebook: true,
  },
];

// ---------------------------------------------------------------------------
// Descriptions
// ---------------------------------------------------------------------------

const DESCRIPTIONS: Record<string, string> = {
  // V1
  "v1-1": "Build a linear regression model from scratch using PyTorch tensors and autograd, fitting a line to synthetic data with gradient descent.",
  "v1-2": "Create a custom Dataset class and use DataLoader to batch, shuffle, and iterate over your data for training.",
  "v1-3": "Implement a custom activation function as a PyTorch module and use it inside a neural network.",
  "v1-4": "Build Huber loss from scratch — a robust loss function that's quadratic for small errors and linear for large ones.",
  "v1-5": "Construct a multi-layer feedforward neural network with nonlinear activations, training it on a classification task.",
  "v1-6": "Log training metrics, model graphs, and embeddings to TensorBoard for interactive visualization.",
  "v1-7": "Serialize model weights with state_dict and reload them, handling device mapping and architecture changes.",
  "v1-8": "Build and train a convolutional neural network to classify CIFAR-10 images using conv layers, pooling, and batch norm.",
  "v1-9": "Write a recurrent neural network cell from scratch using raw tensor operations, implementing the hidden state update manually.",
  "v1-10": "Apply random flips, crops, color jitter, and other augmentations to training images using torchvision's transform pipeline.",
  "v1-11": "Profile PyTorch operations using torch.utils.benchmark, measuring GPU/CPU time and comparing implementations.",
  "v1-12": "Build an autoencoder that learns normal data patterns, then detect anomalies by measuring reconstruction error.",
  "v1-13": "Apply post-training quantization to reduce model size and inference latency while maintaining accuracy.",
  "v1-14": "Use automatic mixed precision to train models faster with float16, managing loss scaling and gradient overflow.",
  "v1-15": "Explore how different weight initialization strategies (Xavier, Kaiming, etc.) affect CNN training convergence.",
  "v1-16": "Build convolutional layers from raw tensor operations — implementing forward passes with cross-correlation and learnable filters.",
  "v1-17": "Code the LSTM cell equations from scratch: forget gate, input gate, cell state update, and output gate using raw tensors.",
  "v1-18": "Recreate the AlexNet architecture that started the deep learning revolution, with its 5 conv + 3 FC layer design.",
  "v1-19": "Encode documents and queries into dense vectors and retrieve relevant passages using cosine similarity search.",
  "v1-20": "Build k-nearest neighbors using PyTorch tensor operations for efficient distance computation and majority voting.",
  "v1-21": "Extend 2D convolutions to 3D for volumetric medical image segmentation on CT scans.",
  "v1-22": "Implement SiLU activation with a custom autograd Function, defining both forward and backward passes manually.",
  "v1-23": "Blend the content of one image with the artistic style of another using feature matching in a pretrained CNN.",
  "v1-24": "Implement message passing and node aggregation for graph-structured data using raw PyTorch operations.",
  "v1-25": "Implement GCN layers with adjacency-matrix-based neighborhood aggregation for node classification tasks.",
  "v1-26": "Build the full transformer architecture — multi-head attention, positional encoding, encoder-decoder stacks — from raw tensors.",
  "v1-27": "Implement a generative adversarial network with competing generator and discriminator networks trained via minimax loss.",
  "v1-28": "Build an encoder-decoder model with Bahdanau attention for sequence transduction tasks like translation.",
  "v1-29": "Set up multi-GPU training with DDP, handling process groups, gradient synchronization, and data sharding.",
  "v1-30": "Use PyTorch's sparse tensor formats (COO, CSR) for memory-efficient operations on data with many zeros.",
  "v1-31": "Visualize which image regions drive CNN predictions using Grad-CAM heatmaps and SHAP feature attributions.",
  "v1-32": "Train a simple linear classifier on frozen CLIP embeddings to evaluate representation quality on downstream tasks.",
  "v1-33": "Project high-dimensional embeddings from different modalities into 2D/3D space for visual analysis using t-SNE or UMAP.",
  "v1-34": "Build ViT from scratch — patch embedding, class token, positional encoding, and transformer encoder for image classification.",
  "v1-35": "Build a VAE with encoder, reparameterization trick, decoder, and combined reconstruction + KL divergence loss.",

  // V2
  "v2-1": "Compute KL divergence between two probability distributions from scratch, essential for VAEs and knowledge distillation.",
  "v2-2": "Build Root Mean Square Layer Normalization used in LLaMA and modern transformers — simpler and faster than LayerNorm.",
  "v2-3": "Build the BPE tokenizer algorithm that iteratively merges frequent character pairs to build a subword vocabulary.",
  "v2-4": "Build retrieval-augmented generation by encoding a corpus, performing similarity search, and conditioning generation on retrieved context.",
  "v2-5": "Speed up LLM inference by drafting tokens with a small model and verifying them in parallel with the large model.",
  "v2-6": "Build scaled dot-product attention from raw matrix operations — queries, keys, values, scaling, and softmax.",
  "v2-7": "Split attention into multiple heads with independent projections, compute attention per head, and concatenate results.",
  "v2-8": "Build GQA where multiple query heads share key-value heads, reducing KV cache memory while preserving quality.",
  "v2-9": "Add key-value caching to attention so previously computed keys and values are reused during autoregressive generation.",
  "v2-10": "Build the fixed sinusoidal positional encoding from 'Attention Is All You Need' using sin/cos at different frequencies.",
  "v2-11": "Build Rotary Position Embeddings that encode relative positions by rotating query and key vectors in complex space.",
  "v2-12": "Build a complete small language model end-to-end: tokenizer integration, transformer blocks, and autoregressive text generation.",
  "v2-13": "Implement the GPTQ algorithm for post-training quantization, compressing model weights using Hessian-guided rounding.",
  "v2-14": "Build beam search that maintains top-k partial sequences at each step, producing higher-quality text than greedy decoding.",
  "v2-15": "Restrict token sampling to the k most probable tokens, then renormalize and sample for diverse yet coherent text.",
  "v2-16": "Sample from the smallest set of tokens whose cumulative probability exceeds p, adapting vocabulary size dynamically.",
  "v2-17": "Scale logits by a temperature parameter before softmax to control the randomness of generated text.",
  "v2-18": "Add low-rank adapter matrices to a pretrained layer, enabling efficient fine-tuning with a fraction of the parameters.",
  "v2-19": "Build a gated MoE layer that routes tokens to specialized expert networks, scaling model capacity without proportional compute.",
  "v2-20": "Fine-tune a language model on instruction-following data using supervised fine-tuning with cross-entropy loss.",
  "v2-21": "Align a language model with human preferences using a reward model and reinforcement learning with PPO.",
  "v2-22": "Train a language model directly on preference pairs without a separate reward model using Direct Preference Optimization.",
  "v2-23": "Implement dynamic batching that adds/removes sequences mid-batch to maximize GPU utilization during inference.",
  "v2-24": "Split documents into overlapping chunks optimized for embedding and retrieval in a dense passage retrieval pipeline.",
  "v2-25": "Combine data, tensor, pipeline, sequence, and expert parallelism to train models across thousands of GPUs.",

  // V3
  "v3-1": "Build numerically stable softmax using the log-sum-exp trick, handling overflow and underflow in raw tensor math.",
  "v3-2": "Code Lloyd's algorithm with PyTorch tensors: random init, distance computation, centroid updates, and convergence check.",
  "v3-3": "Build k-nearest neighbors classification using vectorized distance computation and top-k selection on tensors.",
  "v3-4": "Build binary logistic regression from scratch with sigmoid, binary cross-entropy, and manual gradient descent updates.",
  "v3-5": "Build the InfoNCE contrastive loss and a CLIP-style training loop that aligns image and text embeddings.",
  "v3-6": "Build 2D sinusoidal position encodings for vision transformers, encoding both row and column positions in patch grids.",
  "v3-7": "Sort logits, compute cumulative probabilities, mask tokens below the nucleus threshold, and sample from the filtered distribution.",
  "v3-8": "Select the k highest-probability tokens, zero out the rest, renormalize, and sample for controlled text generation.",
  "v3-9": "Maintain and expand top-scoring partial sequences at each decoding step with length normalization and early stopping.",
  "v3-10": "Divide logits by a temperature scalar before softmax to sharpen or flatten the token probability distribution.",
  "v3-11": "Inject trainable low-rank decomposition matrices (A, B) into a frozen linear layer for parameter-efficient fine-tuning.",
  "v3-12": "Cache key and value tensors from previous timesteps so each new token only computes attention over one new position.",
  "v3-13": "Restrict attention to a fixed local window around each token, reducing memory from O(n²) to O(n·w) for long sequences.",
  "v3-14": "Compute the Direct Preference Optimization loss that trains a policy directly from preference pairs without a reward model.",
  "v3-15": "Build Proximal Policy Optimization with clipped surrogate objective, value function baseline, and KL penalty for RLHF.",
  "v3-16": "Trade compute for memory by recomputing intermediate activations during backward instead of storing them all.",
  "v3-17": "Build a gated MoE with top-k routing, load balancing loss, and expert capacity constraints for sparse computation.",
  "v3-18": "Draft tokens with a fast model, verify in parallel with the target model, and accept/reject to guarantee identical output distribution.",
  "v3-19": "Dynamically slot sequences in and out of a running batch as they finish, maximizing throughput without padding waste.",
  "v3-20": "Build the full denoising diffusion pipeline: forward noise schedule, U-Net denoiser, and reverse sampling to generate images.",
  "v3-21": "Build deterministic DDIM sampling with classifier-free guidance to control image generation quality and conditioning adherence.",
  "v3-22": "Build Mamba's selective scan mechanism with input-dependent parameters, achieving linear-time sequence modeling without attention.",
  "v3-23": "Build ViT with masked autoencoder pretraining — randomly mask patches, encode visible ones, decode to reconstruct.",
  "v3-24": "Write a GPU kernel in Triton that fuses the softmax computation into a single pass, eliminating intermediate memory reads.",
  "v3-25": "Write the tiled, fused attention kernel that computes exact attention in O(n) memory using online softmax in Triton.",
  "v3-26": "Build Fully Sharded Data Parallel: shard parameters across GPUs, all-gather before forward, reduce-scatter gradients after backward.",
  "v3-27": "Build Group Relative Policy Optimization that scores multiple completions per prompt and uses group-relative advantages.",
  "v3-28": "Combine KV caching, continuous batching, and memory management into a production-grade inference server.",
  "v3-29": "Train a smaller student model to match a larger teacher's soft predictions using temperature-scaled KL divergence loss.",
  "v3-30": "Distribute attention computation across GPUs in a ring topology, enabling context lengths that exceed single-GPU memory.",
};

// ---------------------------------------------------------------------------
// Combined question list
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Track & LLM Path assignment (new revamp structure)
// ---------------------------------------------------------------------------

// Curated LLM Path: "Implement LLM from Scratch" — recommended order + stages.
// Only items with actual notebooks are included for the initial launch.
const LLM_PATH_CONFIG: Record<string, { order: number; stage: string }> = {
  // Foundations
  "v2-3": { order: 1, stage: "foundations" }, // BPE
  "v2-10": { order: 2, stage: "foundations" }, // Sinusoidal
  "v2-11": { order: 3, stage: "foundations" }, // RoPE
  "v2-6": { order: 4, stage: "foundations" }, // Attention from scratch
  // Core Transformer
  "v2-7": { order: 5, stage: "core-transformer" }, // Multi-Head Attention
  "v2-8": { order: 6, stage: "core-transformer" }, // GQA
  "v3-12": { order: 7, stage: "core-transformer" }, // KV Cache
  "v3-13": { order: 8, stage: "core-transformer" }, // Sliding Window
  // Full Model
  "v2-12": { order: 9, stage: "full-model" }, // SmolLM from Scratch
  // Alignment & PEFT
  "v3-11": { order: 10, stage: "alignment" }, // LoRA
  "v3-14": { order: 11, stage: "alignment" }, // DPO
  "v3-15": { order: 12, stage: "alignment" }, // PPO
  "v3-27": { order: 13, stage: "alignment" }, // GRPO
  // Decoding & Inference
  "v3-10": { order: 14, stage: "decoding-inference" }, // Temperature
  "v3-8": { order: 15, stage: "decoding-inference" }, // Top-k
  "v3-7": { order: 16, stage: "decoding-inference" }, // Top-p
  "v3-18": { order: 17, stage: "decoding-inference" }, // Speculative Decoding
  "v3-19": { order: 18, stage: "decoding-inference" }, // Continuous Batching
  "v3-28": { order: 19, stage: "decoding-inference" }, // Full Inference Engine
  // Systems (capstone for path)
  "v3-17": { order: 20, stage: "systems" }, // MoE
};

function assignTracksAndPath(q: any): Partial<Question> {
  const id = q.id as string;
  const set = q.set as QuestionSet;
  const difficulty = q.difficulty as Difficulty;
  const category = q.category as string | undefined;

  const tracks = new Set<"basics" | "advanced" | "llm-path">();

  // === BASICS ===
  if (set === "v1" && (difficulty === "basic" || difficulty === "easy")) {
    tracks.add("basics");
  }
  // A few accessible medium v1 items that feel foundational
  if (["v1-15", "v1-16", "v1-17"].includes(id)) {
    tracks.add("basics");
  }
  if (set === "v3" && category === "classical-ml") {
    tracks.add("basics");
  }

  // === ADVANCED (default for hard/expert + systems) ===
  if (difficulty === "hard" || difficulty === "expert") {
    tracks.add("advanced");
  }
  if (set === "v3" && ["gpu-systems", "llm-inference", "alignment-training"].includes(category || "")) {
    tracks.add("advanced");
  }
  if (set === "v1" && ["v1-22", "v1-26", "v1-27", "v1-28", "v1-31"].includes(id)) {
    tracks.add("advanced");
  }
  // Modern hard architectures that aren't core LLM path
  if (set === "v3" && category === "modern-architectures" && !LLM_PATH_CONFIG[id]) {
    tracks.add("advanced");
  }

  // === LLM LEARNING PATH (curated) ===
  if (LLM_PATH_CONFIG[id]) {
    tracks.add("llm-path");
    // Many LLM path items are also advanced-level practice
    if (difficulty === "hard" || difficulty === "expert" || ["v3-12", "v3-13", "v3-11"].includes(id)) {
      tracks.add("advanced");
    }
  }

  // Safety: if nothing assigned, put in advanced (most v2/v3 hard stuff)
  if (tracks.size === 0) {
    tracks.add("advanced");
  }

  const pathInfo = LLM_PATH_CONFIG[id];
  return {
    tracks: Array.from(tracks),
    llmPathOrder: pathInfo?.order,
    llmPathStage: pathInfo?.stage,
  };
}

export const questions: Question[] = [
  ...v1Questions,
  ...v2Questions,
  ...v3Questions,
].map((q) => {
  const extra = assignTracksAndPath(q);
  return {
    ...q,
    ...extra,
    description: DESCRIPTIONS[q.id] ?? "",
  } as Question;
});

// ---------------------------------------------------------------------------
// URL helpers
// ---------------------------------------------------------------------------

const GITHUB_RAW =
  "https://raw.githubusercontent.com/Exorust/TorchLeet/main";
const COLAB_BASE =
  "https://colab.research.google.com/github/Exorust/TorchLeet/blob/main";

export function getColabUrl(path: string): string {
  return `${COLAB_BASE}/${path}`;
}

export function getDownloadUrl(path: string): string {
  return `${GITHUB_RAW}/${path}`;
}

// ---------------------------------------------------------------------------
// Query helpers
// ---------------------------------------------------------------------------

export function getQuestionsBySet(set: QuestionSet): Question[] {
  return questions.filter((q) => q.set === set);
}

export function getQuestionsByCategory(category: string): Question[] {
  return questions.filter((q) => q.category === category);
}

export function getQuestionsByDifficulty(difficulty: Difficulty): Question[] {
  return questions.filter((q) => q.difficulty === difficulty);
}

export function getQuestionById(id: string): Question | undefined {
  return questions.find((q) => q.id === id);
}

export function getAllCompanies(): string[] {
  const companySet = new Set<string>();
  for (const q of questions) {
    for (const c of q.companies) {
      companySet.add(c);
    }
  }
  return Array.from(companySet).sort();
}

// ---------------------------------------------------------------------------
// New revamp query helpers (Basics / Advanced / LLM Path + company filter)
// ---------------------------------------------------------------------------

export function getQuestionsByTracks(tracks: ("basics" | "advanced" | "llm-path")[]): Question[] {
  return questions.filter((q) => tracks.some((t) => q.tracks.includes(t)));
}

export function getBasicsQuestions(): Question[] {
  return questions.filter((q) => q.tracks.includes("basics"));
}

export function getAdvancedQuestions(): Question[] {
  return questions.filter((q) => q.tracks.includes("advanced"));
}

export function getLLMPathQuestions(): Question[] {
  return questions
    .filter((q) => q.tracks.includes("llm-path") && typeof q.llmPathOrder === "number")
    .sort((a, b) => (a.llmPathOrder ?? 999) - (b.llmPathOrder ?? 999));
}

export function getLLMPathStages(): { stage: string; title: string; questions: Question[] }[] {
  const pathQs = getLLMPathQuestions();
  const stageOrder = ["foundations", "core-transformer", "full-model", "alignment", "decoding-inference", "systems"];
  const stageTitles: Record<string, string> = {
    foundations: "1. Foundations (Tokenization & Position)",
    "core-transformer": "2. Core Transformer Building Blocks",
    "full-model": "3. Full Small Language Model",
    alignment: "4. Alignment & Efficient Fine-Tuning",
    "decoding-inference": "5. Decoding & Efficient Inference",
    systems: "6. Systems & Scaling",
  };

  const grouped = new Map<string, Question[]>();
  for (const q of pathQs) {
    const st = q.llmPathStage || "other";
    if (!grouped.has(st)) grouped.set(st, []);
    grouped.get(st)!.push(q);
  }

  return stageOrder
    .filter((s) => grouped.has(s))
    .map((stage) => ({
      stage,
      title: stageTitles[stage] || stage,
      questions: grouped.get(stage)!,
    }));
}

/** Filter a list of questions by one or more companies (case-insensitive, OR) */
export function filterQuestionsByCompanies(qs: Question[], companies: string[]): Question[] {
  if (!companies || companies.length === 0) return qs;
  const lowers = companies.map((c) => c.toLowerCase());
  return qs.filter((q) =>
    q.companies.some((c) => lowers.includes(c.toLowerCase()))
  );
}

export function getQuestionsByCompany(company: string): Question[] {
  const lower = company.toLowerCase();
  return questions.filter((q) =>
    q.companies.some((c) => c.toLowerCase() === lower),
  );
}

// ---------------------------------------------------------------------------
// Advanced list topic grouping (for sorted topic view, mirroring LLM Path)
// ---------------------------------------------------------------------------

export function getAdvancedGroupedByTopic(): { topic: string; title: string; questions: Question[] }[] {
  const adv = getAdvancedQuestions();

  const topicDefs: Array<{ topic: string; title: string; filter: (q: Question) => boolean }> = [
    {
      topic: "gpu-kernels",
      title: "GPU Systems & Kernels",
      filter: (q) => q.category === "gpu-systems" || /triton|flash.?attention|fsdp|ring.?attention/i.test(q.title),
    },
    {
      topic: "modern-arch",
      title: "Modern Architectures",
      filter: (q) => q.category === "modern-architectures" || /mamba|mixture of experts|vit|mae|ddpm|ddim|diffusion|knowledge distillation/i.test(q.title),
    },
    {
      topic: "inference-systems",
      title: "Advanced Inference & Systems",
      filter: (q) => q.category === "llm-inference" || /inference engine|continuous batching|speculative decoding/i.test(q.title),
    },
    {
      topic: "alignment-systems",
      title: "Alignment & Training Systems",
      filter: (q) => q.category === "alignment-training" || /gradient checkpoint|grpo|ppo|dpo/i.test(q.title),
    },
    {
      topic: "hard-foundations",
      title: "Hard Foundations & XAI",
      filter: (q) =>
        (q.set === "v1" && q.difficulty === "hard") ||
        /custom autograd|gan|seq-to-seq|explainable|xai|graph neural/i.test(q.title),
    },
  ];

  return topicDefs
    .map((def) => {
      const qs = adv.filter(def.filter);
      return qs.length > 0 ? { topic: def.topic, title: def.title, questions: qs } : null;
    })
    .filter((g): g is { topic: string; title: string; questions: Question[] } => g !== null);
}
