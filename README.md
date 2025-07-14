<div align="center">
  <img src="torchleet-llm.png" alt="Robot Image">
  <!-- <h1>TorchLeet</h1> -->
  <p align="center">
    🐦 <a href="https://twitter.com/charoori_ai">Follow me on Twitter</a> •
    ➡️ <a href="https://github.com/Exorust/TorchLeet/tree/new-llm?tab=readme-ov-file#llm-set">Jump to LLMs!</a>
    📧 <a href="mailto:chandrahas.aroori@gmail.com?subject=LLM%20Cookbook">Feedback</a>
  </p>
  <p>
    <a href="https://www.buymeacoffee.com/charoori_ai" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/default-orange.png" alt="Buy Me A Coffee" height="39" width="170"></a>
  </p>
</div>
<br/>

TorchLeet is broken into two sets of questions:
1. **Question Set**: A collection of PyTorch practice problems, ranging from basic to hard, designed to enhance your skills in deep learning and PyTorch.
2. **LLM Set**: A new set of questions focused on understanding and implementing Large Language Models (LLMs) from scratch, including attention mechanisms, embeddings, and more.

> [!NOTE]
> Avoid using GPT. Try to solve these problems on your own. The goal is to learn and understand PyTorch concepts deeply.

>  [!NOTE] 
> Yes. I used GPT to help write the code and I ended up testing it out myself as practise. I found the strategy to be super useful

## Table of Contents
- [Question Set](#question-set)
   - [🔵Basic](#basic)
   - [🟢Easy](#easy)
   - [🟡Medium](#medium)
   - [🔴Hard](#hard)
- [LLM Set](#llm-set)
- [Getting Started](#getting-started)
   - [1. Install Dependencies](#1-install-dependencies)
   - [2. Structure](#2-structure)
   - [3. How to Use](#3-how-to-use)
- [Contribution](#contribution)


## Question Set

### 🔵Basic
Mostly for beginners to get started with PyTorch.

1. [Implement linear regression](torch/basic/lin-regression/lin-regression.ipynb) [(Solution)](torch/basic/lin-regression/lin-regression_SOLN.ipynb)
2. [Write a custom Dataset and Dataloader to load from a CSV file](torch/basic/custom-dataset/custom-dataset.ipynb) [(Solution)](torch/basic/custom-dataset/custom-dataset_SOLN.ipynb) 
3. [Write a custom activation function (Simple)](torch/basic/custom-activation/custom-activation.ipynb) [(Solution)](torch/basic/custom-activation/custom-activation_SOLN.ipynb)
4. [Implement Custom Loss Function (Huber Loss)](torch/basic/custom-loss/custom-loss.ipynb) [(Solution)](torch/basic/custom-loss/custom-loss_SOLN.ipynb)  
5. [Implement a Deep Neural Network](torch/basic/custom-DNN/custon-DNN.ipynb) [(Solution)](torch/basic/custom-DNN/custon-DNN_SOLN.ipynb)  
6. [Visualize Training Progress with TensorBoard in PyTorch](torch/basic/tensorboard/tensorboard.ipynb) [(Solution)](torch/basic/tensorboard/tensorboard_SOLN.ipynb)  
7. [Save and Load Your PyTorch Model](torch/basic/save-model/save_model.ipynb) [(Solution)](torch/basic/save-model/save_model_SOLN.ipynb) 
10. Implement Softmax function from scratch

---

### 🟢Easy
Recommended for those who have a basic understanding of PyTorch and want to practice their skills.
1. [Implement a CNN on CIFAR-10](torch/easy/cnn/CNN.ipynb) [(Solution)](torch/easy/cnn/CNN_SOLN.ipynb)  
2. [Implement an RNN from Scratch](torch/easy/rnn/RNN.ipynb) [(Solution)](torch/easy/rnn/RNN_SOLN.ipynb)  
3. [Use `torchvision.transforms` to apply data augmentation](torch/easy/augmentation/augmentation.ipynb) [(Solution)](torch/easy/augmentation/augmentation_SOLN.ipynb)  
4. [Add a benchmark to your PyTorch code](torch/easy/benchmark/bench.ipynb) [(Solution)](torch/easy/benchmark/bench_SOLN.ipynb)  
5. [Train an autoencoder for anomaly detection](torch/easy/autoencoder/autoencoder.ipynb) [(Solution)](torch/easy/autoencoder/autoencoder_SOLN.ipynb)
6. [Quantize your language model](torch/easy/quantize-lm/quantize-language-model.ipynb) [(Solution)](torch/easy/quantize-lm/quantize-language-model_SOLN.ipynb)
7. [Implement Mixed Precision Training using torch.cuda.amp](torch/easy/cuda-amp/cuda-amp.ipynb) [(Solution)](torch/easy/cuda-amp/cuda-amp_SOLN.ipynb)
   
---

### 🟡Medium 
These problems are designed to challenge your understanding of PyTorch and deep learning concepts. They require you to implement things from scratch or apply advanced techniques.
1. [Implement parameter initialization for a CNN](torch/medium/cnn-param-init/CNN_ParamInit.ipynb) [(Solution)](torch/medium/cnn-param-init/CNN_ParamInit_SOLN.ipynb)
2. [Implement a CNN from Scratch](torch/medium/cnn-scratch/CNN_scratch.ipynb) [(Solution)](torch/medium/cnn-scratch/CNN_scratch_SOLN.ipynb) 
3. [Implement an LSTM from Scratch](torch/medium/lstm/LSTM.ipynb) [(Solution)](torch/medium/lstm/LSTM_SOLN.ipynb)  
4. Implement AlexNet from scratch 
5. Build a Dense Retrieval System using PyTorch
6.  Implement KNN from scratch in PyTorch
7.  [Train a 3D CNN network for segmenting CT images](torch/medium/3dcnn/3DCNN.ipynb) [Solution](torch/medium/3dcnn/3DCNN_SOLN.ipynb)

---

### 🔴Hard
These problems are for advanced users who want to push their PyTorch skills to the limit. They involve complex architectures, custom layers, and advanced techniques.
1. [Write a custom Autograd function for activation (SILU)](torch/hard/custom-autograd/custom-autgrad-function.ipynb) [(Solution)](torch/hard/custom-autograd/custom-autgrad-function_SOLN.ipynb)
2. Write a Neural Style Transfer  
3. Build a Graph Neural Network (GNN) from scratch
4. Build a Graph Convolutional Network (GCN) from scratch
5. [Write a Transformer](torch/hard/transformer/transformer.ipynb) [(Solution)](torch/hard/transformer/transformer_SOLN.ipynb)  
6. [Write a GAN](torch/hard/GAN/GAN.ipynb) [(Solution)](torch/hard/GAN/GAN_SOLN.ipynb)  
7. [Write Sequence-to-Sequence with Attention](torch/hard/seq-seq/seq-to-seq-with-Attention.ipynb) [(Solution)](torch/hard/seq-seq/seq-to-seq-with-Attention_SOLN.ipynb)  
8. [Enable distributed training in pytorch (DistributedDataParallel)]
9. [Work with Sparse Tensors]
10. [Add GradCam/SHAP to explain the model.](torch/hard/xai/xai.ipynb) [(Solution)](torch/hard/xai/xai_SOLN.ipynb)
11. Linear Probe on CLIP Features
12. Add Cross Modal Embedding Visualization to CLIP (t-SNE/UMAP)
13. Implement a Vision Transformer
14. Implement a Variational Autoencoder

---

## LLM Set

**An all new set of questions to help you understand and implement Large Language Models from scratch.**

Each question is designed to take you one step closer to building your own LLM.

1. Implement KL Divergence Loss
2. Implement RMS Norm
3. [Implement Byte Pair Encoding from Scratch](llm/Byte-Pair-Encoder/BPE-q3.ipynb) [(Solution)](llm/Byte-Pair-Encoder/BPE-q3.ipynb)
4. Create a RAG Search of Embeddings from a set of Reviews
5. Implement Predictive Prefill with Speculative Decoding
6. [Implement Attention from Scratch](llm/Implement-Attention-from-Scratch/attention-q4-Question.ipynb) [(Solution)](llm/Implement-Attention-from-Scratch/attention-q4.ipynb)
7. [Implement Multi-Head Attention from Scratch](llm/Multi-Head-Attention/multi-head-attention-q5-Question.ipynb) [(Solution)](llm/Multi-Head-Attention/multi-head-attention-q5.ipynb)
8. [Implement Grouped Query Attention from Scratch](llm/Grouped-Query-Attention/grouped-query-attention-Question.ipynb) [(Solution)](llm/Grouped-Query-Attention/grouped-query-attention.ipynb)
9. Implement KV Cache in Multi-Head Attention from Scratch
10. [Implement Sinusoidal Embeddings](llm/Sinusoidal-Positional-Embedding/sinusoidal-q7-Question.ipynb) [(Solution)](llm/Sinusoidal-Positional-Embedding/sinusoidal-q7.ipynb)
11. [Implement ROPE Embeddings](llm/Rotary-Positional-Embedding/rope-q8-Question.ipynb) [(Solution)](llm/Rotary-Positional-Embedding/rope-q8.ipynb)
12. [Implement SmolLM from Scratch](llm/SmolLM/smollm-q12-Question.ipynb) [(Solution)](llm/SmolLM/smollm-q12.ipynb)
13. Implement Quantization of Models
    1.  GPTQ
14. Implement Beam Search atop LLM for decoding
15. Implement Top K Sampling atop LLM for decoding
16. Implement Top p Sampling atop LLM for decoding
17. Implement Temperature Sampling atop LLM for decoding
18. Implement LoRA on a layer of an LLM
    1.  QLoRA
19. Mix two models to create a mixture of Experts
20. Apply SFT on SmolLM 
21. Apply RLHF on SmolLM
22. Implement DPO based RLHF
23. Add continuous batching to your LLM
24. Chunk Textual Data for Dense Passage Retrieval
25. Implement Large scale Training => 5D Parallelism

**What's cool? 🚀**
- **Diverse Questions**: Covers beginner to advanced PyTorch concepts (e.g., tensors, autograd, CNNs, GANs, and more).
- **Guided Learning**: Includes incomplete code blocks (`...` and `#TODO`) for hands-on practice along with Answers

## Getting Started

### 1. Install Dependencies
- Install pytorch: [Install pytorch locally](https://pytorch.org/get-started/locally/)
- Some problems need other packages. Install as needed.

### 2. Structure
- `<E/M/H><ID>/`: Easy/Medium/Hard along with the question ID.
- `<E/M/H><ID>/qname.ipynb`: The question file with incomplete code blocks.
- `<E/M/H><ID>/qname_SOLN.ipynb`: The corresponding solution file.

### 3. How to Use
- Navigate to questions/ and pick a problem
- Fill in the missing code blocks `(...)` and address the `#TODO` comments.
- Test your solution and compare it with the corresponding file in `solutions/`.

**Happy Learning! 🚀**


# Contribution
Feel free to contribute by adding new questions or improving existing ones. Ensure that new problems are well-documented and follow the project structure. Submit a PR and tag the authors.

# Authors

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
        💻 AI/ML Dev
        <br />
        <a href="ttps://twitter.com/charoori_ai" target="_blank">
          <img src="https://upload.wikimedia.org/wikipedia/commons/6/60/Twitter_Logo_as_of_2021.svg" width="20px;" alt="Twitter"/>
        </a> 
        <a href="https://www.linkedin.com/in/chandrahas-aroori/" target="_blank">
          <img src="https://upload.wikimedia.org/wikipedia/commons/0/0e/LinkedIn_Logo_2013.svg" width="20px;" alt="LinkedIn"/>
        </a>
        <a href="mailto:charoori@bu.edu" target="_blank">
          <img src="https://upload.wikimedia.org/wikipedia/commons/a/a6/Email_icon.svg" width="20px;" alt="Email"/>
        </a>
      </td>
      <td align="center">
        <a href="https://github.com/CaslowChien">
          <img src="https://avatars.githubusercontent.com/u/99608452?v=4" width="100px;" alt="Caslow Chien"/>
          <br />
          <b>Caslow Chien</b>
        </a>
        <br />
        💻 Developer
        <br />
        <a href="https://caslowchien.github.io/caslow.github.io/" target="_blank">
          <img src="https://upload.wikimedia.org/wikipedia/commons/6/60/Twitter_Logo_as_of_2021.svg" width="20px;" alt="Website"/>
        </a> 
        <a href="https://www.linkedin.com/in/caslow/" target="_blank">
          <img src="https://upload.wikimedia.org/wikipedia/commons/0/0e/LinkedIn_Logo_2013.svg" width="20px;" alt="LinkedIn"/>
        </a>
        <a href="mailto:caslow@bu.edu" target="_blank">
          <img src="https://upload.wikimedia.org/wikipedia/commons/a/a6/Email_icon.svg" width="20px;" alt="Email"/>
        </a>
      </td>
    </tr>
  </table>
</div>

                        
## Stargazers over time
[![Stargazers over time](https://starchart.cc/Exorust/TorchLeet.svg?variant=adaptive)](https://starchart.cc/Exorust/TorchLeet)
