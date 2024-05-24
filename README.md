GPT from Scratch

This repository contains an implementation of a Generative Pre-trained Transformer (GPT) from scratch using PyTorch. The model is trained on a character-level language modeling task. This implementation serves as a basic starting point for understanding and experimenting with GPT models.
Overview

This project includes:

    Data preprocessing
    Model architecture
    Training loop
    Text generation

Requirements

    Python 3.7+
    PyTorch

Setup

    Clone the repository:

    git clone https://github.com/yourusername/gpt-from-scratch.git
    cd gpt-from-scratch

    Install the required dependencies:

    pip install torch

    Place your training data in a file named input.txt in the project directory.

Hyperparameters

The following hyperparameters are used for training the model:

batch_size = 64        # Number of sequences processed in parallel
block_size = 256       # Length of the context for predictions
max_iters = 5000       # Number of training iterations
learning_rate = 3e-4   # Learning rate for the optimizer
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 200       # Number of evaluation iterations
eval_interval = 500    # Interval for evaluating the model on validation set
n_embd = 384           # Embedding dimension
n_head = 6             # Number of attention heads
n_layer = 6            # Number of transformer layers
dropout = 0.2          # Dropout rate

Usage
Data Preparation

Ensure your input.txt file contains the text data you want to train on. The code reads this file, builds a vocabulary of unique characters, and splits the data into training and validation sets.
Training

Run the script to start training the model:

python train.py

During training, the model's loss on both the training and validation sets will be periodically printed.
Text Generation

After training, the model generates text based on a given context. The generated text will be printed at the end of the training script.
Code Explanation
Model Components
Self-Attention Head

The Head class implements a single head of self-attention.
Multi-Head Attention

The MultiHeadAttention class combines multiple attention heads.
FeedForward Network

The FeedForward class implements a simple feed-forward neural network.
Transformer Block

The Block class represents a transformer block, which consists of multi-head attention followed by a feed-forward network.
GPT Language Model

The GPTLanguageModel class combines all the components to form the GPT model. It includes methods for forward propagation and text generation.
Training Loop

The training loop iterates over the dataset, evaluates the model periodically, and prints the training and validation losses. After training, the model generates text based on a given context.
Example

Below is an example of how to run the script and generate text:

    Prepare input.txt with your training data.
    Run the training script:

    python train.py

    The script will print the generated text at the end of the training process.

Notes

    This implementation is a basic starting point. For more advanced use cases, consider using well-established libraries such as Hugging Face's transformers.
    Adjust hyperparameters as needed based on your specific dataset and computational resources.

License

This project is licensed under the MIT License. See the LICENSE file for details.
