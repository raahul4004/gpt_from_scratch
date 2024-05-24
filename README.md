# GPT from Scratch

This repository contains an implementation of a GPT (Generative Pretrained Transformer) model from scratch using PyTorch. The code demonstrates the complete process of building, training, and generating text with a transformer-based language model. This project is designed to be a hands-on learning experience for understanding the inner workings of transformer models.

## Features

- Tokenizes and processes text data
- Implements key components of a transformer model including multi-head self-attention and feed-forward neural networks
- Trains the model on a given text dataset
- Generates new text based on the trained model

## Requirements

- Python 3.6 or higher
- PyTorch
- CUDA (optional, for GPU acceleration)

## Getting Started

### Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/gpt-from-scratch.git
    cd gpt-from-scratch
    ```

2. Install the required packages:
    ```bash
    pip install torch
    ```

### Usage

1. Prepare your training data:
    - Place your text data in a file named `input.txt` in the root directory of the project.

2. Run the training script:
    ```bash
    python train.py
    ```

### Code Overview

#### Hyperparameters and Configuration

```python
batch_size = 64
block_size = 256
max_iters = 5000
learning_rate = 3e-4
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 200
eval_interval = 500
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
```

#### Data Preparation

- The text data is read from `input.txt`.
- Characters are encoded into integers and split into training and validation sets.

#### Model Components

- **Head**: One head of self-attention.
- **MultiHeadAttention**: Multiple self-attention heads in parallel.
- **FeedForward**: A feed-forward neural network.
- **Block**: A transformer block consisting of multi-head attention and feed-forward layers.
- **GPTLanguageModel**: The main model combining the embedding layers, transformer blocks, and a linear layer for output.

#### Training Loop

- The model is trained using the AdamW optimizer.
- Loss is calculated using cross-entropy and evaluated on both training and validation sets.
- The model's performance is logged at regular intervals.

#### Text Generation

- The `generate` method in `GPTLanguageModel` is used to generate new text sequences based on the trained model.

### Example

```bash
python train.py
```

This will train the model on the data provided in `input.txt` and periodically output training and validation losses.

## Acknowledgements

This project is inspired by various implementations of GPT models and aims to provide a clear and educational implementation of transformer models from scratch.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

Feel free to modify and expand upon this project. Contributions are welcome! If you encounter any issues or have suggestions, please open an issue or submit a pull request.

Happy coding!

