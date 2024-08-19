# Bigram Model - README

## Overview

This repository contains a Python implementation of a Bigram Language Model. The model is designed to predict the next word in a sequence based on the probability of word pairs (bigrams). This implementation is optimized to run on a MacBook, leveraging Apple's MPS (Metal Performance Shaders) for efficient computation on M1/M2 chips.

## Prerequisites

Before running the code, ensure you have the following prerequisites installed:

- **Python 3.8+**: The code is compatible with Python 3.8 and above.
- **PyTorch**: Install the latest version of PyTorch that supports Apple's MPS.
- **Apple Silicon**: This code is optimized for MacBooks with M1/M2 chips.

### Python Dependencies

To install the necessary Python packages, you can use `pip`:

```bash
pip install torch numpy
```

## Running the Model

### 1. Clone the Repository

First, clone the repository to your local machine:

```bash
git clone <repository-url>
cd <repository-directory>
```

### 2. Run the Bigram Model

To run the bigram model, execute the following command in your terminal:

```bash
python bigram_model.py
```

### 3. Leveraging MPS (Apple Silicon)

The code is configured to use Apple’s MPS backend if available. This enables efficient training and inference on MacBooks with M1/M2 chips.

Ensure that the tensors and the model are correctly moved to the MPS device as shown in the code:

```python
import torch

# Check if MPS is available and set the device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Move your model and tensors to the device
model.to(device)
input_tensor = input_tensor.to(device)
```

## Troubleshooting

### Common Issues

1. **MPS Device Not Recognized**: Ensure that your PyTorch installation supports MPS. If the device is not recognized, it might be due to an outdated version of PyTorch or macOS.
2. **Performance Issues**: If you experience slow performance, try reducing the batch size or sequence length, as larger values may cause excessive memory usage.
