# CliffordNet: Clifford Algebra-based Neural Network for CIFAR-100

## Overview

This is a deep learning model implementation based on Clifford algebra (geometric algebra) for CIFAR-100 image classification tasks. The model achieves feature interaction through geometric product operations and employs a gated geometric residual connection mechanism.

## Training Results

### Key Performance Metrics
| Metric | Nano               | Fast  | Base  |
|--------|--------------------|-------|-------|
| Best Test Accuracy | **77.94%** (Epoch 192) | **81.18%**  (Epoch 199)   | **81.95%**  (Epoch 200)     |
| Results in the paper | 76.41%             | 77.63% | 78.05% |

## Model Architecture

### Core Components

#### 1. CliffordInteraction Module
This module implements a feature interaction mechanism based on Clifford algebra:

- **Dual Stream Generation**:
  - **Context Stream**: Extracts spatial context information through depthwise convolution
  - **State Stream**: Performs channel projection through 1×1 convolution

- **Sparse Rolling Interaction**:
  - Uses cyclic shift operation (`torch.roll`) to implement translation operators
  - Computes two components of the geometric product:
    - **Scalar Component (Dot)**: `SiLU(H * Ts(C))`
    - **Bivector Component (Wedge)**: `H * Ts(C) - Ts(H) * C`
  - Supports multi-scale interaction (controlled by shifts parameter)

#### 2. CliffordBlock
Complete transformation block containing:
- GroupNorm normalization
- CliffordInteraction geometric interaction
- **Gated Geometric Residual (GGR)**:
  ```
  M = Cat([X_ln, G_feat])
  alpha = Sigmoid(Linear_gate(M))
  H_mix = SiLU(X_ln) + alpha * G_feat
  X = X_prev + Drop(gamma * H_mix)
  ```
- Layer Scale and DropPath regularization

#### 3. CliffordNet (Complete Network)
- **Stem**: Convolution + Batch Normalization + SiLU activation
- **Backbone**: Multiple stacked CliffordBlocks
- **Head**: Global Average Pooling + Linear classifier

### Model Variants

#### CliffordNet-Nano
- Parameters: ~1.4M
- Embedding Dimension: 128
- Depth: 12 layers
- Shift Parameters: [1, 2]
- DropPath Rate: 0.05

#### CliffordNet-Fast
- Parameters: ~2.6M
- Embedding Dimension: 160
- Depth: 12 layers
- Shift Parameters: [1, 2, 4, 8, 15] (richer multi-scale interaction)
- DropPath Rate: 0.1

## Training Configuration

### Hyperparameters
- **Dataset**: CIFAR-100 (100-class image classification)
- **Batch Size**: 128
- **Training Epochs**: 200 epochs
- **Optimizer**: AdamW
  - Learning Rate (initial): 1e-3
  - Weight Decay: 0.05
  - Gradient Clipping: max_norm=1.0

### Learning Rate Scheduling Strategy
Uses SequentialLR combined strategy:
1. **Warmup Phase**: First 5 epochs
   - Linear increase from 0.001× to 1.0× initial learning rate
2. **Cosine Annealing**: Epochs 6-200
   - Cosine decay from 1e-3 to 1e-6

### Data Augmentation
**Training Set Augmentation**:
- RandomCrop(32, padding=4)
- RandomHorizontalFlip
- TrivialAugmentWide (auto augmentation)
- Normalize (mean and std normalization)
- RandomErasing (p=0.1, random erasing)

**Test Set**:
- Normalize only




## Usage

### Requirements
```bash
torch>=1.9.0
torchvision>=0.10.0
timm>=0.4.12
```

### Run Training
```python
python train.py
```

### Custom Model
```python
# Use Nano model
model = cliffordnet_nano(num_classes=100)

# Use Fast model
model = cliffordnet_fast(num_classes=100)

# Custom configuration
model = CliffordNet(
    img_size=32,
    in_chans=3,
    num_classes=100,
    embed_dim=192,      # Adjust channel dimension
    depth=12,            # Adjust depth
    shifts=[1, 2, 4],   # Adjust multi-scale configuration
    drop_path_rate=0.1
)
```

## Performance Comparison

CliffordNet-Nano achieves **77.94%** test accuracy on CIFAR-100 with only **1.4M parameters**, demonstrating excellent performance for a lightweight model.

## Future Improvements

1. Test CliffordNet-Fast model (more shifts) performance
2. Experiment with stronger data augmentation strategies (e.g., Mixup, CutMix)
3. Try knowledge distillation or model ensemble
4. Adjust balance between architecture depth and width
5. Explore other datasets (ImageNet, CIFAR-10, etc.)

## Citation

If you use this code, please cite the relevant research work on Clifford algebra and neural networks.

## License

MIT License





