# Assignment 1
Build and train a decoder-only Transformer language model from scratch.

## Training Experiments

Training a 6-layer decoder-only Transformer on TinyStories with the following approximate configuration:

| Component | Value |
|---|---:|
| Vocabulary size | 10,000 |
| Context length | 256 |
| Number of layers | 6 |
| Model dimension | 512 |
| FFN dimension | 1344 |
| Dataset | TinyStories |
| Total tokens | 327,680,000 |
| Gradient clipping | `max_norm = 1.0` |

The experiments sweep over learning rate and batch size while keeping the total token budget fixed. TensorBoard was used to track training loss, learning rate schedule, and gradient norm.
<img width="1221" height="715" alt="image" src="https://github.com/user-attachments/assets/65556af8-5f6c-4c59-be7f-1e749ae55315" />



## Observations

### 1. Larger batch size improves loss decrease per optimizer step

A clear pattern is that larger batch sizes reduce the number of optimizer steps required to process the same number of tokens, while the loss often decreases faster when plotted against optimizer steps.

This does **not** mean that a larger batch produces a larger gradient magnitude or a larger update magnitude per step. Since the loss is averaged over the batch, the gradient scale is not expected to grow linearly with batch size.

Instead, the main reason is that each optimizer step is estimated from more training examples. A larger batch gives a lower-variance gradient estimate, so the update direction is usually more stable and better aligned with the true full-dataset gradient. In other words, each step “sees” more data.

Therefore, when the x-axis is optimizer steps, larger batch sizes can appear to train faster because every step contains more tokens. A fair comparison should also consider loss versus tokens processed or wall-clock time.
<img width="305" height="541" alt="image" src="https://github.com/user-attachments/assets/c83ffa05-efa5-42ee-8f8d-970c86f8e0cc" />


### 2. Learning rate and gradient norm show an empirical inverse relationship

Across several runs, models trained with the same learning rate tend to converge toward similar gradient-norm ranges. More interestingly, when the learning rate is increased by approximately 3x, the observed gradient norm often decreases toward roughly 1/3 of the previous value.

This suggests an empirical compensation effect:

```text
effective update scale ≈ learning_rate × gradient_norm
```
In many stable runs, the product of learning rate and gradient norm appears to remain in a similar range. One possible explanation is that a larger learning rate moves the model parameters more aggressively into regions of the loss landscape where the local gradient norm is smaller. This does not mean that the optimizer explicitly enforces grad_norm ∝ 1 / lr; rather, it is an observed consequence of the interaction between the learning rate, the loss landscape, Adam-style optimization, and gradient clipping.

### 3. Too large a learning rate causes instability or divergence
Although increasing the learning rate can speed up training, it also makes optimization less stable. When the learning rate becomes too large, the model may overshoot useful regions of the loss landscape, causing the training loss and gradient norm to oscillate heavily or diverge.

This is especially visible in runs where the training loss stops decreasing and becomes dominated by high-frequency noise.
Here lr = 0.01
<img width="1219" height="466" alt="image" src="https://github.com/user-attachments/assets/b9e502a7-1cf7-4363-af30-e2a99719aa1c" />

