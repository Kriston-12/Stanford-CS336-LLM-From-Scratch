import torch
from torch import Tensor
from dataclasses import dataclass
from implementation.checkpointing import save_checkpoint, load_checkpoint
from implementation.data_loading import DataLoader
from implementation.consineLR_scheduler import consine_lr_schedule
from implementation.AdamW import AdamW
from implementation.transformer_LM import Transformer
import numpy as np

@dataclass
class ModelConfig:
    vocab_size: int
    context_length: int
    d_model: int
    num_layers: int
    num_heads: int
    d_ff: int
    rope_theta: float = 10000.0
    weights: dict[str, Tensor] | None = None

@dataclass
class OptimizerConfig:
    lr: float = 3e-4
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    max_grad_norm: float | None = 1.0

@dataclass
class TrainingConfig:
    model: ModelConfig
    optimizer: OptimizerConfig
    train_data_path: str
    val_data_path: str | None = None
    split_val_from_train: bool = True
    batch_size: int = 32
    total_steps: int = 10000
    warmup_steps: int = 1000
    min_lr: float = 3e-5
    device: str = "cuda"
    eval_every: int = 500
    log_every: int = 50
    checkpoint_every: int = 1000
    checkpoint_path: str = "checkpoints/latest.pt"

def build_model(config: ModelConfig):
    if config.weights is None:
        weights = {}
        for i in range(config.num_layers):
            weights[f"layers.{i}.attn.q_proj.weight"] = torch.empty((config.d_model, config.d_model))
            weights[f"layers.{i}.attn.k_proj.weight"] = torch.empty((config.d_model, config.d_model))
            weights[f"layers.{i}.attn.v_proj.weight"] = torch.empty((config.d_model, config.d_model))
            weights[f"layers.{i}.attn.output_proj.weight"] = torch.empty((config.d_model, config.d_model))
            weights[f"layers.{i}.ln1.weight"] = torch.empty((config.d_model,))
            weights[f"layers.{i}.ffn.w1.weight"] = torch.empty((config.d_ff, config.d_model))
            weights[f"layers.{i}.ffn.w3.weight"] = torch.empty((config.d_ff, config.d_model))
            weights[f"layers.{i}.ffn.w2.weight"] = torch.empty((config.d_model, config.d_ff))
            weights[f"layers.{i}.ln2.weight"] = torch.empty((config.d_model,))
        weights["token_embeddings.weight"] = torch.empty((config.vocab_size, config.d_model))
        weights["ln_final.weight"] = torch.empty((config.d_model,))
        weights["lm_head.weight"] = torch.empty((config.vocab_size, config.d_model))
        config.weights = weights
    return Transformer(
        **config
    )

def build_optimizer(model: Transformer, config: OptimizerConfig):
    return AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
        betas=(config.beta1, config.beta2),
        eps=config.eps
    )


def train(config: TrainingConfig):
    train_data = None
    val_data = None
    if config.split_val_from_train:
        entire_dataset = np.memmap(config.train_data_path, dtype=np.int32, mode='r')
        split_idx = int(len(entire_dataset) * 0.9)
        train_data = entire_dataset[:split_idx]
        val_data = entire_dataset[split_idx:]
    else:
        train_data = np.memmap(config.train_data_path, dtype=np.int32, mode='r')
        val_data = np.memmap(config.val_data_path, dtype=np.int32, mode='r')

    model = build_model(config.model).to(config.device)
    optimizer = build_optimizer(model, config.optimizer)

    train_data_loader = DataLoader(train_data, config.batch_size, config.context_length)
    val_data_loader = DataLoader(val_data, config.batch_size, config.context_length)
    # Training loop to be done
    for step in range(config.total_steps):
        # Sample a batch of data
        # Forward pass
        # Compute loss
        # Backward pass and optimization step
        # Logging and checkpointing
        x, y = train_data_loader.get_batch()