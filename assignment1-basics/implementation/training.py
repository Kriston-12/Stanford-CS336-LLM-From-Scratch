import base64
import hashlib
import torch
from torch import Tensor
from dataclasses import dataclass
from implementation.checkpointing import save_checkpoint, load_checkpoint
from implementation.data_loading import DataLoader
from implementation.consineLR_scheduler import consine_lr_schedule
from implementation.AdamW import AdamW
from implementation.tokenizer import Tokenizer
from implementation.transformer_LM import Transformer
from implementation.cross_entropy import CrossEntropyLoss
from implementation.gradient_clipper import clip_gradients
import numpy as np
import json
import os
from typing import Tuple
# import matplotlib 
# import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

# matplotlib.use('Agg') # Use a non-interactive backend for matplotlib

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
    vocab_path: str
    merges_path: str
    text_path: str
    split_ratio: float = 0.9 # default attributes must be after non-default attributes
    batch_size: int = 32
    total_steps: int = 10000
    warmup_steps: int = total_steps * 0.1
    consine_cycle_iters: int = total_steps - 2 * warmup_steps
    max_learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    device: str = "cuda"
    eval_every: int = 500
    log_every: int = 50
    checkpoint_every: int = 1000
    checkpoint_path: str = "checkpoints/latest.pt"
    run_name: str = "default_run"

def _b64(b: bytes) -> str:
    return base64.b64encode(b).decode("ascii")

def _unb64(s: str) -> bytes:
    return base64.b64decode(s.encode("ascii"))

def _tokenizer_artifact_paths(
    vocab_path: str,
    merges_path: str,
    text_path: str,
) -> tuple[str, str]:
    text_stem = os.path.splitext(os.path.basename(text_path))[0]

    vocab_root, vocab_ext = os.path.splitext(vocab_path)
    merges_root, merges_ext = os.path.splitext(merges_path)

    resolved_vocab_path = f"{vocab_root}_{text_stem}{vocab_ext or '.json'}"
    resolved_merges_path = f"{merges_root}_{text_stem}{merges_ext or '.txt'}"
    return resolved_vocab_path, resolved_merges_path

def train_bpe_and_write_to_file(
    input_path: str,
    output_vocab_path: str,
    output_merges_path: str,
    vocab_size: int = 10000,
    special_tokens: list[str] = ["<|endoftext|>"]
):
    from implementation.bpe_advanced_impl import BPETrainer

    bpe_trainer = BPETrainer(input_path=input_path, vocab_size=vocab_size, special_tokens=special_tokens)
    vocab, merges = bpe_trainer.train()
    with open(output_vocab_path, "w", encoding="utf-8") as f:
        f.write(json.dumps({k: _b64(v) for k, v in vocab.items()}))
    with open(output_merges_path, "w", encoding="utf-8") as f:
        for b1, b2 in merges:
            f.write(f"{_b64(b1)} {_b64(b2)}\n")

def encode_text_input(
    vocab_path: str,
    merges_path: str,
    text_path: str,
    special_tokens: list[str] = ["<|endoftext|>"]
) -> list[int]:
    
    from implementation.bpe_cppyy import Tokenizer

    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
        vocab = {int(k): _unb64(v) for k, v in vocab.items()}
    merges = []
    with open(merges_path, "r", encoding="utf-8") as f:
        for line in f:
            cleaned = line.rstrip()
            if cleaned and len(cleaned.split(" ")) == 2:
                a, b = cleaned.split(" ")
                merges.append((_unb64(a), _unb64(b)))
    tokenizer = Tokenizer(vocab, merges, special_tokens=special_tokens)
    with open(text_path, "r", encoding="utf-8") as f:
        text = f.read()
    return tokenizer.encode(text)

def build_model_and_dataset(
    config: ModelConfig, 
    vocab_path: str,
    merges_path: str,
    text_path: str,
    vocab_size: int,
    special_tokens: list[str] = ["<|endoftext|>"]
) -> Tuple[Transformer, np.ndarray]:
    resolved_vocab_path, resolved_merges_path = _tokenizer_artifact_paths(
        vocab_path=vocab_path,
        merges_path=merges_path,
        text_path=text_path
    )

    if not os.path.exists(resolved_vocab_path) or not os.path.exists(resolved_merges_path):
        train_bpe_and_write_to_file(
            input_path=text_path,
            output_vocab_path=resolved_vocab_path,
            output_merges_path=resolved_merges_path,
            vocab_size=vocab_size,
            special_tokens=special_tokens
        )

    token_ids = encode_text_input(
        vocab_path=resolved_vocab_path,
        merges_path=resolved_merges_path,
        text_path=text_path,
        special_tokens=special_tokens
    )
    
    data_array = np.array(token_ids, dtype=np.int32)
    model = Transformer(**vars(config))
    return model, data_array

def build_optimizer(model: Transformer, config: OptimizerConfig):
    return AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
        betas=(config.beta1, config.beta2),
        eps=config.eps
    )

def train_step(
    model: Transformer,
    optimizer: AdamW,
    loss_func: CrossEntropyLoss,
    data_loader: DataLoader,
    max_grad_norm: float | None = None,
) -> Tuple[Tensor, Tensor]:
    model.train()
    x, y = data_loader.get_batch()
    y_pred = model(x).view(-1, model.lm_head.shape[0]) # (batch_size * seq_len, vocab_size)
    loss = loss_func(y_pred, y.reshape(-1))
    loss.backward()

    if max_grad_norm is not None:
        grad_norm: torch.Tensor = clip_gradients(model.parameters(), max_grad_norm)
    optimizer.step()
    optimizer.zero_grad()
    return loss.detach(), grad_norm

def train(config: TrainingConfig):
    model, dataset = build_model_and_dataset(config.model, config.vocab_path, config.merges_path, config.text_path, config.model.vocab_size)
    model.to(config.device)
    train_data = dataset[:int(len(dataset) * config.split_ratio)]
    val_data = dataset[int(len(dataset) * config.split_ratio):]
    optimizer = build_optimizer(model, config.optimizer)
    loss_func = CrossEntropyLoss()

    train_data_loader = DataLoader(train_data, config.batch_size, config.model.context_length, config.device)
    val_data_loader = DataLoader(val_data, config.batch_size, config.model.context_length, config.device)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    writer = SummaryWriter(str(os.path.join(current_dir, "runs", "exp_tinystories")))

    # losses = []
    # lrs = []
    # grad_norms = []

    # Training loop to be done
    for step in range(config.total_steps):
        # Sample a batch of data
        # Forward pass
        # Compute loss
        # Backward pass and optimization step
        # Logging and checkpointing
        lr = consine_lr_schedule(
            it=step,
            warmup_iters=config.warmup_steps,
            max_learning_rate=config.max_learning_rate,
            min_learning_rate=config.min_learning_rate,
            cosine_cycle_iters=config.consine_cycle_iters
        )
        # lrs.append(lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        train_loss, grad_norm = train_step(
            model=model,
            optimizer=optimizer,
            loss_func=loss_func,
            data_loader=train_data_loader,
            max_grad_norm=config.optimizer.max_grad_norm
        )
        # losses.append(train_loss.item())
        # grad_norms.append(grad_norm.item())

        writer.add_scalar("Train Loss", train_loss.item(), step)
        writer.add_scalar("Learning Rate", lr, step)
        writer.add_scalar("Grad Norm", grad_norm.item(), step)

        if step % config.log_every == 0:
            print(f"Step {step}: Train Loss = {train_loss.item():.4f}, LR = {lr:.6f}")
        
        if step % config.eval_every == 0:
            model.eval()
            val_loss = 0.0
            num_batches = 0
            with torch.no_grad():
                for _ in range(len(val_data_loader)):
                    x_val, y_val = val_data_loader.get_batch()
                    y_val_pred = model(x_val).view(-1, model.lm_head.shape[0])
                    val_loss += loss_func(y_val_pred, y_val.reshape(-1)).item()
                    num_batches += 1
            if num_batches > 0:
                avg_val_loss = val_loss / num_batches
                print(f"Step {step}: Validation Loss = {avg_val_loss:.4f}")
        
        if step % config.checkpoint_every == 0:
            save_checkpoint(model, optimizer, step, config.checkpoint_path)
    writer.close()

def generate_text(model: Transformer, tokenizer: Tokenizer, prompt: str, device: str, max_length: int = 100) -> str:
    from implementation.decoding import decode
    input_ids = tokenizer.encode(prompt)
    generate_token_ids = decode(
        model=model,
        prompt_token_ids=torch.tensor([input_ids], device=device),
        max_number_of_tokens=max_length,
        temperature=1.0,
        top_p_threshold=1.0,
        end_token_ids=None
    )[0].tolist()
    return tokenizer.decode(generate_token_ids)
    


if __name__ == "__main__":
    learning_rates_to_test = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]
    batch_size_to_test = [32, 64, 128, 256, 512, 1024]

    for lr in learning_rates_to_test:
        for batch_size in batch_size_to_test:
            print(f"\n{'='*40}")
            print(f"Starting sweep for max_learning_rate = {lr}, batch_size = {batch_size}")
            print(f"{'='*40}\n")
            config = TrainingConfig(
                model=ModelConfig(
                vocab_size=10000,
                context_length=256,
                d_model=512,
                num_layers=6,
                num_heads=8,
                d_ff=1344
            ),
            optimizer=OptimizerConfig(
                lr=3e-4,
                weight_decay=0.01,
                beta1=0.9,
                beta2=0.999,
                eps=1e-8,
                max_grad_norm=1.0
            ),
            vocab_path="vocab.json",
            merges_path="merges.txt",
            text_path= os.path.join("tests", "fixtures", "tinystories_sample_5M.txt"),
            split_ratio=0.9,
            # batch_size=3276800 // 256 // 3000, # 327680000 tokens in total, divided by context length and total steps
            batch_size=batch_size,
            total_steps=3000,
            warmup_steps= 3000 * 0.1,
            max_learning_rate=3e-4,
            min_learning_rate=3e-5,
            device="cuda" if torch.cuda.is_available() else "cpu",
            eval_every=500,
            log_every=50,
            checkpoint_every=1000,
            checkpoint_path=f"checkpoints/run_lr_{lr}.pt",
            run_name=f"lr_sweep_{lr}"
        )
        os.makedirs(os.path.dirname(config.checkpoint_path), exist_ok=True)
        train(config)
