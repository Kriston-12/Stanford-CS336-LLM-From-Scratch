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
from typing import Tuple, Optional
from torch.utils.tensorboard import SummaryWriter

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

    # data / runtime
    split_ratio: float = 0.9
    device: str = "cuda"

    # token budget (fixed for fair comparison across runs)
    total_tokens: int = 327_680_000

    # per-step shape
    batch_size: int = 32
    total_steps: int = 0  # will be auto-filled if 0
    grad_accum_steps: int = 1

    # LR schedule
    max_learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    warmup_frac: float = 0.1  # warmup_steps = int(total_steps * warmup_frac)

    # logging / eval / ckpt
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

def _compute_steps_and_accum(
    *,
    total_tokens: int,
    context_length: int,
    batch_size: int,
    requested_steps: int,
    requested_grad_accum: int,
) -> tuple[int, int, int]:
    """
    Returns (total_steps, grad_accum_steps, achieved_total_tokens) while trying to:
    - keep total_tokens fixed
    - respect requested_steps if > 0, else infer steps
    - respect requested_grad_accum if > 0
    """
    assert batch_size > 0 and context_length > 0
    assert total_tokens > 0

    tokens_per_micro_step = batch_size * context_length
    if tokens_per_micro_step <= 0:
        raise ValueError("tokens_per_micro_step must be > 0")

    # If user provided total_steps, we solve for grad_accum (>=1).
    if requested_steps and requested_steps > 0:
        total_steps = int(requested_steps)
        denom = tokens_per_micro_step * total_steps
        grad_accum_steps = max(1, int(round(total_tokens / denom)))
        achieved = tokens_per_micro_step * total_steps * grad_accum_steps
        return total_steps, grad_accum_steps, achieved

    # Else infer total_steps given grad_accum (default 1)
    grad_accum_steps = max(1, int(requested_grad_accum))
    denom = tokens_per_micro_step * grad_accum_steps
    total_steps = max(1, int(round(total_tokens / denom)))
    achieved = tokens_per_micro_step * total_steps * grad_accum_steps
    return total_steps, grad_accum_steps, achieved

def train_step(
    model: Transformer,
    optimizer: AdamW,
    loss_func: CrossEntropyLoss,
    data_loader: DataLoader,
    *,
    max_grad_norm: float | None = None,
    grad_accum_steps: int = 1,
) -> Tuple[Tensor, Tensor]:
    """
    One *optimizer step* worth of work, implemented via grad accumulation:
    - do grad_accum_steps micro-batches
    - average the loss
    - (optional) clip grads
    - optimizer.step()

    Returns (loss, grad_norm).
    """
    model.train()
    optimizer.zero_grad(set_to_none=True)

    grad_accum_steps = max(1, int(grad_accum_steps))
    loss_total: Tensor | None = None

    for _ in range(grad_accum_steps):
        x, y = data_loader.get_batch()
        y_pred = model(x).view(-1, model.lm_head.shape[0])  # (B*T, V)
        loss = loss_func(y_pred, y.reshape(-1))
        loss = loss / grad_accum_steps  # average over micro-batches
        loss.backward()
        loss_total = loss.detach() if loss_total is None else (loss_total + loss.detach())

    if max_grad_norm is not None:
        grad_norm: torch.Tensor = clip_gradients(model.parameters(), max_grad_norm)
    else:
        grad_norm = torch.tensor(float("nan"), device=next(model.parameters()).device)

    optimizer.step()
    return loss_total, grad_norm

def train(config: TrainingConfig):
    # derive steps/accum to keep tokens fixed
    total_steps, grad_accum_steps, achieved_tokens = _compute_steps_and_accum(
        total_tokens=config.total_tokens,
        context_length=config.model.context_length,
        batch_size=config.batch_size,
        requested_steps=config.total_steps,
        requested_grad_accum=config.grad_accum_steps,
    )
    warmup_steps = max(1, int(total_steps * float(config.warmup_frac)))
    cosine_cycle_iters = max(1, total_steps - 2 * warmup_steps)

    # build model/dataset
    model, dataset = build_model_and_dataset(
        config.model,
        config.vocab_path,
        config.merges_path,
        config.text_path,
        config.model.vocab_size,
    )
    model.to(config.device)

    train_data = dataset[:int(len(dataset) * config.split_ratio)]
    val_data = dataset[int(len(dataset) * config.split_ratio):]

    # IMPORTANT: optimizer lr can be anything; we overwrite per-step anyway.
    optimizer = build_optimizer(model, config.optimizer)
    loss_func = CrossEntropyLoss()

    train_data_loader = DataLoader(train_data, config.batch_size, config.model.context_length, config.device)
    val_data_loader = DataLoader(val_data, config.batch_size, config.model.context_length, config.device)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    run_dir = os.path.join(current_dir, "runs", config.run_name)
    writer = SummaryWriter(run_dir)

    # Log run metadata for reproducibility
    writer.add_text("hparams/run_name", config.run_name, 0)
    writer.add_text("hparams/text_path", config.text_path, 0)
    writer.add_scalar("budget/total_tokens_target", float(config.total_tokens), 0)
    writer.add_scalar("budget/total_tokens_achieved", float(achieved_tokens), 0)
    writer.add_scalar("budget/batch_size", float(config.batch_size), 0)
    writer.add_scalar("budget/context_length", float(config.model.context_length), 0)
    writer.add_scalar("budget/total_steps", float(total_steps), 0)
    writer.add_scalar("budget/grad_accum_steps", float(grad_accum_steps), 0)

    print(
        f"[run={config.run_name}] target_tokens={config.total_tokens:,} "
        f"achieved_tokens={achieved_tokens:,} "
        f"(B={config.batch_size}, T={config.model.context_length}, "
        f"steps={total_steps}, accum={grad_accum_steps})"
    )

    for step in range(total_steps):
        lr = consine_lr_schedule(
            it=step,
            warmup_iters=warmup_steps,
            max_learning_rate=config.max_learning_rate,
            min_learning_rate=config.min_learning_rate,
            cosine_cycle_iters=cosine_cycle_iters,
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        train_loss, grad_norm = train_step(
            model=model,
            optimizer=optimizer,
            loss_func=loss_func,
            data_loader=train_data_loader,
            max_grad_norm=config.optimizer.max_grad_norm,
            grad_accum_steps=grad_accum_steps,
        )

        writer.add_scalar("Train/Loss", float(train_loss.item()), step)
        writer.add_scalar("Train/LearningRate", float(lr), step)
        writer.add_scalar("Train/GradNorm", float(grad_norm.item()) if torch.isfinite(grad_norm).all() else float("nan"), step)

        if step % config.log_every == 0:
            print(f"[{config.run_name}] step {step}/{total_steps} loss={train_loss.item():.4f} lr={lr:.6g}")

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
                writer.add_scalar("Val/Loss", float(avg_val_loss), step)
                print(f"[{config.run_name}] step {step}: val_loss={avg_val_loss:.4f}")

        if step % config.checkpoint_every == 0:
            os.makedirs(os.path.dirname(config.checkpoint_path), exist_ok=True)
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
    # Sweep design (log-scale is more standard than random)
    learning_rates_to_test = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
    batch_sizes_to_test = [32, 64, 128, 256]

    # Keep token budget fixed across runs
    TOTAL_TOKENS = 327_680_000

    for max_lr in learning_rates_to_test:
        for bs in batch_sizes_to_test:
            run_name = f"tinystories_lr{max_lr:g}_bs{bs}_tok{TOTAL_TOKENS}"

            config = TrainingConfig(
                model=ModelConfig(
                    vocab_size=10000,
                    context_length=256,
                    d_model=512,
                    num_layers=6,
                    num_heads=8,
                    d_ff=1344,
                ),
                optimizer=OptimizerConfig(
                    lr=max_lr,  # initial lr (schedule overwrites anyway)
                    weight_decay=0.01,
                    beta1=0.9,
                    beta2=0.999,
                    eps=1e-8,
                    max_grad_norm=1.0,
                ),
                vocab_path="vocab.json",
                merges_path="merges.txt",
                text_path=os.path.join("tests", "fixtures", "tinystories_sample_5M.txt"),
                split_ratio=0.9,
                device="cuda" if torch.cuda.is_available() else "cpu",
                total_tokens=TOTAL_TOKENS,
                batch_size=bs,
                total_steps=0,          # let code infer steps to match total_tokens
                grad_accum_steps=1,     # optionally set >1 if you want smaller micro-batches
                max_learning_rate=max_lr,
                min_learning_rate=max_lr / 10,
                warmup_frac=0.1,
                eval_every=500,
                log_every=50,
                checkpoint_every=1000,
                checkpoint_path=f"checkpoints/{run_name}.pt",
                run_name=run_name,
            )

            try:
                train(config)
            except RuntimeError as e:
                # if a run OOM/diverges, keep sweep going
                print(f"[FAILED] run={run_name} error={e}")
                continue
