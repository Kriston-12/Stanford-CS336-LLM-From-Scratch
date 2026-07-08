from cs336_basics.data import get_batch
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW
import numpy.typing as npt
import argparse
import numpy as np
from timeit import default_timer as timer
import torch

def run_model_with_warmup(
    model: BasicsTransformerLM,
    data: npt.NDArray, 
    batch_size: int = 32, 
    seq_len: int = 128, 
    device: str = "cuda",
    warmup_steps: int = 5,
):
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=1e-4)

    for _ in range(warmup_steps):
        inputs, targets = get_batch(data, batch_size, seq_len, device)
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = cross_entropy(outputs.view(-1, outputs.size(-1)), targets.view(-1))
        loss.backward()
        optimizer.step()

    torch.cuda.synchronize()
    forward_times = []
    backward_times = []
    optimizer_times = []
    for _ in range(100):  # Run for 100 iterations
        inputs, targets = get_batch(data, batch_size, seq_len, device)
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        timer_start = timer()
        outputs = model(inputs)
        timer_end = timer()
        forward_times.append(timer_end - timer_start)
        torch.cuda.synchronize()  
        print(f"Forward time: {forward_times[-1]:.4f} seconds")
        loss = cross_entropy(outputs.view(-1, outputs.size(-1)), targets.view(-1))
        
        timer_start = timer()
        loss.backward()
        timer_end = timer()
        backward_times.append(timer_end - timer_start)
        print(f"Backward time: {backward_times[-1]:.4f} seconds")
        torch.cuda.synchronize()  
        timer_start = timer()
        optimizer.step()
        timer_end = timer()
        optimizer_times.append(timer_end - timer_start)
        torch.cuda.synchronize()  
        print(f"Optimizer step time: {optimizer_times[-1]:.4f} seconds")

    print(f"Average forward time: {np.mean(forward_times):.4f} seconds")
    print(f"Average backward time: {np.mean(backward_times):.4f} seconds")
    print(f"Average optimizer step time: {np.mean(optimizer_times):.4f} seconds")


def default_model_configs() -> list[dict]:
    return [
        # Mostly based on GPT-2 configs; use context length 512 unless otherwise specified.
        {"name": "small", "d_model": 768, "d_ff": 3072, "num_layers": 12, "num_heads": 12, "context_length": 512},
        {"name": "medium", "d_model": 1024, "d_ff": 4096, "num_layers": 24, "num_heads": 16, "context_length": 512},
        {"name": "large", "d_model": 1280, "d_ff": 5120, "num_layers": 36, "num_heads": 20, "context_length": 512},
        {"name": "xl", "d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32, "context_length": 512},
        {"name": "10B", "d_model": 4608, "d_ff": 12288, "num_layers": 50, "num_heads": 36, "context_length": 512},
    ]

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Benchmarking script for BasicsTransformerLM")
    argparser.add_argument("-d", "--data_path", type=str, help="Path to the dataset (text file)")
    argparser.add_argument("-b", "--batch_size", type=int, default=32, help="Batch size for training")
    argparser.add_argument("-s", "--seq_len", type=int, default=128, help="Sequence length for training")
    argparser.add_argument("-dev", "--device", type=str, default="cuda", help="Device to run the benchmark on (e.g., 'cuda' or 'cpu')")
    argparser.add_argument("-w", "--warmup_steps", type=int, default=5, help="Number of warmup steps")
    argparser.add_argument("--vocab_size", type=int, default=None, help="Vocabulary size. If omitted, inferred from data.")
    args = argparser.parse_args()

    # dummy dataset for benchmarking purposes
    dataset = np.random.randint(0, 10000, size=(1000000,), dtype=np.int64)  # 1 million tokens

    vocab_size = args.vocab_size if args.vocab_size is not None else int(dataset.max()) + 1

    for config in default_model_configs():
        print(f"Testing model: {config['name']}")
        model = BasicsTransformerLM(
            vocab_size=vocab_size,
            d_model=config["d_model"],
            d_ff=config["d_ff"],
            num_layers=config["num_layers"],
            num_heads=config["num_heads"],
            context_length=config["context_length"]
        )
        run_model_with_warmup(model, dataset, args.batch_size, args.seq_len, args.device, args.warmup_steps)