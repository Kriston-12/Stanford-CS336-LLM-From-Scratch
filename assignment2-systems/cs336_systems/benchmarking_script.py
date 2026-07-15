from cs336_basics.data import get_batch
import cs336_basics.model as basics_model
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW
from cs336_systems.annotated_impl import scaled_dot_product_attention as annotated_scaled_dot_product_attention
import numpy.typing as npt
import argparse
import numpy as np
from timeit import default_timer as timer
import torch
from typing import Callable, TypeVar

T = TypeVar("T")
import os

def run_model_with_warmup(
    size: str,
    result_file: str,
    model: BasicsTransformerLM,
    data: npt.NDArray, 
    batch_size: int = 32, 
    seq_len: int = 128, 
    device: str = "cuda",
    warmup_steps: int = 5,
    cuda_profiler_api: bool = False,
    wall_time: bool = False,
    write_to_file: bool = False,
    benchmark_steps: int = 10,
):
    def timed_step(name: str, fn: Callable[[], T], times: list[float]) -> T:
        if not wall_time:
            return fn()

        if device == "cuda":
            torch.cuda.synchronize()

        timer_start = timer()
        result = fn()

        if device == "cuda":
            torch.cuda.synchronize()

        elapsed = timer() - timer_start
        times.append(elapsed)
        print(f"{name} time: {elapsed:.4f} seconds")
        return result

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
    if cuda_profiler_api:
        torch.cuda.profiler.start()

    forward_times = []
    backward_times = []
    optimizer_times = []
    torch.cuda.reset_peak_memory_stats()
    for _ in range(benchmark_steps):  # Run for benchmark_steps iterations
        inputs, targets = get_batch(data, batch_size, seq_len, device)
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = timed_step("Forward", lambda: model(inputs), forward_times)
        loss = cross_entropy(outputs.view(-1, outputs.size(-1)), targets.view(-1))
        
        timed_step("Backward", lambda: loss.backward(), backward_times)
        timed_step("Optimizer step", lambda: optimizer.step(), optimizer_times)

    torch.cuda.synchronize()
    peak_bytes = torch.cuda.max_memory_allocated()

    if cuda_profiler_api:
        torch.cuda.profiler.stop()
    
    if write_to_file:
        with open(result_file, "a") as f:
            f.write(f"Model size: {size}\n")
            f.write(f"Average forward time: {np.mean(forward_times):.4f} seconds\n")
            f.write(f"Average backward time: {np.mean(backward_times):.4f} seconds\n")
            f.write(f"Average optimizer step time: {np.mean(optimizer_times):.4f} seconds\n")
            f.write(f"Peak memory usage: {peak_bytes / (1024 ** 2):.2f} MB\n")
            f.write("\n\n")
    else:
        print(f"Model size: {size}")
        print(f"Average forward time: {np.mean(forward_times):.4f} seconds")
        print(f"Average backward time: {np.mean(backward_times):.4f} seconds")
        print(f"Average optimizer step time: {np.mean(optimizer_times):.4f} seconds")
        print(f"Peak memory usage: {peak_bytes / (1024 ** 2):.2f} MB")
        print("\n\n")


def default_model_configs() -> list[dict]:
    return [
        # Mostly based on GPT-2 configs; use context length 512 unless otherwise specified.
        {"name": "small", "d_model": 768, "d_ff": 3072, "num_layers": 12, "num_heads": 12, "context_length": 512},
        {"name": "medium", "d_model": 1024, "d_ff": 4096, "num_layers": 24, "num_heads": 16, "context_length": 512},
        {"name": "large", "d_model": 1280, "d_ff": 5120, "num_layers": 36, "num_heads": 20, "context_length": 512},
        {"name": "xl", "d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32, "context_length": 512},
        # {"name": "10B", "d_model": 4608, "d_ff": 12288, "num_layers": 50, "num_heads": 36, "context_length": 512},
    ]

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Benchmarking script for BasicsTransformerLM")
    argparser.add_argument("-d", "--data_path", type=str, help="Path to the dataset (text file)")
    argparser.add_argument("-b", "--batch_size", type=int, default=16, help="Batch size for training")
    argparser.add_argument("-s", "--seq_len", type=int, default=128, help="Sequence length for training")
    argparser.add_argument("-dev", "--device", type=str, default="cuda", help="Device to run the benchmark on (e.g., 'cuda' or 'cpu')")
    argparser.add_argument("-w", "--warmup_steps", type=int, default=2, help="Number of warmup steps")
    argparser.add_argument("--vocab_size", type=int, default=None, help="Vocabulary size. If omitted, inferred from data.")
    argparser.add_argument("--annotated_attention", action="store_true", help="Use the NVTX-annotated attention implementation.")
    argparser.add_argument("--cuda_profiler_api", action="store_true", help="Capture only the measured region when using Nsight Systems with cudaProfilerApi.")
    argparser.add_argument("--wall_time", action="store_true", help="Print Python wall-clock timings for forward, backward, and optimizer steps.")
    argparser.add_argument("--model_size", type=str, choices=["small", "medium", "large", "xl", "all"], default="all", help="Model size to benchmark.")
    args = argparser.parse_args()

    if args.annotated_attention:
        basics_model.scaled_dot_product_attention = annotated_scaled_dot_product_attention

    # dummy dataset for benchmarking purposes
    dataset = np.random.randint(0, 10000, size=(1000000,), dtype=np.int64)  # 1 million tokens
    vocab_size = args.vocab_size if args.vocab_size is not None else int(dataset.max()) + 1
    
    bench_mark_file = f"Warmup{args.warmup_steps}_Batch{args.batch_size}.txt"
    bench_mark_dir = os.path.join(os.path.dirname(__file__), "benchmark_results")
    os.makedirs(bench_mark_dir, exist_ok=True)

    for config in default_model_configs():
        if args.model_size != "all" and config["name"] != args.model_size:
            continue
        print(f"Testing model: {config['name']}")
        model = BasicsTransformerLM(
            vocab_size=vocab_size,
            d_model=config["d_model"],
            d_ff=config["d_ff"],
            num_layers=config["num_layers"],
            num_heads=config["num_heads"],
            context_length=config["context_length"]
        )
        run_model_with_warmup(
            size = config["name"],
            result_file = os.path.join(bench_mark_dir, bench_mark_file),
            model = model,
            data = dataset,
            batch_size = args.batch_size,
            seq_len = args.seq_len,
            device = args.device,
            warmup_steps = args.warmup_steps,
            cuda_profiler_api = args.cuda_profiler_api,
            wall_time = args.wall_time,
        )
