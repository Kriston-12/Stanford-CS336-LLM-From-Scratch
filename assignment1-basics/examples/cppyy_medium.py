"""
BPE-like benchmark: pair counting on token sequences.

Goal:
- Show a realistic hotspot from BPE training.
- Compare:
    1. pure Python single-thread
    2. Python multiprocessing
    3. C++ single-thread via cppyy
    4. C++ multithread via cppyy

Why this is more realistic than a synthetic arithmetic kernel:
- BPE repeatedly scans token sequences.
- A core hotspot is counting adjacent token pairs.
- This benchmark focuses on that exact pattern.

Design:
- Generate one large token sequence once in Python.
- Convert it once to std::vector<int>.
- Count adjacent pairs and return the most frequent pair plus its count.
- For parallel runs:
    * Python multiprocessing splits the token list into chunks with 1-token overlap.
    * C++ multithreading splits the std::vector with the same overlap idea.

Important note:
This is still a reduced benchmark, not a full BPE trainer. The point is to isolate
a real BPE hotspot and compare Python vs cppyy/C++ fairly.
"""

import time
import random
import multiprocessing as mp
import cppyy


cppyy.cppdef(r"""
#include <vector>
#include <unordered_map>
#include <thread>
#include <cstdint>
#include <algorithm>
#include <utility>

static inline uint64_t pack_pair(uint32_t a, uint32_t b) {
    return (static_cast<uint64_t>(a) << 32) | static_cast<uint64_t>(b);
}

std::pair<uint64_t, uint64_t> count_pairs_cpp_single(const std::vector<int>& tokens) {
    std::unordered_map<uint64_t, uint64_t> counts;
    if (tokens.size() < 2) {
        return {0, 0};
    }

    counts.reserve(tokens.size());

    for (size_t i = 0; i + 1 < tokens.size(); ++i) {
        uint64_t key = pack_pair(
            static_cast<uint32_t>(tokens[i]),
            static_cast<uint32_t>(tokens[i + 1])
        );
        ++counts[key];
    }

    uint64_t best_key = 0;
    uint64_t best_count = 0;
    for (const auto& kv : counts) {
        if (kv.second > best_count) {
            best_key = kv.first;
            best_count = kv.second;
        }
    }
    return {best_key, best_count};
}

std::unordered_map<uint64_t, uint64_t> count_pairs_cpp_range(
    const std::vector<int>& tokens,
    size_t begin,
    size_t end
) {
    std::unordered_map<uint64_t, uint64_t> counts;
    if (end <= begin + 1) {
        return counts;
    }

    counts.reserve(end - begin);

    for (size_t i = begin; i + 1 < end; ++i) {
        uint64_t key = pack_pair(
            static_cast<uint32_t>(tokens[i]),
            static_cast<uint32_t>(tokens[i + 1])
        );
        ++counts[key];
    }
    return counts;
}

std::pair<uint64_t, uint64_t> count_pairs_cpp_mt(const std::vector<int>& tokens, int num_threads) {
    if (tokens.size() < 2) {
        return {0, 0};
    }
    if (num_threads <= 1) {
        return count_pairs_cpp_single(tokens);
    }

    size_t n = tokens.size();
    if (static_cast<size_t>(num_threads) > n - 1) {
        num_threads = static_cast<int>(n - 1);
    }

    std::vector<std::unordered_map<uint64_t, uint64_t>> partials(num_threads);
    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    size_t chunk = (n + num_threads - 1) / num_threads;

    for (int t = 0; t < num_threads; ++t) {
        size_t begin = static_cast<size_t>(t) * chunk;
        size_t end = std::min(begin + chunk + 1, n);

        threads.emplace_back([&tokens, &partials, begin, end, t]() {
            auto local = count_pairs_cpp_range(tokens, begin, end);
            partials[t] = std::move(local);
        });
    }

    for (auto& th : threads) {
        th.join();
    }

    std::unordered_map<uint64_t, uint64_t> merged;
    merged.reserve(tokens.size());

    for (const auto& local : partials) {
        for (const auto& kv : local) {
            merged[kv.first] += kv.second;
        }
    }

    uint64_t best_key = 0;
    uint64_t best_count = 0;
    for (const auto& kv : merged) {
        if (kv.second > best_count) {
            best_key = kv.first;
            best_count = kv.second;
        }
    }
    return {best_key, best_count};
}
""")


def pack_pair_py(a: int, b: int) -> int:
    return ((a & 0xFFFFFFFF) << 32) | (b & 0xFFFFFFFF)


def unpack_pair_py(key: int) -> tuple[int, int]:
    return (key >> 32) & 0xFFFFFFFF, key & 0xFFFFFFFF


def count_pairs_py_single(tokens: list[int]) -> tuple[int, int]:
    counts = {}
    n = len(tokens)
    for i in range(n - 1):
        key = pack_pair_py(tokens[i], tokens[i + 1])
        counts[key] = counts.get(key, 0) + 1

    best_key = 0
    best_count = 0
    for k, v in counts.items():
        if v > best_count:
            best_key = k
            best_count = v
    return best_key, best_count


def _count_pairs_py_chunk(tokens: list[int]) -> dict[int, int]:
    counts = {}
    n = len(tokens)
    for i in range(n - 1):
        key = pack_pair_py(tokens[i], tokens[i + 1])
        counts[key] = counts.get(key, 0) + 1
    return counts


def count_pairs_py_multiprocess(tokens: list[int], num_workers: int) -> tuple[int, int]:
    n = len(tokens)
    chunk = (n + num_workers - 1) // num_workers

    pieces = []
    for start in range(0, n, chunk):
        end = min(start + chunk + 1, n)
        piece = tokens[start:end]
        if len(piece) >= 2:
            pieces.append(piece)

    with mp.Pool(processes=num_workers) as pool:
        partials = pool.map(_count_pairs_py_chunk, pieces)

    merged = {}
    for local in partials:
        for k, v in local.items():
            merged[k] = merged.get(k, 0) + v

    best_key = 0
    best_count = 0
    for k, v in merged.items():
        if v > best_count:
            best_key = k
            best_count = v
    return best_key, best_count


def bench(name, fn, *args, repeat=3):
    times = []
    out0 = None
    for _ in range(repeat):
        t0 = time.perf_counter()
        out = fn(*args)
        t1 = time.perf_counter()
        times.append(t1 - t0)
        if out0 is None:
            out0 = out

    pair = unpack_pair_py(int(out0[0]))
    count = int(out0[1])

    print(name)
    print(f"  best_pair: {pair}")
    print(f"  best_count:{count}")
    print(f"  times:     {[round(x, 4) for x in times]}")
    print(f"  avg:       {sum(times) / len(times):.4f}s")
    print(f"  min:       {min(times):.4f}s")
    print()


def main():
    n = 8_000_000
    vocab_size = 512
    num_workers = max(2, min(8, mp.cpu_count()))

    rng = random.Random(0)
    tokens = [rng.randrange(vocab_size) for _ in range(n)]

    print(f"Num tokens:   {n}")
    print(f"Vocab size:   {vocab_size}")
    print(f"CPU workers:  {num_workers}")
    print()

    print("=== Build std::vector<int> once ===")
    t0 = time.perf_counter()
    tokens_cpp = cppyy.gbl.std.vector[int](tokens)
    t1 = time.perf_counter()
    print(f"build std::vector<int>: {t1 - t0:.4f}s")
    print()

    print("=== Pair counting benchmark ===")
    bench("pure Python single-thread", count_pairs_py_single, tokens)
    bench("Python multiprocessing", count_pairs_py_multiprocess, tokens, num_workers)
    bench("C++ single-thread via cppyy", cppyy.gbl.count_pairs_cpp_single, tokens_cpp)
    bench("C++ multithread via cppyy", cppyy.gbl.count_pairs_cpp_mt, tokens_cpp, num_workers)


if __name__ == "__main__":
    main()