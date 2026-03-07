"""
Summary of the cases in this file

This file compares pure Python list addition with several cppyy/C++ variants.

Key takeaway:
For this workload, cppyy is not faster than pure Python, because the dominant cost
is not the C++ loop itself. The main overhead comes from converting large Python
lists into C++ std::vector<int> objects across the Python/C++ boundary.

This benchmark is split into two parts:

1. End-to-end timing from Python lists
   - Measures Python list -> C++ std::vector conversion
   - Measures the C++ function body
   - Measures return-value wrapping back to Python

2. Compute-only timing on prebuilt std::vector<int>
   - Reuses C++ vectors across runs
   - Reduces Python/C++ conversion noise
   - Better isolates pass-by-value vs pass-by-reference and reserve() effects

Important note:
If a by-value version looks faster than a const-reference version in the
end-to-end benchmark, that does not mean pass-by-value is inherently better.
It more likely means the binding/conversion path dominates the timing.
"""

import time
import cppyy


def add_vecs(v1, v2):
    result = []
    n = min(len(v1), len(v2))
    for i in range(n):
        result.append(v1[i] + v2[i])
    return result


cppyy.cppdef("""
#include <vector>
#include <algorithm>

std::vector<int> add_vecs_ref_noreserve(const std::vector<int>& v1, const std::vector<int>& v2) {
    int n = std::min(v1.size(), v2.size());
    std::vector<int> result;
    for (int i = 0; i < n; ++i) {
        result.push_back(v1[i] + v2[i]);
    }
    return result;
}

std::vector<int> add_vecs_ref_reserve(const std::vector<int>& v1, const std::vector<int>& v2) {
    int n = std::min(v1.size(), v2.size());
    std::vector<int> result;
    result.reserve(n);
    for (int i = 0; i < n; ++i) {
        result.push_back(v1[i] + v2[i]);
    }
    return result;
}

std::vector<int> add_vecs_value_noreserve(std::vector<int> v1, std::vector<int> v2) {
    int n = std::min(v1.size(), v2.size());
    std::vector<int> result;
    for (int i = 0; i < n; ++i) {
        result.push_back(v1[i] + v2[i]);
    }
    return result;
}

std::vector<int> add_vecs_value_reserve(std::vector<int> v1, std::vector<int> v2) {
    int n = std::min(v1.size(), v2.size());
    std::vector<int> result;
    result.reserve(n);
    for (int i = 0; i < n; ++i) {
        result.push_back(v1[i] + v2[i]);
    }
    return result;
}

std::vector<int> add_vecs_ref_indexed(const std::vector<int>& v1, const std::vector<int>& v2) {
    int n = std::min(v1.size(), v2.size());
    std::vector<int> result(n);
    for (int i = 0; i < n; ++i) {
        result[i] = v1[i] + v2[i];
    }
    return result;
}
""")


def bench(name, fn, *args, repeat=5):
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        result = fn(*args)
        t1 = time.perf_counter()
        _ = len(result)
        times.append(t1 - t0)

    avg = sum(times) / len(times)
    print(f"{name}")
    print(f"  times: {[round(x, 4) for x in times]}")
    print(f"  avg:   {avg:.4f}s")
    print(f"  min:   {min(times):.4f}s")
    print()


def main():
    n = 10_000_000
    vec1 = [i for i in range(n)]
    vec2 = [i for i in range(n)]

    print(f"Input size: {n}")
    print()

    print("=== End-to-end benchmark from Python lists ===")
    bench("pure Python", add_vecs, vec1, vec2)
    bench("cppyy ref noreserve", cppyy.gbl.add_vecs_ref_noreserve, vec1, vec2)
    bench("cppyy ref reserve", cppyy.gbl.add_vecs_ref_reserve, vec1, vec2)
    bench("cppyy value noreserve", cppyy.gbl.add_vecs_value_noreserve, vec1, vec2)
    bench("cppyy value reserve", cppyy.gbl.add_vecs_value_reserve, vec1, vec2)

    print("=== Measure Python list -> std::vector conversion only ===")
    conversion_times = []
    for _ in range(5):
        t0 = time.perf_counter()
        v1_cpp = cppyy.gbl.std.vector[int](vec1)
        v2_cpp = cppyy.gbl.std.vector[int](vec2)
        t1 = time.perf_counter()
        conversion_times.append(t1 - t0)
        _ = len(v1_cpp), len(v2_cpp)

    print("build std::vector<int> from Python lists")
    print(f"  times: {[x for x in conversion_times]}")
    print(f"  avg:   {sum(conversion_times) / len(conversion_times):.4f}s")
    print(f"  min:   {min(conversion_times):.4f}s")
    print()

    print("=== Compute-only benchmark on prebuilt std::vector<int> ===")
    v1_cpp = cppyy.gbl.std.vector[int](vec1)
    v2_cpp = cppyy.gbl.std.vector[int](vec2)
    
    # Everything is much faster now that we're not measuring Python->C++ conversion.
    # Pass-by-reference is faster than pass-by-value, and reserve() is faster than no-reserve, as expected.
    bench("cppyy ref noreserve (prebuilt vectors)", cppyy.gbl.add_vecs_ref_noreserve, v1_cpp, v2_cpp)
    bench("cppyy ref reserve (prebuilt vectors)", cppyy.gbl.add_vecs_ref_reserve, v1_cpp, v2_cpp)
    bench("cppyy value noreserve (prebuilt vectors)", cppyy.gbl.add_vecs_value_noreserve, v1_cpp, v2_cpp)
    bench("cppyy value reserve (prebuilt vectors)", cppyy.gbl.add_vecs_value_reserve, v1_cpp, v2_cpp)
    bench("cppyy ref indexed (prebuilt vectors)", cppyy.gbl.add_vecs_ref_indexed, v1_cpp, v2_cpp)


if __name__ == "__main__":
    main()