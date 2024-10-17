import torch
import torch.nn.functional as F
import time
import math

def benchmark_attention(batch_size, seq_len, num_heads, head_dim, num_runs=10):
    assert torch.cuda.is_available(), "CUDA is not available"
    device = torch.device("cuda")
    dtype = torch.float16  # Use float16 for better performance on GPUs

    # Create input tensors
    q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
    v = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)

    # Benchmark PyTorch attention
    def run_pytorch_attention():
        scale_factor = 1 / math.sqrt(q.size(-1))
        attn_weight = q @ k.transpose(-2, -1) * scale_factor
        attn_weight = torch.softmax(attn_weight, dim=-1)
        return attn_weight @ v

    # Benchmark FlashAttention 2
    def run_flash_attention():
        return F.scaled_dot_product_attention(q, k, v, is_causal=False)

    # Warm-up runs
    for _ in range(5):
        run_pytorch_attention()
        run_flash_attention()

    # Benchmark PyTorch attention
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(num_runs):
        out_pytorch = run_pytorch_attention()
    torch.cuda.synchronize()
    pytorch_time = (time.time() - start_time) / num_runs

    # Benchmark FlashAttention
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(num_runs):
        out_flash = run_flash_attention()
    torch.cuda.synchronize()
    flash_time = (time.time() - start_time) / num_runs

    # Calculate memory usage
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    run_pytorch_attention()
    pytorch_memory = torch.cuda.max_memory_allocated() / 1e6  # Convert to MB

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    run_flash_attention()
    flash_memory = torch.cuda.max_memory_allocated() / 1e6  # Convert to MB

    return {
        "pytorch_time": pytorch_time,
        "flash_time": flash_time,
        "pytorch_memory": pytorch_memory,
        "flash_memory": flash_memory,
        "speedup": pytorch_time / flash_time,
        "memory_reduction": pytorch_memory / flash_memory
    }

# Run benchmarks for different sequence lengths
seq_lengths = [512, 1024, 2048, 4096, 8192, 16384]
batch_size = 16
num_heads = 12
head_dim = 64

results = []

for seq_len in seq_lengths:
    print(f"Benchmarking sequence length: {seq_len}")
    result = benchmark_attention(batch_size, seq_len, num_heads, head_dim)
    results.append((seq_len, result))
    print(f"  PyTorch time: {result['pytorch_time']:.4f} s")
    print(f"  FlashAttention time: {result['flash_time']:.4f} s")
    print(f"  Speedup: {result['speedup']:.2f}x")
    print(f"  PyTorch memory: {result['pytorch_memory']:.2f} MB")
    print(f"  FlashAttention memory: {result['flash_memory']:.2f} MB")
    print(f"  Memory reduction: {result['memory_reduction']:.2f}x")
    print()

# Plot results
import matplotlib.pyplot as plt

seq_lengths, speedups, memory_reductions = zip(*[(r[0], r[1]['speedup'], r[1]['memory_reduction']) for r in results])

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(seq_lengths, speedups, marker='o')
plt.title('FlashAttention Speedup')
plt.xlabel('Sequence Length')
plt.ylabel('Speedup (x)')
plt.xscale('log')

plt.subplot(1, 2, 2)
plt.plot(seq_lengths, memory_reductions, marker='o')
plt.title('FlashAttention Memory Reduction')
plt.xlabel('Sequence Length')
plt.ylabel('Memory Reduction (x)')
plt.xscale('log')

plt.tight_layout()
plt.savefig('attention_benchmark_results.png')
plt.show()