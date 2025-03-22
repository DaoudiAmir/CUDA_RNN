import numpy as np
import time
import argparse
import os
import sys

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from min_rnn.min_gru import MinGRUCell
from min_rnn.min_lstm import MinLSTMCell
from min_rnn.min_gru_cuda import MinGRUCellCUDA
from min_rnn.min_lstm_cuda import MinLSTMCellCUDA
from min_rnn.cuda_utils import CUDA_AVAILABLE

def generate_random_data(size, min_val=-1.0, max_val=1.0):
    """Generate random data."""
    return min_val + (max_val - min_val) * np.random.random(size)

def benchmark_model(model_type, use_cuda, input_size, hidden_size, seq_length, num_runs=5):
    """Benchmark a specific model configuration."""
    print(f"\n--- Benchmarking {model_type} {'with CUDA' if use_cuda else 'on CPU'} ---")
    print(f"Parameters: input_size={input_size}, hidden_size={hidden_size}, seq_length={seq_length}")
    
    # Initialize cell
    if model_type == "gru":
        if use_cuda and CUDA_AVAILABLE:
            cell = MinGRUCellCUDA(input_size, hidden_size)
        else:
            cell = MinGRUCell(input_size, hidden_size)
    else:  # lstm
        if use_cuda and CUDA_AVAILABLE:
            cell = MinLSTMCellCUDA(input_size, hidden_size)
        else:
            cell = MinLSTMCell(input_size, hidden_size)
    
    # Generate random input data
    x = generate_random_data((seq_length, input_size))
    h0 = generate_random_data(hidden_size)
    
    # Warm-up run
    _ = cell.process_sequence(x, h0, seq_length)
    
    # Benchmark sequential mode
    seq_times = []
    for _ in range(num_runs):
        start_time = time.time()
        h_out_seq = cell.process_sequence(x, h0, seq_length)
        seq_times.append(time.time() - start_time)
    
    # Benchmark parallel mode
    par_times = []
    for _ in range(num_runs):
        start_time = time.time()
        h_out_par = cell.process_sequence_parallel(x, h0, seq_length)
        par_times.append(time.time() - start_time)
    
    # Calculate average times
    avg_seq_time = np.mean(seq_times)
    avg_par_time = np.mean(par_times)
    
    # Calculate speedup
    par_speedup = avg_seq_time / avg_par_time if avg_par_time > 0 else float('nan')
    
    # Print results
    print(f"Sequential Time: {avg_seq_time:.6f} seconds")
    print(f"Parallel Time: {avg_par_time:.6f} seconds")
    print(f"Parallel Speedup: {par_speedup:.2f}x")
    
    # Check numerical differences
    max_diff = np.max(np.abs(h_out_seq - h_out_par))
    print(f"Maximum difference between sequential and parallel: {max_diff:.8f}")
    
    return {
        "model": model_type,
        "cuda": use_cuda,
        "input_size": input_size,
        "hidden_size": hidden_size,
        "seq_length": seq_length,
        "sequential_time": avg_seq_time,
        "parallel_time": avg_par_time,
        "parallel_speedup": par_speedup,
        "max_diff": max_diff
    }

def main():
    parser = argparse.ArgumentParser(description="Large-scale RNN Benchmark")
    parser.add_argument("--model", type=str, default="gru", choices=["gru", "lstm"],
                        help="Model to benchmark (gru or lstm)")
    parser.add_argument("--cuda", action="store_true", help="Use CUDA acceleration")
    parser.add_argument("--input-size", type=int, default=512, help="Input size")
    parser.add_argument("--hidden-size", type=int, default=1024, help="Hidden size")
    parser.add_argument("--seq-length", type=int, default=2000, help="Sequence length")
    parser.add_argument("--num-runs", type=int, default=3, help="Number of runs to average over")
    args = parser.parse_args()
    
    # Print CUDA availability
    if args.cuda:
        if CUDA_AVAILABLE:
            print("CUDA is available and will be used for acceleration")
        else:
            print("CUDA is not available, falling back to CPU implementation")
            args.cuda = False
    
    # Run benchmark
    result = benchmark_model(
        args.model, args.cuda, args.input_size, args.hidden_size, args.seq_length, args.num_runs
    )
    
    # Print summary
    print("\n--- Benchmark Summary ---")
    for key, value in result.items():
        if isinstance(value, float):
            print(f"{key}: {value:.6f}")
        else:
            print(f"{key}: {value}")

if __name__ == "__main__":
    main()
