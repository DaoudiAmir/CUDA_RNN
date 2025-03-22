import numpy as np
import time
import matplotlib.pyplot as plt
import os
import sys
import pandas as pd
from tabulate import tabulate

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from min_rnn.min_gru import MinGRUCell
from min_rnn.min_lstm import MinLSTMCell
from min_rnn.min_gru_cuda import MinGRUCellCUDA
from min_rnn.min_lstm_cuda import MinLSTMCellCUDA
from min_rnn.cuda_utils import CUDA_AVAILABLE

def generate_random_data(size, min_val=-1.0, max_val=1.0):
    """Generate random data.
    
    Args:
        size (int or tuple): Size of the array to generate
        min_val (float): Minimum value of the random data
        max_val (float): Maximum value of the random data
        
    Returns:
        numpy.ndarray: Random data
    """
    return min_val + (max_val - min_val) * np.random.random(size)

def benchmark_min_gru(input_size, hidden_size, seq_length, num_runs=5):
    """Benchmark MinGRU implementations.
    
    Args:
        input_size (int): Size of the input vector
        hidden_size (int): Size of the hidden state vector
        seq_length (int): Number of time steps
        num_runs (int): Number of runs to average over
        
    Returns:
        dict: Dictionary of benchmark results
    """
    # Initialize cells
    cpu_cell = MinGRUCell(input_size, hidden_size)
    
    # Generate random input data
    x = generate_random_data((seq_length, input_size))
    h0 = generate_random_data(hidden_size)
    
    # Benchmark CPU sequential mode
    cpu_seq_times = []
    for _ in range(num_runs):
        start_time = time.time()
        h_out_cpu_seq = cpu_cell.process_sequence(x, h0, seq_length)
        cpu_seq_times.append(time.time() - start_time)
    
    # Benchmark CPU parallel mode
    cpu_par_times = []
    for _ in range(num_runs):
        start_time = time.time()
        h_out_cpu_par = cpu_cell.process_sequence_parallel(x, h0, seq_length)
        cpu_par_times.append(time.time() - start_time)
    
    # Check if CUDA is available
    gpu_seq_times = []
    gpu_par_times = []
    if CUDA_AVAILABLE:
        # Initialize CUDA cell
        gpu_cell = MinGRUCellCUDA(input_size, hidden_size)
        
        # Benchmark GPU sequential mode
        for _ in range(num_runs):
            start_time = time.time()
            h_out_gpu_seq = gpu_cell.process_sequence(x, h0, seq_length)
            gpu_seq_times.append(time.time() - start_time)
        
        # Benchmark GPU parallel mode
        for _ in range(num_runs):
            start_time = time.time()
            h_out_gpu_par = gpu_cell.process_sequence_parallel(x, h0, seq_length)
            gpu_par_times.append(time.time() - start_time)
        
        # Check correctness
        max_diff_seq = np.max(np.abs(h_out_cpu_seq - h_out_gpu_seq))
        max_diff_par = np.max(np.abs(h_out_cpu_par - h_out_gpu_par))
        print(f"Maximum difference between CPU and GPU (sequential): {max_diff_seq:.8f}")
        print(f"Maximum difference between CPU and GPU (parallel): {max_diff_par:.8f}")
    
    # Calculate average times
    avg_cpu_seq_time = np.mean(cpu_seq_times)
    avg_cpu_par_time = np.mean(cpu_par_times)
    avg_gpu_seq_time = np.mean(gpu_seq_times) if gpu_seq_times else float('nan')
    avg_gpu_par_time = np.mean(gpu_par_times) if gpu_par_times else float('nan')
    
    # Calculate speedups
    cpu_par_speedup = avg_cpu_seq_time / avg_cpu_par_time
    gpu_seq_speedup = avg_cpu_seq_time / avg_gpu_seq_time if gpu_seq_times else float('nan')
    gpu_par_speedup = avg_cpu_seq_time / avg_gpu_par_time if gpu_par_times else float('nan')
    
    # Print results
    print(f"\n--- MinGRU Benchmark Results ---")
    print(f"Parameters: input_size={input_size}, hidden_size={hidden_size}, seq_length={seq_length}")
    print(f"CPU Sequential Time: {avg_cpu_seq_time:.6f} seconds")
    print(f"CPU Parallel Time: {avg_cpu_par_time:.6f} seconds (Speedup: {cpu_par_speedup:.2f}x)")
    if CUDA_AVAILABLE:
        print(f"GPU Sequential Time: {avg_gpu_seq_time:.6f} seconds (Speedup: {gpu_seq_speedup:.2f}x)")
        print(f"GPU Parallel Time: {avg_gpu_par_time:.6f} seconds (Speedup: {gpu_par_speedup:.2f}x)")
    
    # Return results
    return {
        "model": "MinGRU",
        "input_size": input_size,
        "hidden_size": hidden_size,
        "seq_length": seq_length,
        "cpu_sequential_time": avg_cpu_seq_time,
        "cpu_parallel_time": avg_cpu_par_time,
        "gpu_sequential_time": avg_gpu_seq_time,
        "gpu_parallel_time": avg_gpu_par_time,
        "cpu_parallel_speedup": cpu_par_speedup,
        "gpu_sequential_speedup": gpu_seq_speedup,
        "gpu_parallel_speedup": gpu_par_speedup
    }

def benchmark_min_lstm(input_size, hidden_size, seq_length, num_runs=5):
    """Benchmark MinLSTM implementations.
    
    Args:
        input_size (int): Size of the input vector
        hidden_size (int): Size of the hidden state vector
        seq_length (int): Number of time steps
        num_runs (int): Number of runs to average over
        
    Returns:
        dict: Dictionary of benchmark results
    """
    # Initialize cells
    cpu_cell = MinLSTMCell(input_size, hidden_size)
    
    # Generate random input data
    x = generate_random_data((seq_length, input_size))
    h0 = generate_random_data(hidden_size)
    
    # Benchmark CPU sequential mode
    cpu_seq_times = []
    for _ in range(num_runs):
        start_time = time.time()
        h_out_cpu_seq = cpu_cell.process_sequence(x, h0, seq_length)
        cpu_seq_times.append(time.time() - start_time)
    
    # Benchmark CPU parallel mode
    cpu_par_times = []
    for _ in range(num_runs):
        start_time = time.time()
        h_out_cpu_par = cpu_cell.process_sequence_parallel(x, h0, seq_length)
        cpu_par_times.append(time.time() - start_time)
    
    # Check if CUDA is available
    gpu_seq_times = []
    gpu_par_times = []
    if CUDA_AVAILABLE:
        # Initialize CUDA cell
        gpu_cell = MinLSTMCellCUDA(input_size, hidden_size)
        
        # Benchmark GPU sequential mode
        for _ in range(num_runs):
            start_time = time.time()
            h_out_gpu_seq = gpu_cell.process_sequence(x, h0, seq_length)
            gpu_seq_times.append(time.time() - start_time)
        
        # Benchmark GPU parallel mode
        for _ in range(num_runs):
            start_time = time.time()
            h_out_gpu_par = gpu_cell.process_sequence_parallel(x, h0, seq_length)
            gpu_par_times.append(time.time() - start_time)
        
        # Check correctness
        max_diff_seq = np.max(np.abs(h_out_cpu_seq - h_out_gpu_seq))
        max_diff_par = np.max(np.abs(h_out_cpu_par - h_out_gpu_par))
        print(f"Maximum difference between CPU and GPU (sequential): {max_diff_seq:.8f}")
        print(f"Maximum difference between CPU and GPU (parallel): {max_diff_par:.8f}")
    
    # Calculate average times
    avg_cpu_seq_time = np.mean(cpu_seq_times)
    avg_cpu_par_time = np.mean(cpu_par_times)
    avg_gpu_seq_time = np.mean(gpu_seq_times) if gpu_seq_times else float('nan')
    avg_gpu_par_time = np.mean(gpu_par_times) if gpu_par_times else float('nan')
    
    # Calculate speedups
    cpu_par_speedup = avg_cpu_seq_time / avg_cpu_par_time
    gpu_seq_speedup = avg_cpu_seq_time / avg_gpu_seq_time if gpu_seq_times else float('nan')
    gpu_par_speedup = avg_cpu_seq_time / avg_gpu_par_time if gpu_par_times else float('nan')
    
    # Print results
    print(f"\n--- MinLSTM Benchmark Results ---")
    print(f"Parameters: input_size={input_size}, hidden_size={hidden_size}, seq_length={seq_length}")
    print(f"CPU Sequential Time: {avg_cpu_seq_time:.6f} seconds")
    print(f"CPU Parallel Time: {avg_cpu_par_time:.6f} seconds (Speedup: {cpu_par_speedup:.2f}x)")
    if CUDA_AVAILABLE:
        print(f"GPU Sequential Time: {avg_gpu_seq_time:.6f} seconds (Speedup: {gpu_seq_speedup:.2f}x)")
        print(f"GPU Parallel Time: {avg_gpu_par_time:.6f} seconds (Speedup: {gpu_par_speedup:.2f}x)")
    
    # Return results
    return {
        "model": "MinLSTM",
        "input_size": input_size,
        "hidden_size": hidden_size,
        "seq_length": seq_length,
        "cpu_sequential_time": avg_cpu_seq_time,
        "cpu_parallel_time": avg_cpu_par_time,
        "gpu_sequential_time": avg_gpu_seq_time,
        "gpu_parallel_time": avg_gpu_par_time,
        "cpu_parallel_speedup": cpu_par_speedup,
        "gpu_sequential_speedup": gpu_seq_speedup,
        "gpu_parallel_speedup": gpu_par_speedup
    }

def plot_benchmark_results(results, output_dir="plots"):
    """Plot benchmark results.
    
    Args:
        results (list): List of benchmark result dictionaries
        output_dir (str): Directory to save plots to
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert results to DataFrame
    df = pd.DataFrame(results)
    
    # Group by model
    for model, group in df.groupby("model"):
        # Plot execution time vs sequence length
        plt.figure(figsize=(10, 6))
        plt.plot(group["seq_length"], group["cpu_sequential_time"], "o-", label="CPU Sequential")
        plt.plot(group["seq_length"], group["cpu_parallel_time"], "o-", label="CPU Parallel")
        if CUDA_AVAILABLE:
            plt.plot(group["seq_length"], group["gpu_sequential_time"], "o-", label="GPU Sequential")
            plt.plot(group["seq_length"], group["gpu_parallel_time"], "o-", label="GPU Parallel")
        plt.xlabel("Sequence Length")
        plt.ylabel("Execution Time (s)")
        plt.title(f"{model} Execution Time vs Sequence Length")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, f"{model.lower()}_time_vs_seq_length.png"))
        
        # Plot speedup vs sequence length
        plt.figure(figsize=(10, 6))
        plt.plot(group["seq_length"], group["cpu_parallel_speedup"], "o-", label="CPU Parallel")
        if CUDA_AVAILABLE:
            plt.plot(group["seq_length"], group["gpu_sequential_speedup"], "o-", label="GPU Sequential")
            plt.plot(group["seq_length"], group["gpu_parallel_speedup"], "o-", label="GPU Parallel")
        plt.xlabel("Sequence Length")
        plt.ylabel("Speedup (x)")
        plt.title(f"{model} Speedup vs Sequence Length")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, f"{model.lower()}_speedup_vs_seq_length.png"))
    
    # Plot comparison between MinGRU and MinLSTM
    min_gru_df = df[df["model"] == "MinGRU"]
    min_lstm_df = df[df["model"] == "MinLSTM"]
    
    if not min_gru_df.empty and not min_lstm_df.empty:
        # Plot execution time comparison
        plt.figure(figsize=(10, 6))
        plt.plot(min_gru_df["seq_length"], min_gru_df["cpu_sequential_time"], "o-", label="MinGRU CPU Sequential")
        plt.plot(min_lstm_df["seq_length"], min_lstm_df["cpu_sequential_time"], "o-", label="MinLSTM CPU Sequential")
        if CUDA_AVAILABLE:
            plt.plot(min_gru_df["seq_length"], min_gru_df["gpu_parallel_time"], "o-", label="MinGRU GPU Parallel")
            plt.plot(min_lstm_df["seq_length"], min_lstm_df["gpu_parallel_time"], "o-", label="MinLSTM GPU Parallel")
        plt.xlabel("Sequence Length")
        plt.ylabel("Execution Time (s)")
        plt.title("MinGRU vs MinLSTM Execution Time")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, "min_gru_vs_min_lstm_time.png"))
        
        # Plot speedup comparison
        plt.figure(figsize=(10, 6))
        plt.plot(min_gru_df["seq_length"], min_gru_df["cpu_parallel_speedup"], "o-", label="MinGRU CPU Parallel")
        plt.plot(min_lstm_df["seq_length"], min_lstm_df["cpu_parallel_speedup"], "o-", label="MinLSTM CPU Parallel")
        if CUDA_AVAILABLE:
            plt.plot(min_gru_df["seq_length"], min_gru_df["gpu_parallel_speedup"], "o-", label="MinGRU GPU Parallel")
            plt.plot(min_lstm_df["seq_length"], min_lstm_df["gpu_parallel_speedup"], "o-", label="MinLSTM GPU Parallel")
        plt.xlabel("Sequence Length")
        plt.ylabel("Speedup (x)")
        plt.title("MinGRU vs MinLSTM Speedup")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, "min_gru_vs_min_lstm_speedup.png"))

def run_benchmarks():
    """Run benchmarks for different configurations."""
    # Parameters
    input_size = 128
    hidden_size = 256
    seq_lengths = [10, 50, 100, 200, 500, 1000]
    num_runs = 3
    
    # Run benchmarks
    results = []
    
    for seq_length in seq_lengths:
        print(f"\nRunning benchmarks for sequence length {seq_length}...")
        
        # Benchmark MinGRU
        min_gru_result = benchmark_min_gru(input_size, hidden_size, seq_length, num_runs)
        results.append(min_gru_result)
        
        # Benchmark MinLSTM
        min_lstm_result = benchmark_min_lstm(input_size, hidden_size, seq_length, num_runs)
        results.append(min_lstm_result)
    
    # Plot results
    plot_benchmark_results(results)
    
    # Create a table of results
    table_data = []
    headers = ["Model", "Seq Length", "CPU Seq (s)", "CPU Par (s)", "GPU Seq (s)", "GPU Par (s)", 
               "CPU Par Speedup", "GPU Seq Speedup", "GPU Par Speedup"]
    
    for result in results:
        table_data.append([
            result["model"],
            result["seq_length"],
            f"{result['cpu_sequential_time']:.6f}",
            f"{result['cpu_parallel_time']:.6f}",
            f"{result['gpu_sequential_time']:.6f}" if not np.isnan(result['gpu_sequential_time']) else "N/A",
            f"{result['gpu_parallel_time']:.6f}" if not np.isnan(result['gpu_parallel_time']) else "N/A",
            f"{result['cpu_parallel_speedup']:.2f}x",
            f"{result['gpu_sequential_speedup']:.2f}x" if not np.isnan(result['gpu_sequential_speedup']) else "N/A",
            f"{result['gpu_parallel_speedup']:.2f}x" if not np.isnan(result['gpu_parallel_speedup']) else "N/A"
        ])
    
    print("\n--- Benchmark Results Summary ---")
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    
    # Save table to file
    with open("benchmark_results.txt", "w") as f:
        f.write(tabulate(table_data, headers=headers, tablefmt="grid"))

if __name__ == "__main__":
    run_benchmarks()
