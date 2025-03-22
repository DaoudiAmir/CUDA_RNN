"""
Benchmarking script to measure the performance improvements of optimized CUDA RNN implementations.

This script compares the execution time of CPU vs GPU implementations for both MinGRU and MinLSTM
with various sequence lengths and batch sizes to determine the effectiveness of the optimizations.
"""

import numpy as np
import time
import matplotlib.pyplot as plt
import os
import argparse
from numba import cuda

from min_rnn.min_gru import MinGRUCell
from min_rnn.min_gru_cuda import MinGRUCellCUDA
from min_rnn.min_lstm import MinLSTMCell
from min_rnn.min_lstm_cuda import MinLSTMCellCUDA
from min_rnn.cuda_utils import CUDA_AVAILABLE


def run_benchmark(model_type='gru', input_size=64, hidden_size=128, 
                 seq_lengths=[8, 16, 32, 64, 128, 256, 512, 1024], 
                 num_runs=5, cuda_device=0):
    """
    Run benchmark comparing CPU and GPU implementations.
    
    Args:
        model_type (str): Type of RNN model ('gru' or 'lstm')
        input_size (int): Size of input vector
        hidden_size (int): Size of hidden state vector
        seq_lengths (list): List of sequence lengths to test
        num_runs (int): Number of runs to average for each test case
        cuda_device (int): CUDA device to use
        
    Returns:
        dict: Dictionary containing benchmark results
    """
    if CUDA_AVAILABLE:
        cuda.select_device(cuda_device)
        print(f"Running on CUDA device: {cuda.get_current_device().name}")
    else:
        print("CUDA not available, running only CPU benchmarks")
        
    # Initialize models
    if model_type.lower() == 'gru':
        cpu_model = MinGRUCell(input_size, hidden_size)
        if CUDA_AVAILABLE:
            gpu_model = MinGRUCellCUDA(input_size, hidden_size)
            
        print(f"Benchmarking MinGRU models with input_size={input_size}, hidden_size={hidden_size}")
    else:
        cpu_model = MinLSTMCell(input_size, hidden_size)
        if CUDA_AVAILABLE:
            gpu_model = MinLSTMCellCUDA(input_size, hidden_size)
            
        print(f"Benchmarking MinLSTM models with input_size={input_size}, hidden_size={hidden_size}")
    
    results = {
        'seq_length': [],
        'cpu_time': [],
        'gpu_time': [],
        'speedup': []
    }
    
    # Run benchmarks for each sequence length
    for seq_length in seq_lengths:
        print(f"\nBenchmarking sequence length: {seq_length}")
        
        # Generate random input data
        x_seq = np.random.randn(seq_length, input_size).astype(np.float32)
        h0 = np.random.randn(hidden_size).astype(np.float32)
        
        # Benchmark CPU implementation
        cpu_times = []
        for _ in range(num_runs):
            start_time = time.time()
            
            if model_type.lower() == 'gru':
                # Use parallel processing for long sequences
                if seq_length > 64:
                    h_cpu = cpu_model.process_sequence_parallel(x_seq, h0)
                else:
                    h_cpu = cpu_model.process_sequence(x_seq, h0)
            else:
                # Use parallel processing for long sequences
                if seq_length > 64:
                    h_cpu = cpu_model.process_sequence_parallel(x_seq, h0)
                else:
                    h_cpu = cpu_model.process_sequence(x_seq, h0)
                    
            end_time = time.time()
            cpu_times.append(end_time - start_time)
        
        avg_cpu_time = sum(cpu_times) / len(cpu_times)
        print(f"CPU average time: {avg_cpu_time:.6f} seconds")
        
        # Benchmark GPU implementation if available
        if CUDA_AVAILABLE:
            # Warmup run to initialize CUDA
            if model_type.lower() == 'gru':
                h_gpu = gpu_model.process_sequence(x_seq, h0)
            else:
                h_gpu = gpu_model.process_sequence(x_seq, h0)
                
            # Force synchronization
            cuda.synchronize()
            
            gpu_times = []
            for _ in range(num_runs):
                start_time = time.time()
                
                if model_type.lower() == 'gru':
                    h_gpu = gpu_model.process_sequence(x_seq, h0)
                else:
                    h_gpu = gpu_model.process_sequence(x_seq, h0)
                    
                # Force synchronization to ensure accurate timing
                cuda.synchronize()
                
                end_time = time.time()
                gpu_times.append(end_time - start_time)
            
            avg_gpu_time = sum(gpu_times) / len(gpu_times)
            print(f"GPU average time: {avg_gpu_time:.6f} seconds")
            
            # Calculate speedup
            speedup = avg_cpu_time / avg_gpu_time if avg_gpu_time > 0 else 0
            print(f"Speedup (CPU/GPU): {speedup:.2f}x")
            
            # Verify results match (within floating point tolerance)
            if model_type.lower() == 'gru':
                h_cpu = cpu_model.process_sequence(x_seq, h0)
                h_gpu = gpu_model.process_sequence(x_seq, h0)
            else:
                h_cpu = cpu_model.process_sequence(x_seq, h0)
                h_gpu = gpu_model.process_sequence(x_seq, h0)
                
            error = np.mean(np.abs(h_cpu - h_gpu))
            print(f"Mean absolute error: {error:.6f}")
            
            results['gpu_time'].append(avg_gpu_time)
            results['speedup'].append(speedup)
        else:
            results['gpu_time'].append(0)
            results['speedup'].append(0)
            
        results['seq_length'].append(seq_length)
        results['cpu_time'].append(avg_cpu_time)
    
    return results


def plot_results(results, model_type, output_dir="./benchmark_results"):
    """
    Plot benchmark results.
    
    Args:
        results (dict): Dictionary containing benchmark results
        model_type (str): Type of RNN model ('gru' or 'lstm')
        output_dir (str): Directory to save plots
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot execution time vs sequence length
    plt.figure(figsize=(10, 6))
    plt.plot(results['seq_length'], results['cpu_time'], 'b-o', label='CPU')
    if CUDA_AVAILABLE:
        plt.plot(results['seq_length'], results['gpu_time'], 'r-o', label='GPU (Optimized)')
    plt.xscale('log', base=2)
    plt.yscale('log')
    plt.xlabel('Sequence Length')
    plt.ylabel('Execution Time (seconds)')
    plt.title(f'Min{model_type.upper()} Execution Time vs Sequence Length')
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'min{model_type.lower()}_execution_time.png'))
    
    # Plot speedup vs sequence length
    if CUDA_AVAILABLE:
        plt.figure(figsize=(10, 6))
        plt.plot(results['seq_length'], results['speedup'], 'g-o')
        plt.axhline(y=1, color='r', linestyle='--', label='CPU Baseline')
        plt.xscale('log', base=2)
        plt.xlabel('Sequence Length')
        plt.ylabel('Speedup (CPU/GPU)')
        plt.title(f'Min{model_type.upper()} GPU Speedup vs Sequence Length')
        plt.grid(True, which="both", ls="--")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'min{model_type.lower()}_speedup.png'))


def main():
    parser = argparse.ArgumentParser(description='Benchmark CUDA-optimized RNN implementations.')
    parser.add_argument('--model', type=str, default='both', choices=['gru', 'lstm', 'both'], 
                        help='RNN model type to benchmark')
    parser.add_argument('--input-size', type=int, default=64, 
                        help='Size of input vector')
    parser.add_argument('--hidden-size', type=int, default=128, 
                        help='Size of hidden state vector')
    parser.add_argument('--seq-lengths', type=str, default='8,16,32,64,128,256,512,1024', 
                        help='Comma-separated list of sequence lengths to test')
    parser.add_argument('--num-runs', type=int, default=5, 
                        help='Number of runs to average for each test case')
    parser.add_argument('--cuda-device', type=int, default=0, 
                        help='CUDA device to use')
    parser.add_argument('--output-dir', type=str, default='./benchmark_results', 
                        help='Directory to save results')
    
    args = parser.parse_args()
    
    # Parse sequence lengths
    seq_lengths = [int(s) for s in args.seq_lengths.split(',')]
    
    # Run benchmarks
    if args.model in ['gru', 'both']:
        print("\n" + "="*50)
        print("Benchmarking MinGRU models")
        print("="*50)
        gru_results = run_benchmark(
            model_type='gru',
            input_size=args.input_size,
            hidden_size=args.hidden_size,
            seq_lengths=seq_lengths,
            num_runs=args.num_runs,
            cuda_device=args.cuda_device
        )
        plot_results(gru_results, 'gru', args.output_dir)
    
    if args.model in ['lstm', 'both']:
        print("\n" + "="*50)
        print("Benchmarking MinLSTM models")
        print("="*50)
        lstm_results = run_benchmark(
            model_type='lstm',
            input_size=args.input_size,
            hidden_size=args.hidden_size,
            seq_lengths=seq_lengths,
            num_runs=args.num_runs,
            cuda_device=args.cuda_device
        )
        plot_results(lstm_results, 'lstm', args.output_dir)
    
    print("\nBenchmark complete. Results saved to:", args.output_dir)


if __name__ == "__main__":
    main()
