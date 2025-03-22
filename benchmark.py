import numpy as np
import time
import argparse
import matplotlib.pyplot as plt
from min_rnn.min_gru import MinGRUCell
from min_rnn.min_gru_cuda import MinGRUCellCUDA
from min_rnn.min_lstm import MinLSTMCell
from min_rnn.min_lstm_cuda import MinLSTMCellCUDA

def benchmark_model(model_type, device, input_size, hidden_size, seq_length, batch_size=1, num_runs=10):
    """Benchmark a model's performance.
    
    Args:
        model_type (str): 'gru' or 'lstm'
        device (str): 'cpu' or 'gpu'
        input_size (int): Input size
        hidden_size (int): Hidden size
        seq_length (int): Sequence length
        batch_size (int): Batch size (currently only 1 is supported)
        num_runs (int): Number of runs to average over
        
    Returns:
        float: Average time in milliseconds
    """
    # Initialize model
    if model_type == 'gru':
        if device == 'cpu':
            model = MinGRUCell(input_size, hidden_size)
        else:
            model = MinGRUCellCUDA(input_size, hidden_size)
    elif model_type == 'lstm':
        if device == 'cpu':
            model = MinLSTMCell(input_size, hidden_size)
        else:
            model = MinLSTMCellCUDA(input_size, hidden_size)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Generate random input data
    x = np.random.randn(seq_length, input_size).astype(np.float32)
    h0 = np.random.randn(hidden_size).astype(np.float32)
    
    # Warm-up run
    if device == 'cpu':
        model.process_sequence(x, h0, seq_length)
    else:
        model.process_sequence_parallel(x, h0, seq_length)
    
    # Benchmark
    times = []
    for _ in range(num_runs):
        start_time = time.time()
        if device == 'cpu':
            model.process_sequence(x, h0, seq_length)
        else:
            model.process_sequence_parallel(x, h0, seq_length)
        end_time = time.time()
        times.append((end_time - start_time) * 1000)  # Convert to ms
    
    avg_time = sum(times) / len(times)
    return avg_time

def run_benchmarks(model_types, devices, input_size, hidden_size, seq_lengths, batch_size=1, num_runs=10):
    """Run benchmarks for different configurations.
    
    Args:
        model_types (list): List of model types ('gru', 'lstm')
        devices (list): List of devices ('cpu', 'gpu')
        input_size (int): Input size
        hidden_size (int): Hidden size
        seq_lengths (list): List of sequence lengths to test
        batch_size (int): Batch size
        num_runs (int): Number of runs to average over
        
    Returns:
        dict: Results dictionary
    """
    results = {}
    
    for model_type in model_types:
        for device in devices:
            key = f"{model_type}_{device}"
            results[key] = []
            
            print(f"Benchmarking {model_type.upper()} on {device.upper()}...")
            for seq_length in seq_lengths:
                print(f"  Sequence length: {seq_length}")
                avg_time = benchmark_model(
                    model_type, device, input_size, hidden_size, 
                    seq_length, batch_size, num_runs
                )
                results[key].append(avg_time)
                print(f"  Average time: {avg_time:.2f} ms")
    
    return results

def plot_results(results, seq_lengths, output_file=None):
    """Plot benchmark results.
    
    Args:
        results (dict): Results dictionary
        seq_lengths (list): List of sequence lengths
        output_file (str, optional): Output file path for the plot
    """
    plt.figure(figsize=(10, 6))
    
    for key, times in results.items():
        model_type, device = key.split('_')
        label = f"{model_type.upper()} ({device.upper()})"
        marker = 'o' if device == 'cpu' else 's'
        linestyle = '-' if device == 'cpu' else '--'
        color = 'blue' if model_type == 'gru' else 'red'
        
        plt.plot(seq_lengths, times, marker=marker, linestyle=linestyle, 
                 label=label, color=color)
    
    plt.xlabel('Sequence Length')
    plt.ylabel('Time (ms)')
    plt.title('Model Performance Comparison')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    if output_file:
        plt.savefig(output_file)
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Benchmark MinRNN models')
    parser.add_argument('--model', choices=['gru', 'lstm', 'both'], default='both',
                        help='Model type to benchmark')
    parser.add_argument('--device', choices=['cpu', 'gpu', 'both'], default='both',
                        help='Device to run on')
    parser.add_argument('--input-size', type=int, default=128,
                        help='Input size')
    parser.add_argument('--hidden-size', type=int, default=128,
                        help='Hidden size')
    parser.add_argument('--seq-lengths', type=int, nargs='+', 
                        default=[10, 50, 100, 200, 500, 1000],
                        help='Sequence lengths to test')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Batch size')
    parser.add_argument('--num-runs', type=int, default=5,
                        help='Number of runs to average over')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file for the plot')
    
    args = parser.parse_args()
    
    # Determine which models and devices to benchmark
    model_types = []
    if args.model == 'both' or args.model == 'gru':
        model_types.append('gru')
    if args.model == 'both' or args.model == 'lstm':
        model_types.append('lstm')
    
    devices = []
    if args.device == 'both' or args.device == 'cpu':
        devices.append('cpu')
    if args.device == 'both' or args.device == 'gpu':
        devices.append('gpu')
    
    # Run benchmarks
    results = run_benchmarks(
        model_types, devices, args.input_size, args.hidden_size,
        args.seq_lengths, args.batch_size, args.num_runs
    )
    
    # Plot results
    plot_results(results, args.seq_lengths, args.output)
    
    # Print speedup for GPU vs CPU
    for model_type in model_types:
        if 'cpu' in devices and 'gpu' in devices:
            cpu_times = results[f"{model_type}_cpu"]
            gpu_times = results[f"{model_type}_gpu"]
            
            print(f"\n{model_type.upper()} Speedup (CPU vs GPU):")
            for i, seq_length in enumerate(args.seq_lengths):
                speedup = cpu_times[i] / gpu_times[i]
                print(f"  Sequence length {seq_length}: {speedup:.2f}x")

if __name__ == '__main__':
    main()
