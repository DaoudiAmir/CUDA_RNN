import numpy as np
import time
import argparse
import matplotlib.pyplot as plt
from min_rnn.min_gru import MinGRUCell
from min_rnn.min_gru_cuda import MinGRUCellCUDA
from min_rnn.min_lstm import MinLSTMCell
from min_rnn.min_lstm_cuda import MinLSTMCellCUDA

def benchmark_model(model_type, device, input_size, hidden_size, seq_length, batch_size=1, num_runs=3):
    """Benchmark a model's performance with large computation needs.
    
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
    print(f"  Running {model_type.upper()} on {device.upper()} with:")
    print(f"    - Input size: {input_size}")
    print(f"    - Hidden size: {hidden_size}")
    print(f"    - Sequence length: {seq_length}")
    
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
    print("  Performing warm-up run...")
    if device == 'cpu':
        model.process_sequence(x, h0, seq_length)
    else:
        model.process_sequence_parallel(x, h0, seq_length)
    
    # Benchmark
    times = []
    print(f"  Running {num_runs} benchmarks...")
    for i in range(num_runs):
        print(f"    Run {i+1}/{num_runs}...")
        start_time = time.time()
        if device == 'cpu':
            model.process_sequence(x, h0, seq_length)
        else:
            model.process_sequence_parallel(x, h0, seq_length)
        end_time = time.time()
        times.append((end_time - start_time) * 1000)  # Convert to ms
        print(f"    Time: {times[-1]:.2f} ms")
    
    avg_time = sum(times) / len(times)
    print(f"  Average time: {avg_time:.2f} ms")
    return avg_time

def main():
    parser = argparse.ArgumentParser(description='Benchmark MinRNN models with large computation')
    parser.add_argument('--model', choices=['gru', 'lstm', 'both'], default='both',
                        help='Model type to benchmark')
    parser.add_argument('--device', choices=['cpu', 'gpu', 'both'], default='both',
                        help='Device to run on')
    parser.add_argument('--input-size', type=int, default=1024,
                        help='Input size')
    parser.add_argument('--hidden-size', type=int, default=1024,
                        help='Hidden size')
    parser.add_argument('--seq-length', type=int, default=2000,
                        help='Sequence length')
    parser.add_argument('--num-runs', type=int, default=3,
                        help='Number of runs to average over')
    
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
    
    results = {}
    
    for model_type in model_types:
        for device in devices:
            key = f"{model_type}_{device}"
            print(f"\nBenchmarking {model_type.upper()} on {device.upper()}...")
            
            avg_time = benchmark_model(
                model_type, device, args.input_size, args.hidden_size, 
                args.seq_length, num_runs=args.num_runs
            )
            results[key] = avg_time
    
    # Print speedup for GPU vs CPU
    print("\n=== RESULTS SUMMARY ===")
    for model_type in model_types:
        if 'cpu' in devices and 'gpu' in devices:
            cpu_time = results[f"{model_type}_cpu"]
            gpu_time = results[f"{model_type}_gpu"]
            
            speedup = cpu_time / gpu_time
            print(f"{model_type.upper()} Speedup (GPU vs CPU): {speedup:.2f}x")
            print(f"  CPU: {cpu_time:.2f} ms")
            print(f"  GPU: {gpu_time:.2f} ms")

if __name__ == '__main__':
    main()
