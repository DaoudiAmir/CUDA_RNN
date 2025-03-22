import numpy as np
import argparse
import time
import os
import sys

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from min_rnn.min_gru import MinGRUCell
from min_rnn.min_lstm import MinLSTMCell
from min_rnn.min_gru_cuda import MinGRUCellCUDA
from min_rnn.min_lstm_cuda import MinLSTMCellCUDA
from min_rnn.cuda_utils import CUDA_AVAILABLE
from min_rnn.benchmarks.benchmark import run_benchmarks

def demo_min_gru(use_cuda=False):
    """Demonstrate the use of MinGRU.
    
    Args:
        use_cuda (bool): Whether to use CUDA acceleration
    """
    print("\n--- MinGRU Demo ---")
    
    # Parameters
    input_size = 10
    hidden_size = 20
    seq_length = 5
    
    # Create cell
    if use_cuda and CUDA_AVAILABLE:
        print("Using CUDA acceleration")
        cell = MinGRUCellCUDA(input_size, hidden_size)
    else:
        print("Using CPU implementation")
        cell = MinGRUCell(input_size, hidden_size)
    
    # Generate random input data
    x = np.random.random((seq_length, input_size))
    h0 = np.zeros(hidden_size)
    
    # Process single step
    print("\nProcessing single step...")
    h1 = cell.forward(x[0], h0)
    print(f"Input shape: {x[0].shape}")
    print(f"Hidden state shape: {h1.shape}")
    print(f"Hidden state (first 5 elements): {h1[:5]}")
    
    # Process sequence (sequential mode)
    print("\nProcessing sequence (sequential mode)...")
    start_time = time.time()
    h_seq = cell.process_sequence(x, h0)
    seq_time = time.time() - start_time
    print(f"Sequence shape: {x.shape}")
    print(f"Output shape: {h_seq.shape}")
    print(f"Output (last time step, first 5 elements): {h_seq[-1, :5]}")
    print(f"Processing time: {seq_time:.6f} seconds")
    
    # Process sequence (parallel mode)
    print("\nProcessing sequence (parallel mode)...")
    start_time = time.time()
    h_par = cell.process_sequence_parallel(x, h0)
    par_time = time.time() - start_time
    print(f"Output shape: {h_par.shape}")
    print(f"Output (last time step, first 5 elements): {h_par[-1, :5]}")
    print(f"Processing time: {par_time:.6f} seconds")
    if par_time > 0:
        print(f"Speedup: {seq_time / par_time:.2f}x")
    else:
        print("Speedup: N/A (parallel processing too fast to measure)")
    
    # Check that sequential and parallel modes produce the same results
    max_diff = np.max(np.abs(h_seq - h_par))
    print(f"\nMaximum difference between sequential and parallel modes: {max_diff:.8f}")

def demo_min_lstm(use_cuda=False):
    """Demonstrate the use of MinLSTM.
    
    Args:
        use_cuda (bool): Whether to use CUDA acceleration
    """
    print("\n--- MinLSTM Demo ---")
    
    # Parameters
    input_size = 10
    hidden_size = 20
    seq_length = 5
    
    # Create cell
    if use_cuda and CUDA_AVAILABLE:
        print("Using CUDA acceleration")
        cell = MinLSTMCellCUDA(input_size, hidden_size)
    else:
        print("Using CPU implementation")
        cell = MinLSTMCell(input_size, hidden_size)
    
    # Generate random input data
    x = np.random.random((seq_length, input_size))
    h0 = np.zeros(hidden_size)
    
    # Process single step
    print("\nProcessing single step...")
    h1 = cell.forward(x[0], h0)
    print(f"Input shape: {x[0].shape}")
    print(f"Hidden state shape: {h1.shape}")
    print(f"Hidden state (first 5 elements): {h1[:5]}")
    
    # Process sequence (sequential mode)
    print("\nProcessing sequence (sequential mode)...")
    start_time = time.time()
    h_seq = cell.process_sequence(x, h0)
    seq_time = time.time() - start_time
    print(f"Sequence shape: {x.shape}")
    print(f"Output shape: {h_seq.shape}")
    print(f"Output (last time step, first 5 elements): {h_seq[-1, :5]}")
    print(f"Processing time: {seq_time:.6f} seconds")
    
    # Process sequence (parallel mode)
    print("\nProcessing sequence (parallel mode)...")
    start_time = time.time()
    h_par = cell.process_sequence_parallel(x, h0)
    par_time = time.time() - start_time
    print(f"Output shape: {h_par.shape}")
    print(f"Output (last time step, first 5 elements): {h_par[-1, :5]}")
    print(f"Processing time: {par_time:.6f} seconds")
    if par_time > 0:
        print(f"Speedup: {seq_time / par_time:.2f}x")
    else:
        print("Speedup: N/A (parallel processing too fast to measure)")
    
    # Check that sequential and parallel modes produce the same results
    max_diff = np.max(np.abs(h_seq - h_par))
    print(f"\nMaximum difference between sequential and parallel modes: {max_diff:.8f}")

def main():
    parser = argparse.ArgumentParser(description="Minimal RNN Demo")
    parser.add_argument("--model", type=str, default="all", choices=["gru", "lstm", "all"],
                        help="Model to run (gru, lstm, or all)")
    parser.add_argument("--cuda", action="store_true", help="Use CUDA acceleration")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmarks")
    args = parser.parse_args()
    
    # Print CUDA availability
    if args.cuda:
        if CUDA_AVAILABLE:
            print("CUDA is available and will be used for acceleration")
        else:
            print("CUDA is not available, falling back to CPU implementation")
    
    # Run demos
    if args.model in ["gru", "all"]:
        demo_min_gru(args.cuda)
    
    if args.model in ["lstm", "all"]:
        demo_min_lstm(args.cuda)
    
    # Run benchmarks
    if args.benchmark:
        run_benchmarks()

if __name__ == "__main__":
    main()
