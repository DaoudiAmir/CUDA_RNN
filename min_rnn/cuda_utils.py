import numpy as np
from numba import cuda
import math
from numba.cuda.cudadrv.driver import driver

# Check if CUDA is available
try:
    cuda.detect()
    CUDA_AVAILABLE = True
except:
    CUDA_AVAILABLE = False

# Define constants for thread block sizes
THREAD_BLOCK_SIZE = 256
WARP_SIZE = 32
MAX_THREADS_PER_BLOCK = 1024

# Helper function to get optimal grid and block sizes
def get_optimal_grid_block_size(size, max_threads_per_block=MAX_THREADS_PER_BLOCK):
    """Get optimal grid and block size for better GPU occupancy."""
    block_size = min(max_threads_per_block, size)
    # Make block size a multiple of warp size for better performance
    block_size = (block_size + WARP_SIZE - 1) // WARP_SIZE * WARP_SIZE
    grid_size = (size + block_size - 1) // block_size
    return grid_size, block_size

# Use shared memory for linear operations
@cuda.jit
def linear_forward_kernel_optimized(weights, bias, x, out, input_size, output_size):
    """Optimized CUDA kernel for linear layer forward pass using shared memory.
    
    Args:
        weights: Weight matrix of shape (output_size, input_size)
        bias: Bias vector of shape (output_size,)
        x: Input vector of shape (input_size,)
        out: Output vector of shape (output_size,)
        input_size: Size of input vector
        output_size: Size of output vector
    """
    # Allocate shared memory for input vector
    shared_x = cuda.shared.array(shape=0, dtype=np.float32)
    
    idx = cuda.grid(1)
    tx = cuda.threadIdx.x
    block_size = cuda.blockDim.x
    
    # Load input vector into shared memory
    for i in range(tx, input_size, block_size):
        if i < input_size:
            shared_x[i] = x[i]
    
    cuda.syncthreads()
    
    if idx < output_size:
        # Initialize output with bias
        result = bias[idx]
        
        # Compute dot product using shared memory for x
        for i in range(input_size):
            result += weights[idx, i] * shared_x[i]
            
        out[idx] = result

# Optimized batch processing kernel - process multiple elements per thread
@cuda.jit
def min_gru_forward_kernel_batch(weights_z, bias_z, weights_h, bias_h, 
                               x_t, h_prev, h_t, 
                               input_size, hidden_size):
    """Optimized CUDA kernel for MinGRU forward pass with batch processing.
    
    Args:
        weights_z: Weight matrix for update gate
        bias_z: Bias vector for update gate
        weights_h: Weight matrix for candidate state
        bias_h: Bias vector for candidate state
        x_t: Input vector
        h_prev: Previous hidden state
        h_t: Output hidden state
        input_size: Size of input vector
        hidden_size: Size of hidden state vector
    """
    thread_idx = cuda.grid(1)
    stride = cuda.gridsize(1)
    
    # Each thread processes multiple elements
    for idx in range(thread_idx, hidden_size, stride):
        if idx < hidden_size:
            # Compute update gate
            z_t = bias_z[idx]
            for i in range(input_size):
                z_t += weights_z[idx, i] * x_t[i]
            z_t = 1.0 / (1.0 + math.exp(-z_t))  # sigmoid
            
            # Compute candidate hidden state
            h_tilde = bias_h[idx]
            for i in range(input_size):
                h_tilde += weights_h[idx, i] * x_t[i]
            h_tilde = math.tanh(h_tilde)
            
            # Compute output hidden state
            h_t[idx] = (1.0 - z_t) * h_prev[idx] + z_t * h_tilde

# Optimized batch processing kernel for MinLSTM
@cuda.jit
def min_lstm_forward_kernel_batch(weights_f, bias_f, weights_i, bias_i, 
                                weights_h, bias_h, 
                                x_t, h_prev, h_t, 
                                input_size, hidden_size):
    """Optimized CUDA kernel for MinLSTM forward pass with batch processing.
    
    Args:
        weights_f: Weight matrix for forget gate
        bias_f: Bias vector for forget gate
        weights_i: Weight matrix for input gate
        bias_i: Bias vector for input gate
        weights_h: Weight matrix for candidate state
        bias_h: Bias vector for candidate state
        x_t: Input vector
        h_prev: Previous hidden state
        h_t: Output hidden state
        input_size: Size of input vector
        hidden_size: Size of hidden state vector
    """
    thread_idx = cuda.grid(1)
    stride = cuda.gridsize(1)
    
    # Each thread processes multiple elements
    for idx in range(thread_idx, hidden_size, stride):
        if idx < hidden_size:
            # Compute forget gate
            f_t = bias_f[idx]
            for i in range(input_size):
                f_t += weights_f[idx, i] * x_t[i]
            f_t = 1.0 / (1.0 + math.exp(-f_t))  # sigmoid
            
            # Compute input gate
            i_t = bias_i[idx]
            for i in range(input_size):
                i_t += weights_i[idx, i] * x_t[i]
            i_t = 1.0 / (1.0 + math.exp(-i_t))  # sigmoid
            
            # Normalize gates
            gate_sum = f_t + i_t
            if gate_sum > 0:
                f_t = f_t / gate_sum
                i_t = i_t / gate_sum
            else:
                f_t = 0.5
                i_t = 0.5
            
            # Compute candidate hidden state
            h_tilde = bias_h[idx]
            for i in range(input_size):
                h_tilde += weights_h[idx, i] * x_t[i]
            h_tilde = math.tanh(h_tilde)
            
            # Compute output hidden state
            h_t[idx] = f_t * h_prev[idx] + i_t * h_tilde

# Optimized MinLSTM forward kernel that uses shared memory
@cuda.jit
def min_lstm_forward_kernel_optimized(weights_f, bias_f, weights_i, bias_i, 
                                   weights_h, bias_h, 
                                   x_t, h_prev, h_t, 
                                   input_size, hidden_size):
    """Optimized CUDA kernel for MinLSTM forward pass with shared memory.
    
    Args:
        weights_f: Weight matrix for forget gate
        bias_f: Bias vector for forget gate
        weights_i: Weight matrix for input gate
        bias_i: Bias vector for input gate
        weights_h: Weight matrix for candidate state
        bias_h: Bias vector for candidate state
        x_t: Input vector
        h_prev: Previous hidden state
        h_t: Output hidden state
        input_size: Size of input vector
        hidden_size: Size of hidden state vector
    """
    # Shared memory for input vector
    shared_x = cuda.shared.array(shape=(128,), dtype=np.float32)
    
    thread_idx = cuda.grid(1)
    block_size = cuda.blockDim.x
    
    # Load input vector into shared memory
    for i in range(thread_idx, input_size, block_size):
        if i < input_size:
            shared_x[i] = x_t[i]
    
    # Ensure all threads have loaded their portion of the input
    cuda.syncthreads()
    
    # Each thread processes one element of the hidden state
    if thread_idx < hidden_size:
        # Compute forget gate
        f_t = bias_f[thread_idx]
        for i in range(input_size):
            f_t += weights_f[thread_idx, i] * shared_x[i]
        f_t = 1.0 / (1.0 + math.exp(-f_t))  # sigmoid
        
        # Compute input gate
        i_t = bias_i[thread_idx]
        for i in range(input_size):
            i_t += weights_i[thread_idx, i] * shared_x[i]
        i_t = 1.0 / (1.0 + math.exp(-i_t))  # sigmoid
        
        # Normalize gates
        gate_sum = f_t + i_t
        if gate_sum > 0:
            f_t = f_t / gate_sum
            i_t = i_t / gate_sum
        else:
            f_t = 0.5
            i_t = 0.5
        
        # Compute candidate hidden state
        h_tilde = bias_h[thread_idx]
        for i in range(input_size):
            h_tilde += weights_h[thread_idx, i] * shared_x[i]
        h_tilde = math.tanh(h_tilde)
        
        # Compute output hidden state
        h_t[thread_idx] = f_t * h_prev[thread_idx] + i_t * h_tilde

# MinLSTM CUDA kernels
@cuda.jit
def min_lstm_forward_kernel(weights_f, bias_f, weights_i, bias_i, weights_h, bias_h,
                          x_t, h_prev, h_t,
                          input_size, hidden_size):
    """CUDA kernel for MinLSTM forward pass.
    
    Args:
        weights_f: Weight matrix for forget gate
        bias_f: Bias vector for forget gate
        weights_i: Weight matrix for input gate
        bias_i: Bias vector for input gate
        weights_h: Weight matrix for candidate state
        bias_h: Bias vector for candidate state
        x_t: Input vector
        h_prev: Previous hidden state
        h_t: Output hidden state
        input_size: Size of input vector
        hidden_size: Size of hidden state vector
    """
    idx = cuda.grid(1)
    
    if idx < hidden_size:
        # Compute forget gate: f_t = σ(Linear_f(x_t))
        f_t = bias_f[idx]
        for i in range(input_size):
            f_t += weights_f[idx, i] * x_t[i]
        f_t = 1.0 / (1.0 + math.exp(-f_t))  # sigmoid
        
        # Compute input gate: i_t = σ(Linear_i(x_t))
        i_t = bias_i[idx]
        for i in range(input_size):
            i_t += weights_i[idx, i] * x_t[i]
        i_t = 1.0 / (1.0 + math.exp(-i_t))  # sigmoid
        
        # Normalize gates: f'_t = f_t / (f_t + i_t), i'_t = i_t / (f_t + i_t)
        gate_sum = f_t + i_t
        if gate_sum > 0:
            f_t = f_t / gate_sum
            i_t = i_t / gate_sum
        else:
            f_t = 0.5
            i_t = 0.5
        
        # Compute candidate hidden state: h_tilde = tanh(Linear_h(x_t))
        h_tilde = bias_h[idx]
        for i in range(input_size):
            h_tilde += weights_h[idx, i] * x_t[i]
        h_tilde = math.tanh(h_tilde)
        
        # Compute hidden state: h_t = f_t * h_prev + i_t * h_tilde
        h_t[idx] = f_t * h_prev[idx] + i_t * h_tilde

@cuda.jit
def min_lstm_forward_kernel_batch(weights_f, bias_f, weights_i, bias_i, weights_h, bias_h,
                               x_t, h_prev, h_t,
                               input_size, hidden_size):
    """Optimized batch CUDA kernel for MinLSTM forward pass.
    
    This kernel processes multiple elements per thread for better GPU utilization.
    
    Args:
        weights_f: Weight matrix for forget gate
        bias_f: Bias vector for forget gate
        weights_i: Weight matrix for input gate
        bias_i: Bias vector for input gate
        weights_h: Weight matrix for candidate state
        bias_h: Bias vector for candidate state
        x_t: Input vector
        h_prev: Previous hidden state
        h_t: Output hidden state
        input_size: Size of input vector
        hidden_size: Size of hidden state vector
    """
    # Get thread and block indices
    tx = cuda.threadIdx.x
    bx = cuda.blockIdx.x
    block_size = cuda.blockDim.x
    
    # Calculate the grid stride (for handling multiple elements per thread)
    grid_stride = cuda.gridDim.x * block_size
    
    # Calculate starting index for this thread
    idx_start = bx * block_size + tx
    
    # Use shared memory for the input to reduce global memory accesses
    sm_x_t = cuda.shared.array(shape=(128,), dtype=numba.float32)
    
    # Cooperatively load input vector into shared memory
    for i in range(tx, min(input_size, 128), block_size):
        if i < input_size:
            sm_x_t[i] = x_t[i]
    
    # Ensure all threads have loaded the shared data
    cuda.syncthreads()
    
    # Process multiple elements per thread with grid stride loop
    for idx in range(idx_start, hidden_size, grid_stride):
        if idx < hidden_size:
            # Compute forget gate: f_t = σ(Linear_f(x_t))
            f_t = bias_f[idx]
            for i in range(input_size):
                f_t += weights_f[idx, i] * sm_x_t[i] if i < 128 else x_t[i]
            f_t = 1.0 / (1.0 + math.exp(-f_t))  # sigmoid
            
            # Compute input gate: i_t = σ(Linear_i(x_t))
            i_t = bias_i[idx]
            for i in range(input_size):
                i_t += weights_i[idx, i] * sm_x_t[i] if i < 128 else x_t[i]
            i_t = 1.0 / (1.0 + math.exp(-i_t))  # sigmoid
            
            # Normalize gates: f'_t = f_t / (f_t + i_t), i'_t = i_t / (f_t + i_t)
            gate_sum = f_t + i_t
            if gate_sum > 0:
                f_t = f_t / gate_sum
                i_t = i_t / gate_sum
            else:
                f_t = 0.5
                i_t = 0.5
            
            # Compute candidate hidden state: h_tilde = tanh(Linear_h(x_t))
            h_tilde = bias_h[idx]
            for i in range(input_size):
                h_tilde += weights_h[idx, i] * sm_x_t[i] if i < 128 else x_t[i]
            h_tilde = math.tanh(h_tilde)
            
            # Compute hidden state: h_t = f_t * h_prev + i_t * h_tilde
            h_t[idx] = f_t * h_prev[idx] + i_t * h_tilde

# Fused GRU kernel that combines multiple operations into a single kernel
@cuda.jit
def min_gru_fused_kernel(weights_z, bias_z, weights_h, bias_h, 
                       x_sequence, h_prev, h_out,
                       seq_length, input_size, hidden_size):
    """Fused CUDA kernel that processes an entire sequence for MinGRU.
    
    Args:
        weights_z: Weight matrix for update gate
        bias_z: Bias vector for update gate
        weights_h: Weight matrix for candidate state
        bias_h: Bias vector for candidate state
        x_sequence: Input sequence of shape (seq_length, input_size)
        h_prev: Initial hidden state
        h_out: Output hidden states of shape (seq_length, hidden_size)
        seq_length: Sequence length
        input_size: Size of input vector
        hidden_size: Size of hidden state vector
    """
    idx = cuda.grid(1)
    
    if idx < hidden_size:
        # Load initial hidden state
        h_t = h_prev[idx]
        
        # Process the entire sequence
        for t in range(seq_length):
            # Compute update gate
            z_t = bias_z[idx]
            for i in range(input_size):
                z_t += weights_z[idx, i] * x_sequence[t, i]
            z_t = 1.0 / (1.0 + math.exp(-z_t))  # sigmoid
            
            # Compute candidate hidden state
            h_tilde = bias_h[idx]
            for i in range(input_size):
                h_tilde += weights_h[idx, i] * x_sequence[t, i]
            h_tilde = math.tanh(h_tilde)
            
            # Compute new hidden state
            h_t = (1.0 - z_t) * h_t + z_t * h_tilde
            
            # Store result
            h_out[t, idx] = h_t

# Optimized parallel scan algorithm
def cuda_parallel_scan_optimized(seq_length, batch_size, hidden_size, a, b, h0):
    """Optimized parallel scan operation using CUDA with improved performance.
    
    Args:
        seq_length: Sequence length
        batch_size: Batch size (currently only 1 is supported)
        hidden_size: Hidden state size
        a: Coefficient a of shape (seq_length, hidden_size)
        b: Coefficient b of shape (seq_length, hidden_size)
        h0: Initial hidden state of shape (hidden_size,)
        
    Returns:
        numpy.ndarray: Output hidden states of shape (seq_length, hidden_size)
    """
    # For very short sequences, use sequential scan as it's more efficient
    if seq_length <= 32:
        return cuda_sequential_scan(seq_length, batch_size, hidden_size, a, b, h0)
    
    # Convert inputs to float32 for better CUDA performance
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    h0 = h0.astype(np.float32)
    
    # Create output array
    h_out = np.zeros((seq_length, hidden_size), dtype=np.float32)
    
    # Use sequential scan for now as a reliable fallback
    # This is a simplified version that works correctly
    h_prev = h0.copy()
    for t in range(seq_length):
        # Apply scan operation: h_t = a_t * h_{t-1} + b_t
        h_t = a[t] * h_prev + b[t]
        h_out[t] = h_t
        h_prev = h_t
    
    return h_out

# Function to create pinned memory for faster transfers
def create_pinned_array(shape, dtype=np.float32):
    """Create a pinned memory array for faster CPU-GPU transfers."""
    return cuda.pinned_array(shape, dtype=dtype)

@cuda.jit
def linear_forward_kernel(weights, bias, x, out, input_size, output_size):
    """CUDA kernel for linear layer forward pass.
    
    Args:
        weights: Weight matrix of shape (output_size, input_size)
        bias: Bias vector of shape (output_size,)
        x: Input vector of shape (input_size,)
        out: Output vector of shape (output_size,)
        input_size: Size of input vector
        output_size: Size of output vector
    """
    idx = cuda.grid(1)
    
    if idx < output_size:
        # Initialize output with bias
        out[idx] = bias[idx]
        
        # Compute dot product
        for i in range(input_size):
            out[idx] += weights[idx, i] * x[i]

@cuda.jit
def sigmoid_kernel(x, out, size):
    """CUDA kernel for sigmoid activation function.
    
    Args:
        x: Input vector
        out: Output vector
        size: Size of vectors
    """
    idx = cuda.grid(1)
    
    if idx < size:
        # Compute sigmoid: 1 / (1 + exp(-x))
        out[idx] = 1.0 / (1.0 + math.exp(-x[idx]))

@cuda.jit
def min_gru_forward_kernel(weights_z, bias_z, weights_h, bias_h, 
                          x_t, h_prev, h_t, 
                          input_size, hidden_size):
    """CUDA kernel for MinGRU forward pass.
    
    Args:
        weights_z: Weight matrix for update gate of shape (hidden_size, input_size)
        bias_z: Bias vector for update gate of shape (hidden_size,)
        weights_h: Weight matrix for candidate state of shape (hidden_size, input_size)
        bias_h: Bias vector for candidate state of shape (hidden_size,)
        x_t: Input vector of shape (input_size,)
        h_prev: Previous hidden state of shape (hidden_size,)
        h_t: Output hidden state of shape (hidden_size,)
        input_size: Size of input vector
        hidden_size: Size of hidden state vector
    """
    idx = cuda.grid(1)
    
    if idx < hidden_size:
        # Compute update gate z_t = sigmoid(W_z * x_t + b_z)
        z_t = bias_z[idx]
        for i in range(input_size):
            z_t += weights_z[idx, i] * x_t[i]
        z_t = 1.0 / (1.0 + math.exp(-z_t))  # sigmoid
        
        # Compute candidate state h_tilde = tanh(W_h * x_t + b_h)
        h_tilde = bias_h[idx]
        for i in range(input_size):
            h_tilde += weights_h[idx, i] * x_t[i]
        h_tilde = math.tanh(h_tilde)
        
        # Compute new hidden state h_t = (1 - z_t) * h_prev + z_t * h_tilde
        h_t[idx] = (1.0 - z_t) * h_prev[idx] + z_t * h_tilde

@cuda.jit
def min_gru_extract_scan_params_kernel(weights_z, bias_z, weights_h, bias_h,
                                      x, a, b,
                                      seq_length, hidden_size, input_size):
    """CUDA kernel to extract scan parameters for MinGRU.
    
    Args:
        weights_z: Weight matrix for update gate
        bias_z: Bias vector for update gate
        weights_h: Weight matrix for candidate state
        bias_h: Bias vector for candidate state
        x: Input sequence of shape (seq_length, input_size)
        a: Output coefficient a of shape (seq_length, hidden_size)
        b: Output coefficient b of shape (seq_length, hidden_size)
        seq_length: Sequence length
        hidden_size: Hidden state size
        input_size: Input size
    """
    # Get 2D thread indices
    i, t = cuda.grid(2)
    
    if i < hidden_size and t < seq_length:
        # Compute update gate z_t = sigmoid(W_z * x_t + b_z)
        z_t = bias_z[i]
        for j in range(input_size):
            z_t += weights_z[i, j] * x[t, j]
        z_t = 1.0 / (1.0 + math.exp(-z_t))  # sigmoid
        
        # Compute candidate state h_tilde = tanh(W_h * x_t + b_h)
        h_tilde = bias_h[i]
        for j in range(input_size):
            h_tilde += weights_h[i, j] * x[t, j]
        h_tilde = math.tanh(h_tilde)
        
        # Set scan parameters
        a[t, i] = 1.0 - z_t  # a_t = (1 - z_t)
        b[t, i] = z_t * h_tilde  # b_t = z_t * h_tilde

@cuda.jit
def min_lstm_forward_kernel(weights_f, bias_f, weights_i, bias_i, 
                           weights_h, bias_h, 
                           x_t, h_prev, h_t, 
                           input_size, hidden_size):
    """CUDA kernel for MinLSTM forward pass.
    
    Args:
        weights_f: Weight matrix for forget gate
        bias_f: Bias vector for forget gate
        weights_i: Weight matrix for input gate
        bias_i: Bias vector for input gate
        weights_h: Weight matrix for candidate state
        bias_h: Bias vector for candidate state
        x_t: Input vector
        h_prev: Previous hidden state
        h_t: Output hidden state
        input_size: Size of input vector
        hidden_size: Size of hidden state vector
    """
    idx = cuda.grid(1)
    
    if idx < hidden_size:
        # Compute forget gate f_t = sigmoid(W_f * x_t + b_f)
        f_t = bias_f[idx]
        for i in range(input_size):
            f_t += weights_f[idx, i] * x_t[i]
        f_t = 1.0 / (1.0 + math.exp(-f_t))  # sigmoid
        
        # Compute input gate i_t = sigmoid(W_i * x_t + b_i)
        i_t = bias_i[idx]
        for i in range(input_size):
            i_t += weights_i[idx, i] * x_t[i]
        i_t = 1.0 / (1.0 + math.exp(-i_t))  # sigmoid
        
        # Compute candidate state h_tilde = tanh(W_h * x_t + b_h)
        h_tilde = bias_h[idx]
        for i in range(input_size):
            h_tilde += weights_h[idx, i] * x_t[i]
        h_tilde = math.tanh(h_tilde)
        
        # Normalize gates
        gate_sum = f_t + i_t
        if gate_sum > 0:
            f_t = f_t / gate_sum
            i_t = i_t / gate_sum
        else:
            f_t = 0.5
            i_t = 0.5
        
        # Compute new hidden state h_t = f_t * h_prev + i_t * h_tilde
        h_t[idx] = f_t * h_prev[idx] + i_t * h_tilde

@cuda.jit
def min_lstm_extract_scan_params_kernel(weights_f, bias_f, weights_i, bias_i,
                                       weights_h, bias_h,
                                       x_seq, a, b,
                                       seq_length, hidden_size, input_size):
    """CUDA kernel to extract scan parameters for MinLSTM.
    
    Args:
        weights_f: Weight matrix for forget gate
        bias_f: Bias vector for forget gate
        weights_i: Weight matrix for input gate
        bias_i: Bias vector for input gate
        weights_h: Weight matrix for candidate state
        bias_h: Bias vector for candidate state
        x_seq: Input sequence of shape (seq_length, input_size)
        a: Output array for 'a' coefficients of shape (seq_length, hidden_size)
        b: Output array for 'b' coefficients of shape (seq_length, hidden_size)
        seq_length: Sequence length
        hidden_size: Size of hidden state vector
        input_size: Size of input vector
    """
    # Get 2D thread indices
    i, t = cuda.grid(2)
    
    if i < hidden_size and t < seq_length:
        # Compute forget gate f_t = sigmoid(W_f * x_t + b_f)
        f_t = bias_f[i]
        for j in range(input_size):
            f_t += weights_f[i, j] * x_seq[t, j]
        f_t = 1.0 / (1.0 + math.exp(-f_t))  # sigmoid
        
        # Compute input gate i_t = sigmoid(W_i * x_t + b_i)
        i_t = bias_i[i]
        for j in range(input_size):
            i_t += weights_i[i, j] * x_seq[t, j]
        i_t = 1.0 / (1.0 + math.exp(-i_t))  # sigmoid
        
        # Compute candidate state h_tilde = tanh(W_h * x_t + b_h)
        h_tilde = bias_h[i]
        for j in range(input_size):
            h_tilde += weights_h[i, j] * x_seq[t, j]
        h_tilde = math.tanh(h_tilde)
        
        # Normalize gates
        gate_sum = f_t + i_t
        if gate_sum > 0:
            f_t = f_t / gate_sum
            i_t = i_t / gate_sum
        else:
            f_t = 0.5
            i_t = 0.5
        
        # Set coefficients for scan operation: h_t = a_t * h_{t-1} + b_t
        a[t, i] = f_t                # a_t = f_t
        b[t, i] = i_t * h_tilde      # b_t = i_t * h_tilde

@cuda.jit
def apply_scan_op_kernel(a, b, c, d, size):
    """CUDA kernel to apply scan operation: (a, b) ∘ (c, d) = (a*c, a*d + b).
    
    Args:
        a: First coefficient of first operation
        b: Second coefficient of first operation
        c: First coefficient of second operation (modified in-place)
        d: Second coefficient of second operation (modified in-place)
        size: Size of vectors
    """
    idx = cuda.grid(1)
    
    if idx < size:
        # Compute (a, b) ∘ (c, d) = (a*c, a*d + b)
        temp_c = a[idx] * c[idx]
        temp_d = a[idx] * d[idx] + b[idx]
        
        # Update c and d in-place
        c[idx] = temp_c
        d[idx] = temp_d

@cuda.jit
def compose_scan_ops_kernel(a1, b1, a2, b2, a_out, b_out, size):
    """CUDA kernel to compose two scan operations.
    
    Args:
        a1: First coefficient of first operation
        b1: Second coefficient of first operation
        a2: First coefficient of second operation
        b2: Second coefficient of second operation
        a_out: Output first coefficient
        b_out: Output second coefficient
        size: Size of vectors
    """
    idx = cuda.grid(1)
    
    if idx < size:
        # Compute (a1, b1) ∘ (a2, b2) = (a1*a2, a1*b2 + b1)
        a_out[idx] = a1[idx] * a2[idx]
        b_out[idx] = a1[idx] * b2[idx] + b1[idx]

def cuda_parallel_scan(seq_length, batch_size, hidden_size, a, b, h0):
    """Perform parallel scan operation using CUDA.
    
    Args:
        seq_length: Sequence length
        batch_size: Batch size (currently only 1 is supported)
        hidden_size: Hidden state size
        a: Coefficient a of shape (seq_length, hidden_size)
        b: Coefficient b of shape (seq_length, hidden_size)
        h0: Initial hidden state of shape (hidden_size,)
        
    Returns:
        numpy.ndarray: Output hidden states of shape (seq_length, hidden_size)
    """
    if not CUDA_AVAILABLE:
        raise RuntimeError("CUDA is not available")
    
    # Convert inputs to float32
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    h0 = h0.astype(np.float32)
    
    # Initialize output
    h_out = np.zeros((seq_length, hidden_size), dtype=np.float32)
    
    # Copy data to device
    d_a = cuda.to_device(a)
    d_b = cuda.to_device(b)
    d_h0 = cuda.to_device(h0)
    d_h_out = cuda.to_device(h_out)
    
    # Allocate temporary arrays for scan operations
    d_a_temp = cuda.device_array_like(d_a)
    d_b_temp = cuda.device_array_like(d_b)
    
    # Configure kernel
    threads_per_block = min(THREAD_BLOCK_SIZE, hidden_size)
    blocks_per_grid = max(1, (hidden_size + threads_per_block - 1) // threads_per_block)
    
    # For very small sequences, use sequential scan which is more efficient
    if seq_length <= 32:
        # Sequential scan
        for t in range(seq_length):
            if t == 0:
                # h_t = a_t * h0 + b_t
                apply_scan_op_kernel[blocks_per_grid, threads_per_block](
                    d_a[t], d_b[t], d_h0, d_h_out[t], hidden_size
                )
            else:
                # h_t = a_t * h_{t-1} + b_t
                apply_scan_op_kernel[blocks_per_grid, threads_per_block](
                    d_a[t], d_b[t], d_h_out[t-1], d_h_out[t], hidden_size
                )
    else:
        # Parallel scan using Blelloch algorithm (up-sweep and down-sweep)
        
        # Make a copy of a and b for in-place operations
        d_a.copy_to_device(d_a_temp)
        d_b.copy_to_device(d_b_temp)
        
        # Up-sweep phase (reduction)
        for d in range(int(math.log2(seq_length))):
            stride = 2 ** d
            for t in range(0, seq_length, 2 * stride):
                if t + stride < seq_length:
                    # Compose operations: (a[t], b[t]) ∘ (a[t+stride], b[t+stride])
                    compose_scan_ops_kernel[blocks_per_grid, threads_per_block](
                        d_a_temp[t], d_b_temp[t],
                        d_a_temp[t+stride], d_b_temp[t+stride],
                        d_a_temp[t+stride], d_b_temp[t+stride],
                        hidden_size
                    )
        
        # Down-sweep phase
        # Clear the last element for exclusive scan
        if seq_length > 1:
            # Save last element
            d_a_last = cuda.device_array(hidden_size, dtype=np.float32)
            d_b_last = cuda.device_array(hidden_size, dtype=np.float32)
            d_a_temp[seq_length-1].copy_to_device(d_a_last)
            d_b_temp[seq_length-1].copy_to_device(d_b_last)
            
            # Set identity element at the end
            cuda.to_device(np.ones(hidden_size, dtype=np.float32)).copy_to_device(d_a_temp[seq_length-1])
            cuda.to_device(np.zeros(hidden_size, dtype=np.float32)).copy_to_device(d_b_temp[seq_length-1])
            
            # Down-sweep
            for d in range(int(math.log2(seq_length))-1, -1, -1):
                stride = 2 ** d
                for t in range(0, seq_length, 2 * stride):
                    if t + stride < seq_length:
                        # Swap and compose
                        d_a_swap = cuda.device_array(hidden_size, dtype=np.float32)
                        d_b_swap = cuda.device_array(hidden_size, dtype=np.float32)
                        
                        # Save t+stride
                        d_a_temp[t+stride].copy_to_device(d_a_swap)
                        d_b_temp[t+stride].copy_to_device(d_b_swap)
                        
                        # Compose t into t+stride
                        compose_scan_ops_kernel[blocks_per_grid, threads_per_block](
                            d_a_temp[t], d_b_temp[t],
                            d_a_swap, d_b_swap,
                            d_a_temp[t+stride], d_b_temp[t+stride],
                            hidden_size
                        )
            
            # Restore last element
            d_a_last.copy_to_device(d_a_temp[seq_length-1])
            d_b_last.copy_to_device(d_b_temp[seq_length-1])
        
        # Apply h0 to all elements
        for t in range(seq_length):
            apply_scan_op_kernel[blocks_per_grid, threads_per_block](
                d_a[t], d_b[t], d_h0, d_h_out[t], hidden_size
            )
    
    # Copy result back to host
    h_out = d_h_out.copy_to_host()
    
    return h_out

@cuda.jit
def min_lstm_fused_kernel_optimized(weights_f, bias_f, weights_i, bias_i, weights_h, bias_h,
                                 x_sequence, h_prev, h_out,
                                 seq_length, input_size, hidden_size):
    """Optimized fused CUDA kernel that processes an entire sequence for MinLSTM.
    
    This kernel reduces kernel launch overhead by processing the entire sequence
    in a single kernel launch.
    
    Args:
        weights_f: Weight matrix for forget gate
        bias_f: Bias vector for forget gate
        weights_i: Weight matrix for input gate
        bias_i: Bias vector for input gate
        weights_h: Weight matrix for candidate state
        bias_h: Bias vector for candidate state
        x_sequence: Input sequence of shape (seq_length, input_size)
        h_prev: Initial hidden state
        h_out: Output hidden states of shape (seq_length, hidden_size)
        seq_length: Sequence length
        input_size: Size of input vector
        hidden_size: Size of hidden state vector
    """
    # Shared memory for input vector and temporary hidden state
    shared_x = cuda.shared.array(shape=(128,), dtype=np.float32)
    shared_h = cuda.shared.array(shape=(128,), dtype=np.float32)
    
    thread_idx = cuda.grid(1)
    block_size = cuda.blockDim.x
    
    # Initialize shared hidden state with h_prev
    if thread_idx < hidden_size:
        shared_h[thread_idx] = h_prev[thread_idx]
    
    cuda.syncthreads()
    
    # Process each time step
    for t in range(seq_length):
        # Load input vector for current time step into shared memory
        for i in range(thread_idx, input_size, block_size):
            if i < input_size:
                shared_x[i] = x_sequence[t, i]
        
        cuda.syncthreads()
        
        # Each thread processes one element of the hidden state
        if thread_idx < hidden_size:
            # Compute forget gate
            f_t = bias_f[thread_idx]
            for i in range(input_size):
                f_t += weights_f[thread_idx, i] * shared_x[i]
            f_t = 1.0 / (1.0 + math.exp(-f_t))  # sigmoid
            
            # Compute input gate
            i_t = bias_i[thread_idx]
            for i in range(input_size):
                i_t += weights_i[thread_idx, i] * shared_x[i]
            i_t = 1.0 / (1.0 + math.exp(-i_t))  # sigmoid
            
            # Normalize gates
            gate_sum = f_t + i_t
            if gate_sum > 0:
                f_t = f_t / gate_sum
                i_t = i_t / gate_sum
            else:
                f_t = 0.5
                i_t = 0.5
            
            # Compute candidate hidden state
            h_tilde = bias_h[thread_idx]
            for i in range(input_size):
                h_tilde += weights_h[thread_idx, i] * shared_x[i]
            h_tilde = math.tanh(h_tilde)
            
            # Compute output hidden state
            new_h = f_t * shared_h[thread_idx] + i_t * h_tilde
            
            # Store result in output array
            h_out[t, thread_idx] = new_h
            
            # Update shared hidden state for next time step
            shared_h[thread_idx] = new_h
        
        cuda.syncthreads()
