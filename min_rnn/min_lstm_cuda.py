import numpy as np
import time
from numba import cuda
import numba
import math
from .utils import LinearLayer, sigmoid
from .cuda_utils import (
    CUDA_AVAILABLE, min_lstm_forward_kernel, min_lstm_extract_scan_params_kernel,
    cuda_parallel_scan, apply_scan_op_kernel, compose_scan_ops_kernel,
    get_optimal_grid_block_size, min_lstm_forward_kernel_batch, 
    cuda_parallel_scan_optimized, create_pinned_array, linear_forward_kernel_optimized,
    min_lstm_forward_kernel_optimized, min_lstm_fused_kernel_optimized
)

# Define a fused kernel for MinLSTM sequence processing
@cuda.jit
def min_lstm_fused_kernel(weights_f, bias_f, weights_i, bias_i, weights_h, bias_h,
                        x_sequence, h_prev, h_out,
                        seq_length, input_size, hidden_size):
    """Fused CUDA kernel that processes an entire sequence for MinLSTM.
    
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
    idx = cuda.grid(1)
    
    if idx < hidden_size:
        # Load initial hidden state
        h_t = h_prev[idx]
        
        # Process the entire sequence
        for t in range(seq_length):
            # Compute forget gate
            f_t = bias_f[idx]
            for i in range(input_size):
                f_t += weights_f[idx, i] * x_sequence[t, i]
            f_t = 1.0 / (1.0 + math.exp(-f_t))  # sigmoid
            
            # Compute input gate
            i_t = bias_i[idx]
            for i in range(input_size):
                i_t += weights_i[idx, i] * x_sequence[t, i]
            i_t = 1.0 / (1.0 + math.exp(-i_t))  # sigmoid
            
            # Normalize gates
            gate_sum = f_t + i_t
            if gate_sum > 0:
                f_t = f_t / gate_sum
                i_t = i_t / gate_sum
            
            # Compute candidate hidden state
            h_tilde = bias_h[idx]
            for i in range(input_size):
                h_tilde += weights_h[idx, i] * x_sequence[t, i]
            h_tilde = math.tanh(h_tilde)
            
            # Compute new hidden state
            h_t = f_t * h_t + i_t * h_tilde
            
            # Store result
            h_out[t, idx] = h_t

class MinLSTMCellCUDA:
    """CUDA-accelerated Minimal Long Short-Term Memory (minLSTM) implementation.
    
    The minLSTM model simplifies the standard LSTM by removing the dependence on 
    the previous hidden state in the gate computations:
    
    - Forget Gate: f_t = σ(Linear_f(x_t))
    - Input Gate: i_t = σ(Linear_i(x_t))
    - Candidate State: h_tilde = tanh(Linear_h(x_t))
    - Normalized Gates: f'_t = f_t / (f_t + i_t), i'_t = i_t / (f_t + i_t)
    - Hidden State: h_t = f'_t ⊙ h_{t-1} + i'_t ⊙ h_tilde
    """
    
    def __init__(self, input_size, hidden_size):
        """Initialize a MinLSTM cell.
        
        Args:
            input_size (int): Size of the input vector
            hidden_size (int): Size of the hidden state vector
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Initialize the linear layers
        self.linear_f = LinearLayer(input_size, hidden_size)
        self.linear_i = LinearLayer(input_size, hidden_size)
        self.linear_h = LinearLayer(input_size, hidden_size)
        
        # Check if CUDA is available
        self.cuda_available = CUDA_AVAILABLE
        
        # Pre-allocate device memory for weights and biases if CUDA is available
        if self.cuda_available:
            # Convert weights to float32 for better CUDA performance
            self.d_weights_f = cuda.to_device(self.linear_f.weights.astype(np.float32))
            self.d_bias_f = cuda.to_device(self.linear_f.bias.astype(np.float32))
            self.d_weights_i = cuda.to_device(self.linear_i.weights.astype(np.float32))
            self.d_bias_i = cuda.to_device(self.linear_i.bias.astype(np.float32))
            self.d_weights_h = cuda.to_device(self.linear_h.weights.astype(np.float32))
            self.d_bias_h = cuda.to_device(self.linear_h.bias.astype(np.float32))
            
            # Pre-allocate reusable device memory for common operations
            self.d_h_temp = cuda.device_array(hidden_size, dtype=np.float32)
            
            # Create stream for asynchronous operations
            self.stream = cuda.stream()
            
            # Pre-allocate pinned memory for faster transfers
            self.p_h_temp = create_pinned_array(hidden_size)
    
    def forward(self, x_t, h_prev):
        """Forward pass for a single time step.
        
        Args:
            x_t: Input at time step t
            h_prev: Hidden state at time step t-1
            
        Returns:
            numpy.ndarray: Hidden state at time step t
        """
        if not self.cuda_available:
            # Fallback to CPU implementation if CUDA is not available
            return self._forward_cpu(x_t, h_prev)
            
        # Convert inputs to float32 for better CUDA performance
        x_t = x_t.astype(np.float32)
        h_prev = h_prev.astype(np.float32)
        
        # Allocate device memory using pinned memory for faster transfers
        d_x_t = cuda.to_device(x_t, stream=self.stream)
        d_h_prev = cuda.to_device(h_prev, stream=self.stream)
        d_h_t = cuda.device_array(self.hidden_size, dtype=np.float32, stream=self.stream)
        
        # Determine optimal thread block size for better occupancy
        grid_size, block_size = get_optimal_grid_block_size(self.hidden_size)
        
        # Execute optimized kernel with shared memory
        min_lstm_forward_kernel_optimized[(grid_size,), (block_size,), self.stream](
            self.d_weights_f, self.d_bias_f,
            self.d_weights_i, self.d_bias_i,
            self.d_weights_h, self.d_bias_h,
            d_x_t, d_h_prev, d_h_t,
            self.input_size, self.hidden_size
        )
        
        # Copy result back to host using pinned memory for faster transfers
        h_t = np.empty(self.hidden_size, dtype=np.float32)
        d_h_t.copy_to_host(h_t, stream=self.stream)
        self.stream.synchronize()
        
        return h_t
    
    def _forward_cpu(self, x_t, h_prev):
        """CPU implementation of MinLSTM forward pass.
        
        Args:
            x_t (numpy.ndarray): Input vector of shape (input_size,)
            h_prev (numpy.ndarray): Previous hidden state of shape (hidden_size,)
            
        Returns:
            numpy.ndarray: New hidden state of shape (hidden_size,)
        """
        # Compute forget gate: f_t = σ(Linear_f(x_t))
        f_t = sigmoid(self.linear_f.forward(x_t))
        
        # Compute input gate: i_t = σ(Linear_i(x_t))
        i_t = sigmoid(self.linear_i.forward(x_t))
        
        # Normalize gates: f'_t = f_t / (f_t + i_t), i'_t = i_t / (f_t + i_t)
        gate_sum = f_t + i_t
        f_t = np.divide(f_t, gate_sum, out=np.zeros_like(f_t), where=gate_sum > 0)
        i_t = np.divide(i_t, gate_sum, out=np.zeros_like(i_t), where=gate_sum > 0)
        
        # Compute candidate hidden state: h_tilde = tanh(Linear_h(x_t))
        h_tilde = np.tanh(self.linear_h.forward(x_t))
        
        # Compute hidden state: h_t = f_t * h_prev + i_t * h_tilde
        h_t = f_t * h_prev + i_t * h_tilde
        
        return h_t

    def process_sequence(self, x_seq, h_0):
        """Process a sequence of inputs.
        
        Args:
            x_seq: Input sequence of shape (seq_length, input_size)
            h_0: Initial hidden state of shape (hidden_size,)
            
        Returns:
            numpy.ndarray: Output hidden states of shape (seq_length, hidden_size)
        """
        seq_length = x_seq.shape[0]
        
        if not self.cuda_available:
            # Fallback to CPU implementation if CUDA is not available
            return self._process_sequence_direct(x_seq, h_0)
            
        # For small sequences, use direct computation
        if seq_length <= 16:
            return self._process_sequence_direct_cuda(x_seq, h_0)
            
        # For medium sequences, use fused kernel
        if seq_length <= 128:
            return self._process_sequence_fused_cuda(x_seq, h_0)
            
        # For large sequences, use parallel scan
        # Extract scan parameters: a and b for h_t = a_t * h_{t-1} + b_t
        a, b = self._extract_scan_params_cuda(x_seq)
        
        # Perform parallel scan to compute all hidden states
        return cuda_parallel_scan_optimized(seq_length, 1, self.hidden_size, a, b, h_0)
    
    def _process_sequence_direct_cuda(self, x_seq, h_0):
        """Process a sequence directly using CUDA.
        
        Args:
            x_seq: Input sequence of shape (seq_length, input_size)
            h_0: Initial hidden state of shape (hidden_size,)
            
        Returns:
            numpy.ndarray: Output hidden states of shape (seq_length, hidden_size)
        """
        seq_length = x_seq.shape[0]
        h_out = np.zeros((seq_length, self.hidden_size), dtype=np.float32)
        
        # Convert to float32 for better CUDA performance
        x_seq = x_seq.astype(np.float32)
        h_t = h_0.astype(np.float32)
        
        # Process each time step
        for t in range(seq_length):
            h_t = self.forward(x_seq[t], h_t)
            h_out[t] = h_t
        
        return h_out
    
    def _process_sequence_direct(self, x_seq, h_0):
        """Process a sequence using direct computation.
        
        Args:
            x_seq: Input sequence of shape (seq_length, input_size)
            h_0: Initial hidden state of shape (hidden_size,)
            
        Returns:
            numpy.ndarray: Output hidden states of shape (seq_length, hidden_size)
        """
        seq_length = x_seq.shape[0]
        h_out = np.zeros((seq_length, self.hidden_size), dtype=np.float32)
        
        h_t = h_0.copy()
        for t in range(seq_length):
            h_t = self._forward_cpu(x_seq[t], h_t)
            h_out[t] = h_t
        
        return h_out
    
    def _process_sequence_fused_cuda(self, x_seq, h_0):
        """Process a sequence using a fused kernel.
        
        This reduces kernel launch overhead by processing the entire sequence in one kernel.
        
        Args:
            x_seq: Input sequence of shape (seq_length, input_size)
            h_0: Initial hidden state of shape (hidden_size,)
            
        Returns:
            numpy.ndarray: Output hidden states of shape (seq_length, hidden_size)
        """
        seq_length = x_seq.shape[0]
        
        # Convert inputs to float32 for better CUDA performance
        x_seq = x_seq.astype(np.float32)
        h_0 = h_0.astype(np.float32)
        
        # Transfer inputs to device using pinned memory for faster transfers
        d_x_seq = cuda.to_device(x_seq, stream=self.stream)
        d_h_0 = cuda.to_device(h_0, stream=self.stream)
        d_h_out = cuda.device_array((seq_length, self.hidden_size), dtype=np.float32, stream=self.stream)
        
        # Determine optimal thread block size for better occupancy
        grid_size, block_size = get_optimal_grid_block_size(self.hidden_size)
        
        # Launch fused kernel to process the entire sequence
        min_lstm_fused_kernel_optimized[(grid_size,), (block_size,), self.stream](
            self.d_weights_f, self.d_bias_f, self.d_weights_i, self.d_bias_i,
            self.d_weights_h, self.d_bias_h,
            d_x_seq, d_h_0, d_h_out,
            seq_length, self.input_size, self.hidden_size
        )
        
        # Transfer results back to host using pinned memory for faster transfers
        h_out = np.empty((seq_length, self.hidden_size), dtype=np.float32)
        d_h_out.copy_to_host(h_out, stream=self.stream)
        self.stream.synchronize()
        
        return h_out
    
    def process_sequence_parallel(self, x, h0, seq_length=None):
        """Process a full sequence with MinLSTM (parallel mode) using CUDA.
        
        Args:
            x (numpy.ndarray): Input sequence of shape (seq_length, input_size)
            h0 (numpy.ndarray): Initial hidden state of shape (hidden_size,)
            seq_length (int, optional): Number of time steps. If None, inferred from x.
            
        Returns:
            numpy.ndarray: Output hidden states of shape (seq_length, hidden_size)
        """
        if seq_length is None:
            seq_length = len(x)
        
        # For very short sequences, use the sequential implementation
        # which may be faster due to less overhead
        if seq_length <= 16:
            return self.process_sequence(x, h0)
        
        if not self.cuda_available:
            # Fall back to CPU implementation if CUDA is not available
            from .min_lstm import MinLSTMCell
            cpu_cell = MinLSTMCell(self.input_size, self.hidden_size)
            cpu_cell.linear_f = self.linear_f
            cpu_cell.linear_i = self.linear_i
            cpu_cell.linear_h = self.linear_h
            return cpu_cell.process_sequence_parallel(x, h0, seq_length)
        
        # Convert inputs to float32
        x = x.astype(np.float32)
        h0 = h0.astype(np.float32)
        
        # Extract a and b coefficients for parallel scan
        a, b = self.extract_scan_params(x, seq_length)
        
        # Run parallel scan
        h_out = cuda_parallel_scan_optimized(seq_length, 1, self.hidden_size, a, b, h0)
        
        return h_out
    
    def extract_scan_params(self, x, seq_length=None):
        """Extract parameters for parallel scan.
        
        Args:
            x (numpy.ndarray): Input sequence of shape (seq_length, input_size)
            seq_length (int, optional): Sequence length. If None, inferred from x.
            
        Returns:
            tuple: (a, b) coefficients for scan operation h_t = a_t * h_{t-1} + b_t
        """
        if seq_length is None:
            seq_length = len(x)
            
        return self._extract_scan_params_cuda(x)

    def _extract_scan_params_cuda(self, x_seq):
        """Extract parameters for parallel scan using CUDA.
        
        Args:
            x_seq (numpy.ndarray): Input sequence of shape (seq_length, input_size)
            
        Returns:
            tuple: (a, b) coefficients for scan operation h_t = a_t * h_{t-1} + b_t
        """
        seq_length = x_seq.shape[0]
        
        # Convert to float32 for better CUDA performance
        x_seq = x_seq.astype(np.float32)
        
        # Allocate memory for results
        a = np.zeros((seq_length, self.hidden_size), dtype=np.float32)
        b = np.zeros((seq_length, self.hidden_size), dtype=np.float32)
        
        # Transfer data to device
        d_x_seq = cuda.to_device(x_seq, stream=self.stream)
        d_a = cuda.device_array((seq_length, self.hidden_size), dtype=np.float32, stream=self.stream)
        d_b = cuda.device_array((seq_length, self.hidden_size), dtype=np.float32, stream=self.stream)
        
        # Determine optimal grid and block size for 2D operation
        block_size_x = min(16, self.hidden_size)
        block_size_y = min(16, seq_length)
        grid_size_x = (self.hidden_size + block_size_x - 1) // block_size_x
        grid_size_y = (seq_length + block_size_y - 1) // block_size_y
        
        # Launch kernel to extract scan parameters
        min_lstm_extract_scan_params_kernel[
            (grid_size_x, grid_size_y), 
            (block_size_x, block_size_y),
            self.stream
        ](
            self.d_weights_f, self.d_bias_f, self.d_weights_i, self.d_bias_i,
            self.d_weights_h, self.d_bias_h,
            d_x_seq, d_a, d_b,
            seq_length, self.hidden_size, self.input_size
        )
        
        # Transfer results back to host
        d_a.copy_to_host(a, stream=self.stream)
        d_b.copy_to_host(b, stream=self.stream)
        self.stream.synchronize()
        
        return a, b
    
    def _process_sequence_cpu(self, x_seq, h_0):
        """Process a sequence using CPU.
        
        Args:
            x_seq (numpy.ndarray): Input sequence of shape (seq_length, input_size)
            h_0 (numpy.ndarray): Initial hidden state
            
        Returns:
            numpy.ndarray: Output hidden states
        """
        seq_length = x_seq.shape[0]
        h_seq = np.zeros((seq_length, self.hidden_size))
        
        h_t = h_0.copy()
        for t in range(seq_length):
            h_t = self._forward_cpu(x_seq[t], h_t)
            h_seq[t] = h_t
        
        return h_seq
