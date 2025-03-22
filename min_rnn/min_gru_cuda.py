import numpy as np
import time
import math
from numba import cuda
from .utils import LinearLayer, sigmoid
from .cuda_utils import (
    CUDA_AVAILABLE, min_gru_forward_kernel, min_gru_extract_scan_params_kernel,
    cuda_parallel_scan, apply_scan_op_kernel, compose_scan_ops_kernel,
    get_optimal_grid_block_size, min_gru_forward_kernel_batch, min_gru_fused_kernel,
    cuda_parallel_scan_optimized, create_pinned_array, linear_forward_kernel_optimized
)

class MinGRUCellCUDA:
    """CUDA-accelerated Minimal Gated Recurrent Unit (minGRU) implementation.
    
    The minGRU model simplifies the standard GRU by removing the dependence on 
    the previous hidden state in the gate computations:
    
    - Update Gate: z_t = σ(Linear_z(x_t))
    - Candidate State: h_tilde = Linear_h(x_t)
    - Hidden State: h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h_tilde
    """
    
    def __init__(self, input_size, hidden_size):
        """Initialize a MinGRU cell.
        
        Args:
            input_size (int): Size of the input vector
            hidden_size (int): Size of the hidden state vector
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Initialize the linear layers
        self.linear_z = LinearLayer(input_size, hidden_size)
        self.linear_h = LinearLayer(input_size, hidden_size)
        
        # Check if CUDA is available
        self.cuda_available = CUDA_AVAILABLE
        
        # Pre-allocate device memory for weights and biases if CUDA is available
        if self.cuda_available:
            # Convert weights to float32 for better CUDA performance
            self.d_weights_z = cuda.to_device(self.linear_z.weights.astype(np.float32))
            self.d_bias_z = cuda.to_device(self.linear_z.bias.astype(np.float32))
            self.d_weights_h = cuda.to_device(self.linear_h.weights.astype(np.float32))
            self.d_bias_h = cuda.to_device(self.linear_h.bias.astype(np.float32))
            
            # Pre-allocate reusable device memory for common operations
            self.d_h_temp = cuda.device_array(hidden_size, dtype=np.float32)
            
            # Create stream for asynchronous operations
            self.stream = cuda.stream()
            
            # Pre-allocate pinned memory for faster transfers
            self.p_h_temp = create_pinned_array(hidden_size)
    
    def forward(self, x_t, h_prev):
        """MinGRU forward pass using CUDA.
        
        Args:
            x_t (numpy.ndarray): Input vector of shape (input_size,)
            h_prev (numpy.ndarray): Previous hidden state of shape (hidden_size,)
            
        Returns:
            numpy.ndarray: New hidden state of shape (hidden_size,)
        """
        if not self.cuda_available:
            # Fallback to CPU implementation if CUDA is not available
            return self._forward_cpu(x_t, h_prev)
        
        # Convert inputs to float32 for better CUDA performance
        x_t = x_t.astype(np.float32)
        h_prev = h_prev.astype(np.float32)
        
        # Transfer inputs to device memory
        d_x_t = cuda.to_device(x_t, stream=self.stream)
        d_h_prev = cuda.to_device(h_prev, stream=self.stream)
        d_h_t = cuda.device_array(self.hidden_size, dtype=np.float32, stream=self.stream)
        
        # Determine optimal grid and block size for better occupancy
        grid_size, block_size = get_optimal_grid_block_size(self.hidden_size)
        
        # Execute the optimized batch kernel
        min_gru_forward_kernel_batch[(grid_size,), (block_size,), self.stream](
            self.d_weights_z, self.d_bias_z, self.d_weights_h, self.d_bias_h,
            d_x_t, d_h_prev, d_h_t,
            self.input_size, self.hidden_size
        )
        
        # Transfer result back to host using pinned memory for faster transfer
        h_t = np.empty(self.hidden_size, dtype=np.float32)
        d_h_t.copy_to_host(h_t, stream=self.stream)
        self.stream.synchronize()
        
        return h_t
    
    def _forward_cpu(self, x_t, h_prev):
        """CPU implementation of MinGRU forward pass.
        
        Args:
            x_t (numpy.ndarray): Input vector of shape (input_size,)
            h_prev (numpy.ndarray): Previous hidden state of shape (hidden_size,)
            
        Returns:
            numpy.ndarray: New hidden state of shape (hidden_size,)
        """
        # Compute update gate: z_t = σ(Linear_z(x_t))
        z_t = sigmoid(self.linear_z.forward(x_t))
        
        # Compute candidate hidden state: h_tilde = Linear_h(x_t)
        h_tilde = np.tanh(self.linear_h.forward(x_t))
        
        # Compute hidden state: h_t = (1 - z_t) * h_prev + z_t * h_tilde
        h_t = (1 - z_t) * h_prev + z_t * h_tilde
        
        return h_t

    def process_sequence(self, x_seq, h_0=None):
        """Process an input sequence using MinGRU.
        
        Args:
            x_seq (numpy.ndarray): Input sequence of shape (seq_length, input_size)
            h_0 (numpy.ndarray, optional): Initial hidden state of shape (hidden_size,).
                If None, initialized to zeros.
                
        Returns:
            numpy.ndarray: Output hidden states of shape (seq_length, hidden_size)
        """
        seq_length = x_seq.shape[0]
        
        # Initialize hidden state if not provided
        if h_0 is None:
            h_0 = np.zeros(self.hidden_size)
        
        if not self.cuda_available:
            # Fallback to CPU implementation if CUDA is not available
            return self._process_sequence_cpu(x_seq, h_0)
        
        # For small sequences, use direct computation
        if seq_length <= 8:
            return self._process_sequence_direct_cuda(x_seq, h_0)
        
        # For medium sequences, use optimized fused kernel
        if seq_length <= 64:
            return self._process_sequence_fused_cuda(x_seq, h_0)
        
        # For large sequences, use optimized parallel scan
        # Extract scan parameters: a and b for h_t = a_t * h_{t-1} + b_t
        a, b = self._extract_scan_params_cuda(x_seq)
        
        # Perform parallel scan to compute all hidden states
        return cuda_parallel_scan_optimized(seq_length, 1, self.hidden_size, a, b, h_0)
    
    def _process_sequence_direct_cuda(self, x_seq, h_0):
        """Process a short sequence directly step by step using CUDA.
        
        This is more efficient for very short sequences.
        
        Args:
            x_seq (numpy.ndarray): Input sequence of shape (seq_length, input_size)
            h_0 (numpy.ndarray): Initial hidden state
            
        Returns:
            numpy.ndarray: Output hidden states
        """
        seq_length = x_seq.shape[0]
        h_seq = np.zeros((seq_length, self.hidden_size), dtype=np.float32)
        
        # Convert to float32 for better CUDA performance
        x_seq = x_seq.astype(np.float32)
        h_t = h_0.astype(np.float32)
        
        # Process each time step
        for t in range(seq_length):
            h_t = self.forward(x_seq[t], h_t)
            h_seq[t] = h_t
        
        return h_seq
    
    def _process_sequence_fused_cuda(self, x_seq, h_0):
        """Process a medium-length sequence using a fused kernel.
        
        This reduces kernel launch overhead by processing the entire sequence in one kernel.
        
        Args:
            x_seq (numpy.ndarray): Input sequence of shape (seq_length, input_size)
            h_0 (numpy.ndarray): Initial hidden state
            
        Returns:
            numpy.ndarray: Output hidden states
        """
        seq_length = x_seq.shape[0]
        
        # Convert inputs to float32
        x_seq = x_seq.astype(np.float32)
        h_0 = h_0.astype(np.float32)
        
        # Transfer inputs to device
        d_x_seq = cuda.to_device(x_seq, stream=self.stream)
        d_h_0 = cuda.to_device(h_0, stream=self.stream)
        d_h_seq = cuda.device_array((seq_length, self.hidden_size), dtype=np.float32, stream=self.stream)
        
        # Determine optimal grid and block size
        grid_size, block_size = get_optimal_grid_block_size(self.hidden_size)
        
        # Launch fused kernel to process the entire sequence
        min_gru_fused_kernel[(grid_size,), (block_size,), self.stream](
            self.d_weights_z, self.d_bias_z, self.d_weights_h, self.d_bias_h,
            d_x_seq, d_h_0, d_h_seq,
            seq_length, self.input_size, self.hidden_size
        )
        
        # Transfer results back to host
        h_seq = np.empty((seq_length, self.hidden_size), dtype=np.float32)
        d_h_seq.copy_to_host(h_seq, stream=self.stream)
        self.stream.synchronize()
        
        return h_seq

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
        min_gru_extract_scan_params_kernel[
            (grid_size_x, grid_size_y), 
            (block_size_x, block_size_y),
            self.stream
        ](
            self.d_weights_z, self.d_bias_z, self.d_weights_h, self.d_bias_h,
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

    def process_sequence_parallel(self, x, h0, seq_length=None):
        """Process a full sequence with MinGRU (parallel mode) using CUDA.
        
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
            return self.process_sequence(x, h0, seq_length)
        
        if not self.cuda_available:
            # Fall back to CPU implementation if CUDA is not available
            from .min_gru import MinGRUCell
            cpu_cell = MinGRUCell(self.input_size, self.hidden_size)
            cpu_cell.linear_z = self.linear_z
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
        """Prepare data for parallel scan by extracting a and b coefficients using CUDA.
        
        Args:
            x (numpy.ndarray): Input sequence of shape (seq_length, input_size)
            seq_length (int, optional): Number of time steps. If None, inferred from x.
            
        Returns:
            tuple: (a, b) where:
                a (numpy.ndarray): Coefficient a of shape (seq_length, hidden_size)
                b (numpy.ndarray): Coefficient b of shape (seq_length, hidden_size)
        """
        if seq_length is None:
            seq_length = len(x)
        
        if not self.cuda_available:
            # Fall back to CPU implementation if CUDA is not available
            from .min_gru import MinGRUCell
            cpu_cell = MinGRUCell(self.input_size, self.hidden_size)
            cpu_cell.linear_z = self.linear_z
            cpu_cell.linear_h = self.linear_h
            return cpu_cell.extract_scan_params(x, seq_length)
        
        # Convert input to float32
        x = x.astype(np.float32)
        
        # Initialize a and b arrays
        a = np.zeros((seq_length, self.hidden_size), dtype=np.float32)
        b = np.zeros((seq_length, self.hidden_size), dtype=np.float32)
        
        # Copy data to device
        d_x = cuda.to_device(x)
        d_a = cuda.to_device(a)
        d_b = cuda.to_device(b)
        
        # Configure kernel for 2D grid
        threads_per_block_x = min(16, self.hidden_size)
        threads_per_block_y = min(16, seq_length)
        threads_per_block = (threads_per_block_x, threads_per_block_y)
        
        blocks_per_grid = (
            max(1, (self.hidden_size + threads_per_block[0] - 1) // threads_per_block[0]),
            max(1, (seq_length + threads_per_block[1] - 1) // threads_per_block[1])
        )
        
        # Launch kernel
        min_gru_extract_scan_params_kernel[blocks_per_grid, threads_per_block](
            self.d_weights_z, self.d_bias_z,
            self.d_weights_h, self.d_bias_h,
            d_x, d_a, d_b,
            seq_length, self.hidden_size, self.input_size
        )
        
        # Copy result back to host
        a = d_a.copy_to_host()
        b = d_b.copy_to_host()
        
        return a, b
