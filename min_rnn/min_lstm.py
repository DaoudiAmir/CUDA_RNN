import numpy as np
from .utils import LinearLayer, sigmoid
from .min_gru import compose_scan_ops, apply_scan_op, sequential_scan, parallel_scan

class MinLSTMCell:
    """Minimal Long Short-Term Memory (minLSTM) implementation.
    
    The minLSTM model simplifies the standard LSTM by removing the dependence on 
    the previous hidden state in the gate computations:
    
    - Forget Gate: f_t = σ(Linear_f(x_t))
    - Input Gate: i_t = σ(Linear_i(x_t))
    - Candidate State: h_tilde = Linear_h(x_t)
    - Gate Normalization: f'_t = f_t / (f_t + i_t), i'_t = i_t / (f_t + i_t)
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
    
    def forward(self, x_t, h_prev):
        """MinLSTM forward pass (sequential mode).
        
        Args:
            x_t (numpy.ndarray): Input vector of shape (input_size,)
            h_prev (numpy.ndarray): Previous hidden state of shape (hidden_size,)
            
        Returns:
            numpy.ndarray: New hidden state of shape (hidden_size,)
        """
        # Compute forget gate: f_t = sigmoid(Linear_f(x_t))
        f_t = sigmoid(self.linear_f.forward(x_t))
        
        # Compute input gate: i_t = sigmoid(Linear_i(x_t))
        i_t = sigmoid(self.linear_i.forward(x_t))
        
        # Compute candidate hidden state: h_tilde = Linear_h(x_t)
        h_tilde = self.linear_h.forward(x_t)
        
        # Compute gate sum: gate_sum = f_t + i_t
        gate_sum = f_t + i_t
        
        # Normalize gates: f_prime = f_t / gate_sum, i_prime = i_t / gate_sum
        # Add epsilon for numerical stability
        epsilon = 1e-8
        f_prime = f_t / (gate_sum + epsilon)
        i_prime = i_t / (gate_sum + epsilon)
        
        # Compute h_t = f_prime * h_prev + i_prime * h_tilde
        h_t = f_prime * h_prev + i_prime * h_tilde
        
        return h_t
    
    def process_sequence(self, x, h0, seq_length=None):
        """Process a full sequence with MinLSTM (sequential mode).
        
        Args:
            x (numpy.ndarray): Input sequence of shape (seq_length, input_size)
            h0 (numpy.ndarray): Initial hidden state of shape (hidden_size,)
            seq_length (int, optional): Number of time steps. If None, inferred from x.
            
        Returns:
            numpy.ndarray: Output hidden states of shape (seq_length, hidden_size)
        """
        if seq_length is None:
            seq_length = len(x)
        
        # Initialize output hidden states
        h_out = np.zeros((seq_length, self.hidden_size))
        
        # Initialize h_prev with h0
        h_prev = h0.copy()
        
        # Process each time step
        for t in range(seq_length):
            # Get input for current time step
            x_t = x[t]
            
            # Compute hidden state for current time step
            h_curr = self.forward(x_t, h_prev)
            
            # Store hidden state in output
            h_out[t] = h_curr
            
            # Update h_prev for next time step
            h_prev = h_curr
        
        return h_out
    
    def extract_scan_params(self, x, seq_length=None):
        """Prepare data for parallel scan by extracting a and b coefficients.
        
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
        
        # Initialize a and b arrays
        a = np.zeros((seq_length, self.hidden_size))
        b = np.zeros((seq_length, self.hidden_size))
        
        # Compute a and b for each time step
        for t in range(seq_length):
            # Get input for current time step
            x_t = x[t]
            
            # Compute forget gate: f_t = sigmoid(Linear_f(x_t))
            f_t = sigmoid(self.linear_f.forward(x_t))
            
            # Compute input gate: i_t = sigmoid(Linear_i(x_t))
            i_t = sigmoid(self.linear_i.forward(x_t))
            
            # Compute candidate hidden state: h_tilde = Linear_h(x_t)
            h_tilde = self.linear_h.forward(x_t)
            
            # Compute gate sum: gate_sum = f_t + i_t
            gate_sum = f_t + i_t
            
            # Normalize gates: f_prime = f_t / gate_sum, i_prime = i_t / gate_sum
            # Add epsilon for numerical stability
            epsilon = 1e-8
            f_prime = f_t / (gate_sum + epsilon)
            i_prime = i_t / (gate_sum + epsilon)
            
            # Compute a_t = f_prime
            a[t] = f_prime
            
            # Compute b_t = i_prime * h_tilde
            b[t] = i_prime * h_tilde
        
        return a, b
    
    def process_sequence_parallel(self, x, h0, seq_length=None):
        """Process a full sequence with MinLSTM (parallel mode).
        
        Args:
            x (numpy.ndarray): Input sequence of shape (seq_length, input_size)
            h0 (numpy.ndarray): Initial hidden state of shape (hidden_size,)
            seq_length (int, optional): Number of time steps. If None, inferred from x.
            
        Returns:
            numpy.ndarray: Output hidden states of shape (seq_length, hidden_size)
        """
        if seq_length is None:
            seq_length = len(x)
        
        # Extract a and b coefficients for parallel scan
        a, b = self.extract_scan_params(x, seq_length)
        
        # Run parallel scan
        h_out = parallel_scan(seq_length, 1, self.hidden_size, a, b, h0)
        
        return h_out
