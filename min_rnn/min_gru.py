import numpy as np
from .utils import LinearLayer, sigmoid

class MinGRUCell:
    """Minimal Gated Recurrent Unit (minGRU) implementation.
    
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
    
    def forward(self, x_t, h_prev):
        """MinGRU forward pass (sequential mode).
        
        Args:
            x_t (numpy.ndarray): Input vector of shape (input_size,)
            h_prev (numpy.ndarray): Previous hidden state of shape (hidden_size,)
            
        Returns:
            numpy.ndarray: New hidden state of shape (hidden_size,)
        """
        # Compute update gate: z_t = sigmoid(Linear_z(x_t))
        z_t = sigmoid(self.linear_z.forward(x_t))
        
        # Compute candidate hidden state: h_tilde = Linear_h(x_t)
        h_tilde = self.linear_h.forward(x_t)
        
        # Compute h_t = (1 - z_t) * h_prev + z_t * h_tilde
        h_t = (1.0 - z_t) * h_prev + z_t * h_tilde
        
        return h_t
    
    def process_sequence(self, x, h0, seq_length=None):
        """Process a full sequence with MinGRU (sequential mode).
        
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
            
            # Compute update gate: z_t = sigmoid(Linear_z(x_t))
            z_t = sigmoid(self.linear_z.forward(x_t))
            
            # Compute candidate hidden state: h_tilde = Linear_h(x_t)
            h_tilde = self.linear_h.forward(x_t)
            
            # Compute a_t = (1 - z_t)
            a[t] = 1.0 - z_t
            
            # Compute b_t = z_t * h_tilde
            b[t] = z_t * h_tilde
        
        return a, b
    
    def process_sequence_parallel(self, x, h0, seq_length=None):
        """Process a full sequence with MinGRU (parallel mode).
        
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

def compose_scan_ops(op1_a, op1_b, op2_a, op2_b):
    """Combine two scan operations.
    
    op_out = op2 ○ op1 (composition)
    Where (a, b) ○ (c, d) = (a*c, a*d + b)
    
    Args:
        op1_a (numpy.ndarray): Coefficient a of first operation
        op1_b (numpy.ndarray): Coefficient b of first operation
        op2_a (numpy.ndarray): Coefficient a of second operation
        op2_b (numpy.ndarray): Coefficient b of second operation
        
    Returns:
        tuple: (a_out, b_out) where:
            a_out (numpy.ndarray): Coefficient a of composed operation
            b_out (numpy.ndarray): Coefficient b of composed operation
    """
    # op_out.a = op2.a * op1.a
    a_out = op2_a * op1_a
    
    # op_out.b = op2.a * op1.b + op2.b
    b_out = op2_a * op1_b + op2_b
    
    return a_out, b_out

def apply_scan_op(op_a, op_b, h_in):
    """Apply scan operation to hidden state.
    
    h_out = op.a * h_in + op.b
    
    Args:
        op_a (numpy.ndarray): Coefficient a of operation
        op_b (numpy.ndarray): Coefficient b of operation
        h_in (numpy.ndarray): Input hidden state
        
    Returns:
        numpy.ndarray: Output hidden state
    """
    # h_out = op.a * h_in + op.b
    h_out = op_a * h_in + op_b
    
    return h_out

def sequential_scan(seq_length, batch_size, hidden_size, a, b, h0):
    """Sequential scan for comparison and for small sequence lengths.
    
    Args:
        seq_length (int): Number of time steps
        batch_size (int): Batch size (not used in this implementation)
        hidden_size (int): Size of the hidden state vector
        a (numpy.ndarray): Coefficient a of shape (seq_length, hidden_size)
        b (numpy.ndarray): Coefficient b of shape (seq_length, hidden_size)
        h0 (numpy.ndarray): Initial hidden state of shape (hidden_size,)
        
    Returns:
        numpy.ndarray: Output hidden states of shape (seq_length, hidden_size)
    """
    # Initialize output hidden states
    h_out = np.zeros((seq_length, hidden_size))
    
    # Initialize h_prev with h0
    h_prev = h0.copy()
    
    # Process each time step sequentially
    for t in range(seq_length):
        # Get a and b for current time step
        a_t = a[t]
        b_t = b[t]
        
        # Compute h_t = a_t * h_prev + b_t
        h_curr = a_t * h_prev + b_t
        
        # Store hidden state in output
        h_out[t] = h_curr
        
        # Update h_prev for next time step
        h_prev = h_curr
    
    return h_out

def parallel_scan(seq_length, batch_size, hidden_size, a, b, h0):
    """CPU implementation of the parallel scan algorithm.
    
    a[t] * h[t-1] + b[t] = h[t]
    
    Args:
        seq_length (int): Number of time steps
        batch_size (int): Batch size (not used in this implementation)
        hidden_size (int): Size of the hidden state vector
        a (numpy.ndarray): Coefficient a of shape (seq_length, hidden_size)
        b (numpy.ndarray): Coefficient b of shape (seq_length, hidden_size)
        h0 (numpy.ndarray): Initial hidden state of shape (hidden_size,)
        
    Returns:
        numpy.ndarray: Output hidden states of shape (seq_length, hidden_size)
    """
    # Base case: for short sequences, use sequential scan
    if seq_length <= 4:
        return sequential_scan(seq_length, batch_size, hidden_size, a, b, h0)
    
    # Initialize operations with input a and b values
    ops_a = a.copy()
    ops_b = b.copy()
    
    # Perform parallel scan using tree reduction (up-sweep phase)
    offset = 1
    # Building the tree bottom-up
    d = seq_length >> 1
    while d > 0:
        # In each level, we combine pairs of operations at distance 'offset'
        for t in range(0, seq_length, 2 * offset):
            if t + offset < seq_length:
                # Compose operations: ops[t + offset] = ops[t + offset] ○ ops[t]
                ops_a[t + offset], ops_b[t + offset] = compose_scan_ops(
                    ops_a[t], ops_b[t], ops_a[t + offset], ops_b[t + offset]
                )
        offset *= 2
        d = d >> 1
    
    # Apply operations and compute hidden states (down-sweep phase)
    # Initialize hidden states array
    h_states = np.zeros((seq_length + 1, hidden_size))
    
    # h_states[0] = h0
    h_states[0] = h0
    
    # Compute all hidden states using the scan operations
    for t in range(seq_length):
        h_states[t + 1] = apply_scan_op(ops_a[t], ops_b[t], h_states[t])
    
    # Return all hidden states except the initial h0
    return h_states[1:]
