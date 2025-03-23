import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle
import matplotlib.path as mpath
import matplotlib.patches as mpatches

def create_minlstm_diagram():
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Define colors
    input_color = '#3498db'  # Blue
    gate_color = '#e74c3c'   # Red
    hidden_color = '#2ecc71' # Green
    arrow_color = '#7f8c8d'  # Gray
    norm_color = '#9b59b6'   # Purple
    
    # Define positions
    x_t_pos = (1, 6)
    h_prev_pos = (1, 2)
    f_t_pos = (4, 7)
    i_t_pos = (4, 5)
    h_tilde_pos = (4, 3)
    norm_pos = (6, 6)
    h_t_pos = (9, 4)
    
    # Draw nodes
    node_radius = 0.5
    
    # Input node
    input_circle = plt.Circle(x_t_pos, node_radius, color=input_color, alpha=0.7)
    ax.add_patch(input_circle)
    ax.text(x_t_pos[0], x_t_pos[1], r'$x_t$', ha='center', va='center', fontsize=14)
    
    # Previous hidden state
    h_prev_circle = plt.Circle(h_prev_pos, node_radius, color=hidden_color, alpha=0.7)
    ax.add_patch(h_prev_circle)
    ax.text(h_prev_pos[0], h_prev_pos[1], r'$h_{t-1}$', ha='center', va='center', fontsize=14)
    
    # Gate nodes
    f_t_circle = plt.Circle(f_t_pos, node_radius, color=gate_color, alpha=0.7)
    ax.add_patch(f_t_circle)
    ax.text(f_t_pos[0], f_t_pos[1], r'$f_t$', ha='center', va='center', fontsize=14)
    
    i_t_circle = plt.Circle(i_t_pos, node_radius, color=gate_color, alpha=0.7)
    ax.add_patch(i_t_circle)
    ax.text(i_t_pos[0], i_t_pos[1], r'$i_t$', ha='center', va='center', fontsize=14)
    
    # Normalization node
    norm_circle = plt.Circle(norm_pos, node_radius, color=norm_color, alpha=0.7)
    ax.add_patch(norm_circle)
    ax.text(norm_pos[0], norm_pos[1], r'Norm', ha='center', va='center', fontsize=12)
    
    # Candidate hidden state
    h_tilde_circle = plt.Circle(h_tilde_pos, node_radius, color=hidden_color, alpha=0.7)
    ax.add_patch(h_tilde_circle)
    ax.text(h_tilde_pos[0], h_tilde_pos[1], r'$\tilde{h}_t$', ha='center', va='center', fontsize=14)
    
    # Output hidden state
    h_t_circle = plt.Circle(h_t_pos, node_radius, color=hidden_color, alpha=0.7)
    ax.add_patch(h_t_circle)
    ax.text(h_t_pos[0], h_t_pos[1], r'$h_t$', ha='center', va='center', fontsize=14)
    
    # Draw arrows
    arrow_props = dict(arrowstyle='->', linewidth=1.5, color=arrow_color)
    
    # x_t to f_t
    ax.add_patch(FancyArrowPatch(x_t_pos, f_t_pos, connectionstyle='arc3,rad=0.1', **arrow_props))
    ax.text((x_t_pos[0] + f_t_pos[0])/2 + 0.3, (x_t_pos[1] + f_t_pos[1])/2, 'Linear + Sigmoid', fontsize=10)
    
    # x_t to i_t
    ax.add_patch(FancyArrowPatch(x_t_pos, i_t_pos, connectionstyle='arc3,rad=0', **arrow_props))
    ax.text((x_t_pos[0] + i_t_pos[0])/2 + 0.3, (x_t_pos[1] + i_t_pos[1])/2, 'Linear + Sigmoid', fontsize=10)
    
    # x_t to h_tilde
    ax.add_patch(FancyArrowPatch(x_t_pos, h_tilde_pos, connectionstyle='arc3,rad=-0.1', **arrow_props))
    ax.text((x_t_pos[0] + h_tilde_pos[0])/2 + 0.3, (x_t_pos[1] + h_tilde_pos[1])/2, 'Linear', fontsize=10)
    
    # f_t and i_t to normalization
    ax.add_patch(FancyArrowPatch(f_t_pos, norm_pos, connectionstyle='arc3,rad=0', **arrow_props))
    ax.add_patch(FancyArrowPatch(i_t_pos, norm_pos, connectionstyle='arc3,rad=0', **arrow_props))
    
    # Normalized gates to h_t
    ax.add_patch(FancyArrowPatch(norm_pos, h_t_pos, connectionstyle='arc3,rad=0.1', **arrow_props))
    ax.text((norm_pos[0] + h_t_pos[0])/2, (norm_pos[1] + h_t_pos[1])/2 + 0.3, r"$f^\prime_t, i^\prime_t$", fontsize=10)
    
    # h_prev to h_t
    ax.add_patch(FancyArrowPatch(h_prev_pos, h_t_pos, connectionstyle='arc3,rad=-0.2', **arrow_props))
    ax.text((h_prev_pos[0] + h_t_pos[0])/2 - 0.5, (h_prev_pos[1] + h_t_pos[1])/2 - 0.3, r"$f^\prime_t \odot$", fontsize=10)
    
    # h_tilde to h_t
    ax.add_patch(FancyArrowPatch(h_tilde_pos, h_t_pos, connectionstyle='arc3,rad=0.2', **arrow_props))
    ax.text((h_tilde_pos[0] + h_t_pos[0])/2 + 0.5, (h_tilde_pos[1] + h_t_pos[1])/2 - 0.3, r"$i^\prime_t \odot$", fontsize=10)
    
    # Add title and equations
    ax.set_title('MinLSTM Architecture', fontsize=16)
    
    # Add equations
    equations = [
        r'$f_t = \sigma(W_f x_t + b_f)$',
        r'$i_t = \sigma(W_i x_t + b_i)$',
        r'$\tilde{h}_t = W_h x_t + b_h$',
        r"$f^\prime_t = \frac{f_t}{f_t + i_t}, \quad i^\prime_t = \frac{i_t}{f_t + i_t}$",
        r"$h_t = f^\prime_t \odot h_{t-1} + i^\prime_t \odot \tilde{h}_t$"
    ]
    
    for i, eq in enumerate(equations):
        ax.text(1, 9.5 - i*0.5, eq, fontsize=12)
    
    # Set axis limits and remove ticks
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')
    
    # Add a legend
    legend_elements = [
        mpatches.Patch(color=input_color, alpha=0.7, label='Input'),
        mpatches.Patch(color=gate_color, alpha=0.7, label='Gates'),
        mpatches.Patch(color=norm_color, alpha=0.7, label='Gate Normalization'),
        mpatches.Patch(color=hidden_color, alpha=0.7, label='Hidden State')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    # Add a note about the key difference from standard LSTM
    ax.text(1, 0.5, "Key Differences from Standard LSTM: No hidden state dependency in gates, gate normalization ensures f^\prime_t + i^\prime_t = 1", 
            fontsize=10, style='italic', bbox=dict(facecolor='white', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('minlstm_architecture.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    create_minlstm_diagram()
