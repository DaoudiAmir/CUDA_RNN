import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle
import matplotlib.path as mpath
import matplotlib.patches as mpatches

def create_mingru_diagram():
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Define colors
    input_color = '#3498db'  # Blue
    gate_color = '#e74c3c'   # Red
    hidden_color = '#2ecc71' # Green
    arrow_color = '#7f8c8d'  # Gray
    
    # Define positions
    x_t_pos = (1, 5)
    h_prev_pos = (1, 2)
    z_t_pos = (4, 5)
    h_tilde_pos = (4, 2)
    h_t_pos = (7, 3.5)
    
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
    z_t_circle = plt.Circle(z_t_pos, node_radius, color=gate_color, alpha=0.7)
    ax.add_patch(z_t_circle)
    ax.text(z_t_pos[0], z_t_pos[1], r'$z_t$', ha='center', va='center', fontsize=14)
    
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
    
    # x_t to z_t
    ax.add_patch(FancyArrowPatch(x_t_pos, z_t_pos, connectionstyle='arc3,rad=0', **arrow_props))
    ax.text((x_t_pos[0] + z_t_pos[0])/2, (x_t_pos[1] + z_t_pos[1])/2 + 0.3, 'Linear + Sigmoid', fontsize=10)
    
    # x_t to h_tilde
    ax.add_patch(FancyArrowPatch(x_t_pos, h_tilde_pos, connectionstyle='arc3,rad=0', **arrow_props))
    ax.text((x_t_pos[0] + h_tilde_pos[0])/2, (x_t_pos[1] + h_tilde_pos[1])/2 - 0.3, 'Linear', fontsize=10)
    
    # h_prev to h_t
    ax.add_patch(FancyArrowPatch(h_prev_pos, h_t_pos, connectionstyle='arc3,rad=-0.2', **arrow_props))
    ax.text((h_prev_pos[0] + h_t_pos[0])/2 - 0.5, (h_prev_pos[1] + h_t_pos[1])/2 - 0.3, '(1-z_t) ⊙', fontsize=10)
    
    # h_tilde to h_t
    ax.add_patch(FancyArrowPatch(h_tilde_pos, h_t_pos, connectionstyle='arc3,rad=0.2', **arrow_props))
    ax.text((h_tilde_pos[0] + h_t_pos[0])/2 + 0.5, (h_tilde_pos[1] + h_t_pos[1])/2 - 0.3, 'z_t ⊙', fontsize=10)
    
    # z_t to h_t (indicating its use in the weighted sum)
    ax.add_patch(FancyArrowPatch(z_t_pos, h_t_pos, connectionstyle='arc3,rad=0', **arrow_props))
    
    # Add title and equations
    ax.set_title('MinGRU Architecture', fontsize=16)
    
    # Add equations
    equations = [
        r'$z_t = \sigma(W_z x_t + b_z)$',
        r'$\tilde{h}_t = W_h x_t + b_h$',
        r'$h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$'
    ]
    
    for i, eq in enumerate(equations):
        ax.text(1, 7.5 - i*0.5, eq, fontsize=12)
    
    # Set axis limits and remove ticks
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 8)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')
    
    # Add a legend
    legend_elements = [
        mpatches.Patch(color=input_color, alpha=0.7, label='Input'),
        mpatches.Patch(color=gate_color, alpha=0.7, label='Gate'),
        mpatches.Patch(color=hidden_color, alpha=0.7, label='Hidden State')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    # Add a note about the key difference from standard GRU
    ax.text(1, 0.5, "Key Difference from Standard GRU: No hidden state dependency in gates", 
            fontsize=10, style='italic', bbox=dict(facecolor='white', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('mingru_architecture.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    create_mingru_diagram()
