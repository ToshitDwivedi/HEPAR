"""
HEPAR Springer Benchmarks - Experiment 02: Scaling Analysis
============================================================
Generates log-log complexity plots for all 11 techniques.

Output: figures/fig2_complexity_scaling.png, figures/fig5_crossover.png
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.estimators import EstimatorFactory

DPI = 300
COLORS = {
    'HEPAR': '#2E7D32', 'NEQR': '#1976D2', 'FRQI': '#D32F2F',
    'MCQI': '#7B1FA2', 'GQIR': '#F57C00', '2D-QSNA': '#00796B',
    'EFRQI': '#C2185B', 'IQIR': '#512DA8', 'NASS': '#0097A7',
    'QRMW': '#455A64', 'QUALPI': '#F9A825'
}


def generate_scaling_plots(output_dir: str = 'figures'):
    """Generate all scaling analysis plots."""
    os.makedirs(output_dir, exist_ok=True)
    
    sizes = [8, 16, 32, 64, 128, 256]
    techniques = list(COLORS.keys())
    
    # Collect data
    qubit_data = {tech: [] for tech in techniques}
    depth_data = {tech: [] for tech in techniques}
    gate_data = {tech: [] for tech in techniques}
    
    for size in sizes:
        for tech in techniques:
            try:
                est = EstimatorFactory.create(tech, size)
                metrics = est.get_metrics()
                qubit_data[tech].append(metrics.qubits_logical)
                depth_data[tech].append(metrics.circuit_depth)
                gate_data[tech].append(metrics.gate_count_clifford)
            except:
                qubit_data[tech].append(np.nan)
                depth_data[tech].append(np.nan)
                gate_data[tech].append(np.nan)
    
    # ========== Figure 2: Complexity Scaling (3 subplots) ==========
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Qubits
    ax = axes[0]
    for tech in techniques:
        ax.plot(sizes, qubit_data[tech], 'o-', label=tech, color=COLORS[tech], linewidth=2, markersize=6)
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_xlabel('Image Size (N×N)', fontweight='bold')
    ax.set_ylabel('Logical Qubits', fontweight='bold')
    ax.set_title('(a) Qubit Scaling', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=8, ncol=2)
    
    # Circuit Depth
    ax = axes[1]
    for tech in techniques:
        ax.plot(sizes, depth_data[tech], 'o-', label=tech, color=COLORS[tech], linewidth=2, markersize=6)
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_xlabel('Image Size (N×N)', fontweight='bold')
    ax.set_ylabel('Circuit Depth', fontweight='bold')
    ax.set_title('(b) Depth Scaling', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add IBM Eagle limit line
    ax.axhline(y=100, color='red', linestyle='--', alpha=0.7, label='NISQ Depth Limit')
    
    # Gate Count
    ax = axes[2]
    for tech in techniques:
        ax.plot(sizes, gate_data[tech], 'o-', label=tech, color=COLORS[tech], linewidth=2, markersize=6)
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_xlabel('Image Size (N×N)', fontweight='bold')
    ax.set_ylabel('Gate Count (Clifford)', fontweight='bold')
    ax.set_title('(c) Gate Scaling', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('Figure 2: Complexity Scaling Analysis (8×8 to 256×256)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    path = os.path.join(output_dir, 'fig2_complexity_scaling.png')
    plt.savefig(path, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved: {path}")
    
    # ========== Figure 5: Crossover Plot ==========
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Focus on key techniques for crossover
    key_techs = ['HEPAR', 'NEQR', 'FRQI', '2D-QSNA', 'MCQI']
    
    for tech in key_techs:
        ax.plot(sizes, qubit_data[tech], 'o-', label=tech, color=COLORS[tech], linewidth=2.5, markersize=8)
    
    # Annotate crossover point
    ax.axvline(x=64, color='gray', linestyle=':', alpha=0.7)
    ax.annotate('HEPAR Crossover\n(64×64)', xy=(64, 15), fontsize=10, ha='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # IBM Eagle limit
    ax.axhline(y=127, color='red', linestyle='--', alpha=0.7, linewidth=2)
    ax.text(sizes[-1], 130, 'IBM Eagle Limit (127 qubits)', ha='right', va='bottom', color='red')
    
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_xlabel('Image Size (N×N)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Logical Qubits Required', fontsize=12, fontweight='bold')
    ax.set_title('Figure 5: Scalability Crossover Analysis', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    path = os.path.join(output_dir, 'fig5_crossover_scaling.png')
    plt.savefig(path, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved: {path}")
    
    # ========== Figure 3: Qubit Wall ==========
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Data for 128x128 image (where differences are pronounced)
    size_128 = 128
    qubit_128 = []
    techs_sorted = []
    
    for tech in techniques:
        try:
            est = EstimatorFactory.create(tech, size_128)
            q = est.get_metrics().qubits_logical
            qubit_128.append(q)
            techs_sorted.append(tech)
        except:
            continue
    
    # Sort by qubit count
    sorted_idx = np.argsort(qubit_128)[::-1]
    qubit_128 = [qubit_128[i] for i in sorted_idx]
    techs_sorted = [techs_sorted[i] for i in sorted_idx]
    colors_sorted = [COLORS[t] for t in techs_sorted]
    
    bars = ax.bar(techs_sorted, qubit_128, color=colors_sorted, edgecolor='black')
    
    # IBM Eagle limit
    ax.axhline(y=127, color='red', linestyle='--', linewidth=2, label='IBM Eagle (127 qubits)')
    
    # Annotate bars exceeding limit
    for bar, q, tech in zip(bars, qubit_128, techs_sorted):
        if q > 127:
            bar.set_alpha(0.5)
            ax.annotate('EXCEEDS\nHARDWARE', xy=(bar.get_x() + bar.get_width()/2, q),
                       ha='center', va='bottom', fontsize=8, color='red', fontweight='bold')
    
    ax.set_ylabel('Logical Qubits Required', fontsize=12, fontweight='bold')
    ax.set_title(f'Figure 3: Qubit Requirements for {size_128}×{size_128} Images', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    path = os.path.join(output_dir, 'fig3_qubit_wall.png')
    plt.savefig(path, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved: {path}")


if __name__ == "__main__":
    generate_scaling_plots()
