"""
HEPAR Springer Benchmarks - Experiment 05: Entropy-Aware Scaling
================================================================
Plots circuit metrics vs IMAGE ENTROPY (not just resolution).

This proves HEPAR is structure-aware and explains honest failure modes.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import ImageProcessor
from src.metrics import EntropyCalculator
from src.estimators import HEPAREstimator

DPI = 300
np.random.seed(42)


def generate_test_images_by_entropy(size: int = 32, num_samples: int = 10):
    """
    Generate images spanning low to high entropy.
    
    Returns list of (image, entropy, name) tuples.
    """
    proc = ImageProcessor(size)
    ent = EntropyCalculator()
    
    images = []
    
    # Low entropy: simple shapes
    for i in range(num_samples // 3):
        img = np.zeros((size, size))
        cx, cy = np.random.randint(size//4, 3*size//4, 2)
        r = np.random.randint(size//6, size//3)
        y, x = np.ogrid[:size, :size]
        mask = (x - cx)**2 + (y - cy)**2 <= r**2
        img[mask] = np.random.uniform(0.6, 1.0)
        h = ent.shannon_entropy(img)
        images.append((img, h, f'Simple_{i}'))
    
    # Medium entropy: gradients
    for i in range(num_samples // 3):
        x, y = np.meshgrid(np.linspace(0, 1, size), np.linspace(0, 1, size))
        freq = np.random.uniform(2, 5)
        img = 0.5 + 0.4 * np.sin(freq * x * np.pi) * np.cos(freq * y * np.pi)
        img = np.clip(img, 0, 1)
        h = ent.shannon_entropy(img)
        images.append((img, h, f'Gradient_{i}'))
    
    # High entropy: noise
    for i in range(num_samples - 2 * (num_samples // 3)):
        noise_level = np.random.uniform(0.3, 1.0)
        img = np.clip(np.random.rand(size, size) * noise_level + (1 - noise_level) * 0.5, 0, 1)
        h = ent.shannon_entropy(img)
        images.append((img, h, f'Noise_{i}'))
    
    return sorted(images, key=lambda x: x[1])  # Sort by entropy


def run_entropy_scaling(output_dir: str = 'figures', num_samples: int = 15):
    """
    Generate entropy-aware scaling plots.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    print("=" * 60)
    print("ENTROPY-AWARE SCALING ANALYSIS")
    print("=" * 60)
    
    print("=" * 60)
    
    from src.config import load_config
    config = load_config()
    size = config.image_size
    
    proc = ImageProcessor(size)
    
    results = []
    
    print(f"\nGenerating {num_samples} test images...")
    images = generate_test_images_by_entropy(size, num_samples)
    
    for img, entropy, name in images:
        # Decompose with quadtree
        leaves, _ = proc.quadtree_decompose(img)
        num_leaves = len(leaves)
        
        # Estimate HEPAR metrics
        est = HEPAREstimator(size, num_leaves=num_leaves)
        metrics = est.get_metrics()
        
        # Compression ratio
        cr = (size * size) / max(num_leaves, 1)
        
        # Flag low compression regime
        regime = "COMPRESSION" if cr > 1.2 else "NO_COMPRESSION"
        
        results.append({
            'Image': name,
            'Entropy': entropy,
            'Leaves': num_leaves,
            'Compression_Ratio': cr,
            'Circuit_Depth': metrics.circuit_depth,
            'Gates': metrics.gate_count_clifford,
            'Qubits': metrics.qubits_logical,
            'Regime': regime
        })
        
        print(f"  {name}: H={entropy:.2f}, L={num_leaves}, CR={cr:.1f}x ({regime})")
    
    df = pd.DataFrame(results)
    df.to_csv('results/entropy_scaling.csv', index=False)
    print(f"\n[OK] Saved: results/entropy_scaling.csv")
    
    # ========== Figure: Depth vs Entropy ==========
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Color by regime
    colors = ['#2E7D32' if r == 'COMPRESSION' else '#D32F2F' for r in df['Regime']]
    
    # Subplot 1: Leaves vs Entropy
    ax = axes[0]
    ax.scatter(df['Entropy'], df['Leaves'], c=colors, s=80, edgecolor='black')
    ax.set_xlabel('Shannon Entropy (bits)', fontweight='bold')
    ax.set_ylabel('Quadtree Leaves', fontweight='bold')
    ax.set_title('(a) Leaves vs Entropy', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Subplot 2: Depth vs Entropy
    ax = axes[1]
    ax.scatter(df['Entropy'], df['Circuit_Depth'], c=colors, s=80, edgecolor='black')
    ax.set_xlabel('Shannon Entropy (bits)', fontweight='bold')
    ax.set_ylabel('Circuit Depth', fontweight='bold')
    ax.set_title('(b) Depth vs Entropy', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Subplot 3: Compression Ratio vs Entropy
    ax = axes[2]
    ax.scatter(df['Entropy'], df['Compression_Ratio'], c=colors, s=80, edgecolor='black')
    ax.axhline(y=1.2, color='gray', linestyle='--', alpha=0.7, label='Min useful CR (1.2)')
    ax.set_xlabel('Shannon Entropy (bits)', fontweight='bold')
    ax.set_ylabel('Compression Ratio', fontweight='bold')
    ax.set_title('(c) Compression vs Entropy', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add legend for colors
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2E7D32', edgecolor='black', label='Compression regime'),
        Patch(facecolor='#D32F2F', edgecolor='black', label='No compression')
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.02))
    
    plt.suptitle('Figure: HEPAR Performance vs Image Entropy (Structure-Awareness)', fontsize=14, fontweight='bold', y=1.08)
    plt.tight_layout()
    
    path = os.path.join(output_dir, 'fig8_entropy_scaling.png')
    plt.savefig(path, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved: {path}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("ENTROPY SCALING SUMMARY")
    print("=" * 60)
    comp_df = df[df['Regime'] == 'COMPRESSION']
    nocomp_df = df[df['Regime'] == 'NO_COMPRESSION']
    print(f"  Images in COMPRESSION regime: {len(comp_df)} (avg CR: {comp_df['Compression_Ratio'].mean():.1f}x)")
    print(f"  Images in NO_COMPRESSION regime: {len(nocomp_df)} (avg CR: {nocomp_df['Compression_Ratio'].mean():.1f}x)")
    print(f"\n  Entropy threshold for compression: ~{df[df['Compression_Ratio'] > 1.2]['Entropy'].max():.2f} bits")
    
    return df


if __name__ == "__main__":
    run_entropy_scaling()
