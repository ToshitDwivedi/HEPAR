"""
HEPAR Springer Benchmarks - Leaf Statistics Logger
===================================================
Logs quadtree decomposition statistics for every image.

Outputs:
- Number of leaves
- Depth histogram
- Mean leaf area
- Links classical structure → quantum cost
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import ImageProcessor, DatasetManager
from src.metrics import EntropyCalculator


def log_leaf_statistics(output_dir: str = 'results'):
    """
    Generate comprehensive leaf statistics for all datasets.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("QUADTREE LEAF STATISTICS")
    print("=" * 60)
    
    print("=" * 60)
    
    from src.config import load_config
    config = load_config()
    size = config.image_size
    
    dm = DatasetManager(size=size)
    proc = ImageProcessor(size)
    ent = EntropyCalculator()
    
    all_stats = []
    depth_histograms = {}
    
    for name in ['Lena', 'MRI', 'Noise']:
        print(f"\nProcessing {name}...")
        
        img = proc.load_image(name)
        leaves, density = proc.quadtree_decompose(img)
        sorted_leaves, hamming_savings = proc.gray_code_sort(leaves)
        
        # Calculate statistics
        num_leaves = len(sorted_leaves)
        depths = [leaf.depth for leaf in sorted_leaves]
        areas = [leaf.size ** 2 for leaf in sorted_leaves]
        values = [leaf.value for leaf in sorted_leaves]
        gradients = [leaf.gradient for leaf in sorted_leaves]
        
        entropy = ent.shannon_entropy(img)
        complexity = ent.image_complexity(img)
        compression_ratio = (size * size) / max(num_leaves, 1)
        
        stats = {
            'Dataset': name,
            'Num_Leaves': num_leaves,
            'Min_Depth': min(depths),
            'Max_Depth': max(depths),
            'Mean_Depth': np.mean(depths),
            'Min_Area': min(areas),
            'Max_Area': max(areas),
            'Mean_Area': np.mean(areas),
            'Mean_Intensity': np.mean(values),
            'Std_Intensity': np.std(values),
            'Mean_Gradient': np.mean(gradients),
            'Shannon_Entropy': entropy,
            'Complexity': complexity,
            'Compression_Ratio': compression_ratio,
            'Hamming_Savings': hamming_savings
        }
        all_stats.append(stats)
        
        # Depth histogram
        depth_hist = {}
        for d in range(6):
            depth_hist[d] = depths.count(d)
        depth_histograms[name] = depth_hist
        
        print(f"  Leaves: {num_leaves}")
        print(f"  Depth range: {min(depths)} - {max(depths)}")
        print(f"  Mean area: {np.mean(areas):.1f} px²")
        print(f"  Entropy: {entropy:.2f} bits")
        print(f"  Compression: {compression_ratio:.1f}x")
    
    # Save CSV
    df = pd.DataFrame(all_stats)
    csv_path = os.path.join(output_dir, 'leaf_statistics.csv')
    df.to_csv(csv_path, index=False)
    print(f"\n[OK] Saved: {csv_path}")
    
    # ========== Figure: Leaf Statistics Comparison ==========
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    datasets = df['Dataset'].tolist()
    
    # Subplot 1: Leaves and Compression
    ax = axes[0]
    x = np.arange(len(datasets))
    width = 0.35
    
    ax.bar(x - width/2, df['Num_Leaves'], width, label='Leaves', color='#1976D2')
    ax2 = ax.twinx()
    ax2.bar(x + width/2, df['Compression_Ratio'], width, label='CR', color='#2E7D32', alpha=0.7)
    
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylabel('Number of Leaves', color='#1976D2')
    ax2.set_ylabel('Compression Ratio', color='#2E7D32')
    ax.set_title('(a) Quadtree Size & Compression')
    
    # Subplot 2: Depth Distribution
    ax = axes[1]
    bottoms = np.zeros(len(datasets))
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, 6))
    
    for d in range(6):
        heights = [depth_histograms[ds].get(d, 0) for ds in datasets]
        ax.bar(datasets, heights, bottom=bottoms, label=f'Depth {d}', color=colors[d])
        bottoms += np.array(heights)
    
    ax.set_ylabel('Count')
    ax.set_title('(b) Depth Distribution')
    ax.legend(loc='upper right', fontsize=8)
    
    # Subplot 3: Entropy vs Compression
    ax = axes[2]
    ax.scatter(df['Shannon_Entropy'], df['Compression_Ratio'], s=200, 
               c=['#2E7D32' if cr > 1.5 else '#D32F2F' for cr in df['Compression_Ratio']],
               edgecolor='black')
    
    for i, name in enumerate(datasets):
        ax.annotate(name, (df['Shannon_Entropy'].iloc[i], df['Compression_Ratio'].iloc[i]),
                   textcoords="offset points", xytext=(10, 5), fontsize=11)
    
    ax.set_xlabel('Shannon Entropy (bits)', fontweight='bold')
    ax.set_ylabel('Compression Ratio', fontweight='bold')
    ax.set_title('(c) Entropy vs Compression')
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('Figure: Quadtree Leaf Statistics (Classical → Quantum Cost)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    path = os.path.join(output_dir.replace('results', 'figures'), 'fig9_leaf_statistics.png')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved: {path}")
    
    return df


if __name__ == "__main__":
    log_leaf_statistics()
