"""
HEPAR Springer Benchmarks - Experiment 03: Ablation Study
==========================================================
Compares HEPAR with and without error mitigation (TREX, DD).

Output: 
- figures/fig3_ablation_study.png
- figures/fig6_ablation_violin.png
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, fftshift

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import ImageProcessor, DatasetManager
from src.circuits import HEPAREncoder, Reconstructor
from src.metrics import MetricsCalculator

DPI = 300


def run_ablation_study(output_dir: str = 'figures', num_runs: int = 10):
    """
    Run ablation study comparing mitigation strategies.
    
    Compares:
    - No mitigation (raw noise)
    - TREX only
    - DD only (future)
    - TREX + DD (future)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("HEPAR Ablation Study: Error Mitigation Analysis")
    print("=" * 60)
    
    print(f"Running {num_runs} trials each...\n")
    
    from src.config import load_config
    config = load_config()
    size = config.image_size
    
    dm = DatasetManager(size=size)
    rec = Reconstructor()
    calc = MetricsCalculator()
    
    # Use Lena for ablation
    data = dm.get_image_with_metadata('Lena')
    img = data['image']
    leaves = data['leaves']
    
    print(f"\nDataset: Lena (32×32)")
    print(f"Quadtree leaves: {len(leaves)}")
    print(f"Running {num_runs} trials each...\n")
    
    # Collect SSIM scores
    ssim_raw = []
    ssim_trex = []
    
    hepar_enc = HEPAREncoder()
    
    for i in range(num_runs):
        # Raw (no TREX)
        z_raw, _, _ = hepar_enc.encode(leaves, apply_trex=False)
        img_raw = rec.reconstruct_hepar(z_raw, leaves, img.shape)
        ssim_raw.append(calc.ssim(img, img_raw))
        
        # With TREX
        z_trex, _, _ = hepar_enc.encode(leaves, apply_trex=True)
        img_trex = rec.reconstruct_hepar(z_trex, leaves, img.shape)
        ssim_trex.append(calc.ssim(img, img_trex))
        
        print(f"  Trial {i+1}/{num_runs}: Raw SSIM={ssim_raw[-1]:.4f}, TREX SSIM={ssim_trex[-1]:.4f}")
    
    # Get final images for visualization
    z_raw, _, _ = hepar_enc.encode(leaves, apply_trex=False)
    img_raw = rec.reconstruct_hepar(z_raw, leaves, img.shape)
    
    z_trex, x_trex, _ = hepar_enc.encode(leaves, apply_trex=True)
    img_trex = rec.reconstruct_hepar(z_trex, leaves, img.shape)
    
    # ========== Figure: Side-by-side Ablation ==========
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(img, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title('Original', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(img_raw, cmap='gray', vmin=0, vmax=1)
    ssim_r = np.mean(ssim_raw)
    axes[1].set_title(f'HEPAR (No TREX)\nSSIM: {ssim_r:.4f}', fontsize=14, fontweight='bold', color='#D32F2F')
    axes[1].axis('off')
    
    axes[2].imshow(img_trex, cmap='gray', vmin=0, vmax=1)
    ssim_t = np.mean(ssim_trex)
    axes[2].set_title(f'HEPAR + TREX\nSSIM: {ssim_t:.4f}', fontsize=14, fontweight='bold', color='#2E7D32')
    axes[2].axis('off')
    
    plt.suptitle('Figure: Ablation Study - TREX Error Mitigation', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    path = os.path.join(output_dir, 'fig3_ablation_study.png')
    plt.savefig(path, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"\n[OK] Saved: {path}")
    
    # ========== Figure 6: Violin Plot ==========
    fig, ax = plt.subplots(figsize=(8, 6))
    
    positions = [1, 2]
    data = [ssim_raw, ssim_trex]
    labels = ['No Mitigation', 'TREX']
    colors = ['#D32F2F', '#2E7D32']
    
    parts = ax.violinplot(data, positions=positions, showmeans=True, showmedians=True)
    
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.7)
    
    # Add individual points
    for i, d in enumerate(data):
        x = np.random.normal(positions[i], 0.04, size=len(d))
        ax.scatter(x, d, alpha=0.5, s=30, color='black')
    
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel('SSIM Score', fontsize=12, fontweight='bold')
    ax.set_title('Figure 6: SSIM Distribution by Mitigation Strategy\n(10 trials, Lena 32×32)', 
                 fontsize=14, fontweight='bold')
    
    # SSIM improvement annotation
    improvement = (np.mean(ssim_trex) - np.mean(ssim_raw)) / np.mean(ssim_raw) * 100
    ax.annotate(f'+{improvement:.1f}%', xy=(1.5, np.mean(ssim_trex)), fontsize=14, 
                ha='center', fontweight='bold', color='#2E7D32')
    
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    path = os.path.join(output_dir, 'fig6_ablation_violin.png')
    plt.savefig(path, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved: {path}")
    
    # ========== Figure 4: Spectral Analysis ==========
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    f_orig = np.log1p(np.abs(fftshift(fft2(img))))
    f_hepar = np.log1p(np.abs(fftshift(fft2(img_trex))))
    
    im0 = axes[0].imshow(f_orig, cmap='inferno')
    axes[0].set_title('FFT: Original', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    plt.colorbar(im0, ax=axes[0], fraction=0.046)
    
    im1 = axes[1].imshow(f_hepar, cmap='inferno')
    axes[1].set_title('FFT: HEPAR Reconstruction', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046)
    
    plt.suptitle('Figure 4: Spectral Analysis (2D FFT Log-Magnitude)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    path = os.path.join(output_dir, 'fig4_spectral_analysis.png')
    plt.savefig(path, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved: {path}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("ABLATION STUDY SUMMARY")
    print("=" * 60)
    print(f"  Raw SSIM:  {np.mean(ssim_raw):.4f} ± {np.std(ssim_raw):.4f}")
    print(f"  TREX SSIM: {np.mean(ssim_trex):.4f} ± {np.std(ssim_trex):.4f}")
    print(f"  Improvement: +{improvement:.1f}%")
    
    return {
        'ssim_raw': ssim_raw,
        'ssim_trex': ssim_trex,
        'improvement_pct': improvement
    }


if __name__ == "__main__":
    results = run_ablation_study()
