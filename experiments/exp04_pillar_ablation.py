"""
HEPAR Springer Benchmarks - Experiment 04: Pillar Ablation
===========================================================
Tests each HEPAR pillar independently to prove necessity.

HEPAR-minus-X variants:
- HEPAR-no-QT:   Disable quadtree (use all pixels)
- HEPAR-no-GC:   Disable Gray code sorting
- HEPAR-no-SPAE: Disable phase encoding
- HEPAR-no-TREX: Disable TREX mitigation

If removing a component does NOT hurt metrics, that component is dead weight.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import ImageProcessor, DatasetManager, LeafNode
from src.circuits import HEPAREncoder, Reconstructor
from src.metrics import MetricsCalculator

DPI = 300
np.random.seed(42)


def run_pillar_ablation(output_dir: str = 'results'):
    """
    Run HEPAR-minus-X ablation for each pillar.
    
    Returns DataFrame with SSIM for each variant.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("HEPAR PILLAR ABLATION (HEPAR-minus-X)")
    print("=" * 60)
    
    print("=" * 60)
    
    from src.config import load_config
    config = load_config()
    size = config.image_size
    
    dm = DatasetManager(size=size)
    proc = ImageProcessor(size)
    rec = Reconstructor()
    calc = MetricsCalculator()
    
    data = dm.get_image_with_metadata('Lena')
    img = data['image']
    
    results = []
    
    # ========== HEPAR-full ==========
    print("\n[1/5] HEPAR-full (baseline)...")
    leaves = data['leaves']
    hepar = HEPAREncoder()
    z, _, m = hepar.encode(leaves, apply_trex=True)
    recon = rec.reconstruct_hepar(z, leaves, img.shape)
    ssim_full = calc.ssim(img, recon)
    results.append({'Variant': 'HEPAR-full', 'SSIM': ssim_full, 'Leaves': len(leaves)})
    print(f"    SSIM: {ssim_full:.4f}, Leaves: {len(leaves)}")
    
    # ========== HEPAR-no-QT (use all pixels) ==========
    print("[2/5] HEPAR-no-QT (no quadtree)...")
    # Create one leaf per pixel (no compression)
    all_pixel_leaves = []
    grad = proc.compute_gradient(img)
    for r in range(size):
        for c in range(size):
            all_pixel_leaves.append(LeafNode(
                depth=int(np.log2(size)), row=r, col=c, size=1,
                value=float(img[r, c]),
                gradient=float(grad[r, c]),
                leaf_index=r * size + c
            ))
    z_noqt, _, _ = hepar.encode(all_pixel_leaves[:256], apply_trex=True)  # Limit for simulation
    recon_noqt = rec.reconstruct_hepar(z_noqt, all_pixel_leaves[:256], img.shape)
    ssim_noqt = calc.ssim(img, recon_noqt)
    results.append({'Variant': 'HEPAR-no-QT', 'SSIM': ssim_noqt, 'Leaves': len(all_pixel_leaves)})
    print(f"    SSIM: {ssim_noqt:.4f}, Leaves: {len(all_pixel_leaves)} (capped to 256)")
    
    # ========== HEPAR-no-GC (unsorted leaves) ==========
    print("[3/5] HEPAR-no-GC (no Gray code)...")
    unsorted_leaves, _ = proc.quadtree_decompose(img)
    # Don't sort - use original order
    z_nogc, _, _ = hepar.encode(unsorted_leaves, apply_trex=True)
    recon_nogc = rec.reconstruct_hepar(z_nogc, unsorted_leaves, img.shape)
    ssim_nogc = calc.ssim(img, recon_nogc)
    results.append({'Variant': 'HEPAR-no-GC', 'SSIM': ssim_nogc, 'Leaves': len(unsorted_leaves)})
    print(f"    SSIM: {ssim_nogc:.4f}")
    
    # ========== HEPAR-no-SPAE (no phase encoding) ==========
    print("[4/5] HEPAR-no-SPAE (no phase)...")
    # Zero out gradients
    nophase_leaves = data['leaves'].copy()
    for leaf in nophase_leaves:
        leaf.gradient = 0.0
    z_nospae, _, _ = hepar.encode(nophase_leaves, apply_trex=True)
    recon_nospae = rec.reconstruct_hepar(z_nospae, nophase_leaves, img.shape)
    ssim_nospae = calc.ssim(img, recon_nospae)
    results.append({'Variant': 'HEPAR-no-SPAE', 'SSIM': ssim_nospae, 'Leaves': len(nophase_leaves)})
    print(f"    SSIM: {ssim_nospae:.4f}")
    
    # ========== HEPAR-no-TREX ==========
    print("[5/5] HEPAR-no-TREX (no mitigation)...")
    z_notrex, _, _ = hepar.encode(leaves, apply_trex=False)
    recon_notrex = rec.reconstruct_hepar(z_notrex, leaves, img.shape)
    ssim_notrex = calc.ssim(img, recon_notrex)
    results.append({'Variant': 'HEPAR-no-TREX', 'SSIM': ssim_notrex, 'Leaves': len(leaves)})
    print(f"    SSIM: {ssim_notrex:.4f}")
    
    # Save results
    df = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, 'pillar_ablation.csv')
    df.to_csv(csv_path, index=False)
    print(f"\n[OK] Saved: {csv_path}")
    
    # ========== Figure: Pillar Impact Bar Chart ==========
    fig, ax = plt.subplots(figsize=(10, 6))
    
    variants = df['Variant'].tolist()
    ssims = df['SSIM'].tolist()
    
    colors = ['#2E7D32' if v == 'HEPAR-full' else '#D32F2F' for v in variants]
    bars = ax.bar(variants, ssims, color=colors, edgecolor='black')
    
    # Add baseline reference line
    ax.axhline(y=ssim_full, color='#2E7D32', linestyle='--', alpha=0.7, label='HEPAR-full baseline')
    
    # Annotate drops
    for bar, s, v in zip(bars, ssims, variants):
        if v != 'HEPAR-full':
            drop = ((ssim_full - s) / ssim_full) * 100
            if drop > 0:
                ax.annotate(f'-{drop:.1f}%', xy=(bar.get_x() + bar.get_width()/2, s),
                           ha='center', va='bottom', fontsize=10, color='#D32F2F', fontweight='bold')
    
    ax.set_ylabel('SSIM Score', fontsize=12, fontweight='bold')
    ax.set_title('HEPAR Pillar Ablation: Impact of Removing Each Component', fontsize=14, fontweight='bold')
    ax.legend()
    plt.xticks(rotation=15, ha='right')
    
    plt.tight_layout()
    path = os.path.join('figures', 'fig7_pillar_ablation.png')
    os.makedirs('figures', exist_ok=True)
    plt.savefig(path, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved: {path}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("PILLAR NECESSITY ANALYSIS")
    print("=" * 60)
    for _, row in df.iterrows():
        v = row['Variant']
        s = row['SSIM']
        if v == 'HEPAR-full':
            print(f"  {v}: SSIM = {s:.4f} (baseline)")
        else:
            drop = ((ssim_full - s) / ssim_full) * 100
            status = "NECESSARY" if drop > 1 else "MINIMAL IMPACT"
            print(f"  {v}: SSIM = {s:.4f} ({drop:+.1f}%) → {status}")
    
    return df


if __name__ == "__main__":
    run_pillar_ablation()
