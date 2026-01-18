"""
HEPAR Springer Benchmarks - Master Orchestration Script
========================================================
Runs the complete benchmark pipeline and generates all outputs.

Usage:
    python run_all.py [--quick]
    
Options:
    --quick : Skip ablation multi-run, use single trial
"""

import os
import sys
import time
import argparse

# Ensure consistent paths
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '.')

# Reproducibility seeds (important for Springer reviewers)
import numpy as np
np.random.seed(42)
try:
    from qiskit_algorithms.utils import algorithm_globals
    algorithm_globals.random_seed = 42
except ImportError:
    pass


def run_all(quick_mode: bool = False):
    """
    Run complete benchmark pipeline.
    
    Outputs:
    - results/hepar_comprehensive.csv
    - results/latex_tables.tex
    - figures/fig1_visual_truth.png
    - figures/fig2_complexity_scaling.png
    - figures/fig3_ablation_study.png (or qubit_wall)
    - figures/fig4_spectral_analysis.png
    - figures/fig5_crossover_scaling.png
    - figures/fig6_ablation_violin.png
    """
    start_time = time.time()
    
    print("=" * 70)
    print("HEPAR SPRINGER BENCHMARK SUITE")
    print("Research-Grade Quantum Image Representation Analysis")
    print("=" * 70)
    print(f"Mode: {'Quick' if quick_mode else 'Full'}")
    print()
    
    # Create output directories
    os.makedirs('results', exist_ok=True)
    os.makedirs('figures', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    
    # ========== Step 1: Comparative Analysis ==========
    print("\n" + "=" * 50)
    print("STEP 1: Comparative Analysis (11 Techniques)")
    print("=" * 50)
    
    from experiments.exp01_comparative import run_comparative_analysis
    df, origs, recons, ablation_data = run_comparative_analysis(output_dir='results')
    
    # ========== Step 2: Generate Visual Truth Grid (Fig 1) ==========
    print("\n" + "=" * 50)
    print("STEP 2: Generate Visual Truth Grid")
    print("=" * 50)
    
    generate_visual_truth_grid(origs, recons, 'figures')
    
    # ========== Step 3: Scaling Plots ==========
    print("\n" + "=" * 50)
    print("STEP 3: Scaling Analysis Plots")
    print("=" * 50)
    
    from experiments.exp02_scaling import generate_scaling_plots
    generate_scaling_plots(output_dir='figures')
    
    # ========== Step 4: Ablation Study ==========
    print("\n" + "=" * 50)
    print("STEP 4: Ablation Study")
    print("=" * 50)
    
    from experiments.exp03_ablation import run_ablation_study
    ablation_results = run_ablation_study(
        output_dir='figures',
        num_runs=3 if quick_mode else 10
    )
    
    # ========== Step 5: Generate LaTeX ==========
    print("\n" + "=" * 50)
    print("STEP 5: Generate LaTeX Tables")
    print("=" * 50)
    
    from generate_latex import generate_all_latex
    generate_all_latex('results/hepar_comprehensive.csv', 'results/latex_tables.tex')

    # ========== Step 6: Leaf Statistics ==========
    print("\n" + "=" * 50)
    print("STEP 6: Leaf Statistics")
    print("=" * 50)

    from experiments.exp06_leaf_stats import log_leaf_statistics
    log_leaf_statistics('results')
    
    # ========== Summary ==========
    elapsed = time.time() - start_time
    
    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE!")
    print("=" * 70)
    print(f"\nTotal time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    print("\nGenerated outputs:")
    print("  CSV:")
    print("    - results/hepar_comprehensive.csv (12 metrics × 11 techniques)")
    print("    - results/leaf_statistics.csv (Quadtree metrics)")
    print("  LaTeX:")
    print("    - results/latex_tables.tex (Publication-ready tables)")
    print("  Figures (300 DPI):")
    for f in sorted(os.listdir('figures')):
        if f.endswith('.png'):
            size_kb = os.path.getsize(f'figures/{f}') / 1024
            print(f"    - figures/{f} ({size_kb:.1f} KB)")
    
    print("\n[OK] All outputs ready for Springer submission!")
    

def generate_visual_truth_grid(origs: dict, recons: dict, output_dir: str):
    """Generate Figure 1: Visual Truth Grid."""
    import matplotlib.pyplot as plt
    import numpy as np
    
    datasets = list(origs.keys())
    fig, axes = plt.subplots(len(datasets), 3, figsize=(12, 12))
    
    COLORS = {'NEQR': '#1976D2', 'HEPAR': '#2E7D32'}
    
    for i, ds in enumerate(datasets):
        # Original
        axes[i, 0].imshow(origs[ds], cmap='gray', vmin=0, vmax=1, interpolation='nearest')
        axes[i, 0].set_title(f'{ds}\nOriginal', fontweight='bold', fontsize=11)
        axes[i, 0].axis('off')
        
        # NEQR
        img_neqr = recons[ds].get('NEQR', origs[ds])
        axes[i, 1].imshow(img_neqr, cmap='gray', vmin=0, vmax=1, interpolation='nearest')
        axes[i, 1].set_title('NEQR', fontweight='bold', fontsize=11, color=COLORS['NEQR'])
        axes[i, 1].axis('off')
        
        # HEPAR
        img_hepar = recons[ds].get('HEPAR', origs[ds])
        axes[i, 2].imshow(img_hepar, cmap='gray', vmin=0, vmax=1, interpolation='nearest')
        axes[i, 2].set_title('HEPAR', fontweight='bold', fontsize=11, color=COLORS['HEPAR'])
        axes[i, 2].axis('off')
    
    plt.suptitle('Figure 1: Visual Truth Grid (32×32, interpolation=nearest)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    path = f'{output_dir}/fig1_visual_truth.png'
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved: {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='HEPAR Benchmark Suite')
    parser.add_argument('--quick', action='store_true', help='Quick mode (fewer trials)')
    args = parser.parse_args()
    
    run_all(quick_mode=args.quick)
