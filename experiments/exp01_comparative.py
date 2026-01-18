"""
HEPAR Springer Benchmarks - Experiment 01: Comparative Analysis
================================================================
Generates the comprehensive 12-column comparison table for all 11 techniques.

Output: results/hepar_comprehensive.csv
"""

import os
import sys
import time
import pandas as pd
import numpy as np

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import ImageProcessor, DatasetManager
from src.circuits import HEPAREncoder, NEQREncoder, FRQIEncoder, Reconstructor, SHOTS
from src.noise_models import TREXCalibrator, create_ibm_noise_model, NoisySimulator
from src.metrics import MetricsCalculator, ComprehensiveMetrics
from src.estimators import EstimatorFactory


def run_comparative_analysis(
    datasets: list = None,
    output_dir: str = 'results',
    use_noise: bool = True,
    use_trex: bool = True
) -> pd.DataFrame:
    """
    Run full comparative analysis for 11 QIR techniques.
    
    Args:
        datasets: List of dataset names (default: ['Lena', 'MRI', 'Noise'])
        output_dir: Output directory for CSV
        use_noise: Whether to add IBM-like noise
        use_trex: Whether to apply TREX mitigation
        
    Returns:
        DataFrame with all results
    """
    if datasets is None:
        datasets = ['Lena', 'MRI', 'Noise']
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 70)
    print("HEPAR SPRINGER BENCHMARK - Comparative Analysis")
    print("=" * 70)
    print(f"Datasets: {datasets}")
    print(f"Noise: {use_noise}, TREX: {use_trex}")
    print()
    
    # Initialize components
    from src.config import load_config
    config = load_config()
    size = config.image_size
    
    dm = DatasetManager(size=size)
    proc = ImageProcessor(size)
    rec = Reconstructor()
    calc = MetricsCalculator()
    
    # Encoders
    hepar_enc = HEPAREncoder()
    neqr_enc = NEQREncoder()
    frqi_enc = FRQIEncoder()
    
    all_results = []
    recons = {ds: {} for ds in datasets}
    origs = {}
    ablation_data = {}
    
    # Process each dataset with simulated techniques
    for ds in datasets:
        print(f"\n{'='*50}")
        print(f"Processing: {ds}")
        print('='*50)
        
        data = dm.get_image_with_metadata(ds)
        img = data['image']
        origs[ds] = img
        grad = data['gradient']
        leaves = data['leaves']
        num_leaves = data['num_leaves']
        
        print(f"  Image size: {img.shape}")
        print(f"  Quadtree leaves: {num_leaves}")
        print(f"  Compression potential: {size*size/num_leaves:.1f}x")
        
        # =============== HEPAR ===============
        # Use adaptive threshold if too many leaves (preserves spatial coverage)
        MAX_LEAVES = 100
        from src.utils import verify_complete_coverage, build_leaf_coverage_mask
        
        if num_leaves > MAX_LEAVES:
            print(f"  [Info] Too many leaves ({num_leaves}). Using adaptive threshold...")
            leaves, _, thresh = proc.quadtree_decompose_with_target(img, target_leaves=MAX_LEAVES)
            sorted_leaves, _ = proc.gray_code_sort(leaves)
            leaves = sorted_leaves
            num_leaves = len(leaves)
            print(f"  [Info] Reduced to {num_leaves} leaves with threshold={thresh:.4f}")
        
        # Verify complete coverage before reconstruction
        if not verify_complete_coverage(leaves, img.shape):
            print("  [WARNING] Incomplete quadtree coverage - metrics may be invalid!")
        
        print(f"  [1/3] Running HEPAR (leaves={len(leaves)})...")
        t0 = time.time()
        z_counts, x_counts, h_metrics = hepar_enc.encode(leaves, apply_trex=use_trex)
        t_hepar = (time.time() - t0) * 1000
        
        img_hepar = rec.reconstruct_hepar(z_counts, leaves, img.shape)
        edge_hepar = rec.reconstruct_hepar_phase(x_counts, leaves, img.shape)
        recons[ds]['HEPAR'] = img_hepar
        
        # Ablation: Without TREX
        z_raw, _, _ = hepar_enc.encode(leaves, apply_trex=False)
        img_raw = rec.reconstruct_hepar(z_raw, leaves, img.shape)
        ssim_no_trex = calc.ssim(img, img_raw)
        
        if ds == 'Lena':
            ablation_data = {'raw': img_raw, 'trex': img_hepar, 'orig': img}
        
        h_result = calc.calculate_comprehensive(
            img, img_hepar,
            h_metrics.__dict__,
            'HEPAR', ds, num_leaves, ssim_no_trex, t_hepar
        )
        all_results.append(h_result.to_dict())
        print(f"    SSIM: {h_result.SSIM:.4f}, PSNR: {h_result.PSNR:.2f} dB")
        
        # =============== NEQR ===============
        print("  [2/3] Running NEQR...")
        # Check config to see if we should simulate baselines (slow)
        if hasattr(config, 'simulate_baselines') and not config.simulate_baselines:
             print("    [Info] Skipping NEQR simulation (disabled in config)")
             all_results.append({
                'Dataset': ds, 'Technique': 'NEQR',
                'SSIM': np.nan, 'PSNR': np.nan, 'Fidelity_Hellinger': np.nan,
                'Qubits_Logical': np.nan, 'Circuit_Depth': np.nan, 
                'Gate_Count_Clifford': np.nan, 'Encoding_Time_ms': 0.0,
                'Ablation_NoTREX': np.nan, 'Compression_Ratio': 1.0
             })
        else:
            t0 = time.time()
            n_counts, n_metrics = neqr_enc.encode(img)
            t_neqr = (time.time() - t0) * 1000
            
            img_neqr = rec.reconstruct_neqr(n_counts, img.shape)
            recons[ds]['NEQR'] = img_neqr
            
            n_result = calc.calculate_comprehensive(
                img, img_neqr,
                n_metrics.__dict__,
                'NEQR', ds, size*size, 0.0, t_neqr
            )
            all_results.append(n_result.to_dict())
            print(f"    SSIM: {n_result.SSIM:.4f}, PSNR: {n_result.PSNR:.2f} dB")
        
        # =============== FRQI ===============
        print("  [3/3] Running FRQI...")
        if hasattr(config, 'simulate_baselines') and not config.simulate_baselines:
             print("    [Info] Skipping FRQI simulation (disabled in config)")
             all_results.append({
                'Dataset': ds, 'Technique': 'FRQI',
                'SSIM': np.nan, 'PSNR': np.nan, 'Fidelity_Hellinger': np.nan,
                'Qubits_Logical': np.nan, 'Circuit_Depth': np.nan, 
                'Gate_Count_Clifford': np.nan, 'Encoding_Time_ms': 0.0,
                'Ablation_NoTREX': np.nan, 'Compression_Ratio': 1.0
             })
        else:
            t0 = time.time()
            f_counts, f_metrics = frqi_enc.encode(img)
            t_frqi = (time.time() - t0) * 1000
            
            img_frqi = rec.reconstruct_frqi(f_counts, img.shape)
            recons[ds]['FRQI'] = img_frqi
            
            f_result = calc.calculate_comprehensive(
                img, img_frqi,
                f_metrics.__dict__,
                'FRQI', ds, size*size, 0.0, t_frqi
            )
            all_results.append(f_result.to_dict())
            print(f"    SSIM: {f_result.SSIM:.4f}, PSNR: {f_result.PSNR:.2f} dB")
    
    # Add analytical-only techniques
    print("\n" + "="*50)
    print("Adding Analytical Estimators (8 techniques)")
    print("="*50)
    
    analytical_techs = ['MCQI', 'GQIR', '2D-QSNA', 'EFRQI', 'IQIR', 'NASS', 'QRMW', 'QUALPI']
    
    for tech in analytical_techs:
        estimator = EstimatorFactory.create(tech, size)
        metrics = estimator.get_metrics()
        
        all_results.append({
            'Dataset': 'Analytical',
            'Technique': tech,
            'Qubits_Logical': metrics.qubits_logical,
            'Qubits_Physical': metrics.qubits_physical,
            'Gate_Count_Clifford': metrics.gate_count_clifford,
            'Circuit_Depth': metrics.circuit_depth,
            'SWAP_Overhead': metrics.swap_overhead,
            'Encoding_Time_ms': 0.0,
            'Fidelity_Hellinger': np.nan,
            'PSNR': np.nan,
            'SSIM': np.nan,
            'Ablation_NoTREX': np.nan,
            'Compression_Ratio': 1.0
        })
        print(f"  {tech}: Q={metrics.qubits_logical}, D={metrics.circuit_depth}, Scalability={metrics.scalability}")
    
    # Create DataFrame
    df = pd.DataFrame(all_results)
    
    # Order columns
    cols = [
        'Dataset', 'Technique', 'Qubits_Logical', 'Qubits_Physical',
        'Gate_Count_Clifford', 'Circuit_Depth', 'SWAP_Overhead',
        'Encoding_Time_ms', 'Fidelity_Hellinger', 'PSNR', 'SSIM',
        'Ablation_NoTREX', 'Compression_Ratio'
    ]
    df = df[[c for c in cols if c in df.columns]]
    
    # Save CSV
    csv_path = os.path.join(output_dir, 'hepar_comprehensive.csv')
    try:
        df.to_csv(csv_path, index=False)
        print(f"\n[OK] Saved: {csv_path}")
    except PermissionError:
        # Fallback to timestamped filename if original is locked
        import datetime
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.join(output_dir, f'hepar_comprehensive_{ts}.csv')
        df.to_csv(csv_path, index=False)
        print(f"\n[OK] Saved (fallback): {csv_path}")
    
    # Print summary table
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    print(df.to_string(index=False))
    
    return df, origs, recons, ablation_data


if __name__ == "__main__":
    df, origs, recons, ablation = run_comparative_analysis()
