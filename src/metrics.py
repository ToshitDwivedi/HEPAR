"""
HEPAR Springer Benchmarks - Metrics Calculations
=================================================
PSNR, SSIM, Fidelity, and other quality metrics for QIR comparison.

Implements all 12 metrics for the comprehensive CSV:
1. Technique, 2. Qubits_Logical, 3. Qubits_Physical
4. Gate_Count_Clifford, 5. Circuit_Depth, 6. SWAP_Overhead
7. Encoding_Time_ms, 8. Fidelity_Hellinger, 9. PSNR
10. SSIM, 11. Ablation_NoTREX, 12. Compression_Ratio
"""

import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass, asdict
from scipy.fft import fft2, fftshift

try:
    from skimage.metrics import peak_signal_noise_ratio as calc_psnr
    from skimage.metrics import structural_similarity as calc_ssim
except ImportError:
    # Fallback implementations
    def calc_psnr(original, reconstructed, data_range=1.0):
        mse = np.mean((original - reconstructed) ** 2)
        if mse == 0:
            return float('inf')
        return 10 * np.log10((data_range ** 2) / mse)
    
    def calc_ssim(original, reconstructed, data_range=1.0, **kwargs):
        # Simplified SSIM
        c1 = (0.01 * data_range) ** 2
        c2 = (0.03 * data_range) ** 2
        mu_x = np.mean(original)
        mu_y = np.mean(reconstructed)
        sigma_x = np.var(original)
        sigma_y = np.var(reconstructed)
        sigma_xy = np.cov(original.flatten(), reconstructed.flatten())[0, 1]
        ssim = ((2*mu_x*mu_y + c1) * (2*sigma_xy + c2)) / ((mu_x**2 + mu_y**2 + c1) * (sigma_x + sigma_y + c2))
        return ssim


@dataclass
class ComprehensiveMetrics:
    """Container for all 12 CSV metrics."""
    Dataset: str
    Technique: str
    Qubits_Logical: int
    Qubits_Physical: int
    Gate_Count_Clifford: int
    Circuit_Depth: int
    SWAP_Overhead: int
    Encoding_Time_ms: float
    Fidelity_Hellinger: float
    PSNR: float
    SSIM: float
    Ablation_NoTREX: float
    Compression_Ratio: float
    
    def to_dict(self) -> Dict:
        return asdict(self)


class MetricsCalculator:
    """Calculate all metrics for QIR comparison."""
    
    @staticmethod
    def hellinger_fidelity(p: np.ndarray, q: np.ndarray) -> float:
        """
        Calculate Hellinger fidelity between two images (as probability distributions).
        
        F_H = 1 - H(p, q) / √2
        where H is Hellinger distance.
        
        Args:
            p: Original image
            q: Reconstructed image
            
        Returns:
            Hellinger fidelity in [0, 1]
        """
        p = p.flatten()
        q = q.flatten()
        
        # Normalize to probability distributions
        p_safe = np.abs(p) / max(np.sum(np.abs(p)), 1e-10)
        q_safe = np.abs(q) / max(np.sum(np.abs(q)), 1e-10)
        
        # Hellinger distance
        h_dist = np.sqrt(np.sum((np.sqrt(p_safe) - np.sqrt(q_safe))**2)) / np.sqrt(2)
        
        # Fidelity = 1 - distance
        return max(0, min(1, 1 - h_dist))
    
    @staticmethod
    def psnr(original: np.ndarray, reconstructed: np.ndarray) -> float:
        """Calculate Peak Signal-to-Noise Ratio."""
        return calc_psnr(original, reconstructed, data_range=1.0)
    
    @staticmethod
    def ssim(original: np.ndarray, reconstructed: np.ndarray) -> float:
        """Calculate Structural Similarity Index."""
        # Adjust window size for small images
        win_size = min(7, min(original.shape[0], original.shape[1]))
        if win_size < 3:
            win_size = 3
        if win_size % 2 == 0:
            win_size -= 1
        
        return calc_ssim(original, reconstructed, data_range=1.0, win_size=win_size)
    
    @staticmethod
    def compression_ratio(N: int, num_leaves: int) -> float:
        """
        Calculate compression ratio for quadtree-based techniques.
        
        CR = N² / K where K is number of leaves.
        """
        return (N * N) / max(num_leaves, 1)
    
    @staticmethod
    def spectral_fidelity(original: np.ndarray, reconstructed: np.ndarray) -> float:
        """
        Calculate spectral (FFT) fidelity for high-frequency preservation.
        
        Measures how well high-frequency components are preserved.
        """
        # 2D FFT
        fft_orig = np.abs(fftshift(fft2(original)))
        fft_recon = np.abs(fftshift(fft2(reconstructed)))
        
        # Normalize
        fft_orig = fft_orig / max(fft_orig.max(), 1e-10)
        fft_recon = fft_recon / max(fft_recon.max(), 1e-10)
        
        # MSE in frequency domain
        mse = np.mean((fft_orig - fft_recon) ** 2)
        
        # Convert to fidelity-like metric
        return max(0, 1 - np.sqrt(mse))
    
    @staticmethod
    def edge_preservation(original: np.ndarray, reconstructed: np.ndarray) -> float:
        """
        Calculate edge preservation score using Sobel gradients.
        """
        from scipy.ndimage import sobel
        
        # Compute gradients
        gx_orig = sobel(original, axis=1)
        gy_orig = sobel(original, axis=0)
        grad_orig = np.sqrt(gx_orig**2 + gy_orig**2)
        
        gx_recon = sobel(reconstructed, axis=1)
        gy_recon = sobel(reconstructed, axis=0)
        grad_recon = np.sqrt(gx_recon**2 + gy_recon**2)
        
        # Normalize
        if grad_orig.max() > 0:
            grad_orig = grad_orig / grad_orig.max()
        if grad_recon.max() > 0:
            grad_recon = grad_recon / grad_recon.max()
        
        # Correlation
        return np.corrcoef(grad_orig.flatten(), grad_recon.flatten())[0, 1]
    
    @staticmethod
    def state_fidelity(psi_ideal: np.ndarray, psi_noisy: np.ndarray) -> float:
        """
        Calculate pure-state quantum fidelity.
        
        F = |<ψ_ideal|ψ_noisy>|²
        
        This is the true quantum-appropriate metric for state comparison.
        """
        psi_ideal = psi_ideal.flatten()
        psi_noisy = psi_noisy.flatten()
        
        # Ensure same size
        if len(psi_ideal) != len(psi_noisy):
            return 0.0
        
        # Inner product
        overlap = np.abs(np.vdot(psi_ideal, psi_noisy)) ** 2
        return min(1.0, max(0.0, overlap))
    
    @staticmethod
    def preparation_success_probability(counts: dict, target_states: set, total_shots: int) -> float:
        """
        Calculate probability that state survives above threshold.
        
        P_success = Σ counts[s] / total for s in target_states
        """
        success_count = sum(counts.get(s, 0) for s in target_states)
        return success_count / max(total_shots, 1)
    
    def calculate_comprehensive(
        self,
        original: np.ndarray,
        reconstructed: np.ndarray,
        circuit_metrics: Dict,
        technique: str,
        dataset: str,
        num_leaves: int = None,
        ssim_no_trex: float = None,
        encoding_time_ms: float = 0.0
    ) -> ComprehensiveMetrics:
        """
        Calculate all 12 metrics for CSV output.
        
        Args:
            original: Original image
            reconstructed: Reconstructed image
            circuit_metrics: Dict with circuit stats (from encoder)
            technique: Technique name
            dataset: Dataset name
            num_leaves: Number of quadtree leaves (for HEPAR)
            ssim_no_trex: SSIM without TREX (ablation)
            encoding_time_ms: Encoding time in milliseconds
            
        Returns:
            ComprehensiveMetrics dataclass
        """
        N = original.shape[0]
        
        # Calculate image quality metrics
        psnr_val = self.psnr(original, reconstructed)
        ssim_val = self.ssim(original, reconstructed)
        fidelity_val = self.hellinger_fidelity(original, reconstructed)
        
        # Compression ratio (only meaningful for HEPAR)
        if num_leaves and technique in ['HEPAR']:
            cr = self.compression_ratio(N, num_leaves)
        else:
            cr = 1.0
        
        return ComprehensiveMetrics(
            Dataset=dataset,
            Technique=technique,
            Qubits_Logical=int(circuit_metrics.get('qubits_logical', 0)),
            Qubits_Physical=int(circuit_metrics.get('qubits_physical', 0)),
            Gate_Count_Clifford=int(circuit_metrics.get('gate_count', 0)),
            Circuit_Depth=int(circuit_metrics.get('circuit_depth', 0)),
            SWAP_Overhead=int(circuit_metrics.get('swap_overhead', 0)),
            Encoding_Time_ms=round(encoding_time_ms, 2),
            Fidelity_Hellinger=round(fidelity_val, 4),
            PSNR=round(psnr_val, 2),
            SSIM=round(ssim_val, 4),
            Ablation_NoTREX=round(ssim_no_trex if ssim_no_trex else ssim_val, 4),
            Compression_Ratio=round(cr, 2)
        )


class EntropyCalculator:
    """Calculate entropy metrics for image complexity analysis."""
    
    @staticmethod
    def shannon_entropy(image: np.ndarray, bins: int = 256) -> float:
        """
        Calculate Shannon entropy of image.
        
        H = -Σ p_i log₂(p_i)
        """
        # Histogram
        hist, _ = np.histogram(image.flatten(), bins=bins, range=(0, 1))
        hist = hist / hist.sum()  # Normalize to probability
        
        # Entropy
        hist = hist[hist > 0]  # Remove zeros
        return -np.sum(hist * np.log2(hist))
    
    @staticmethod
    def von_neumann_entropy(density_matrix: np.ndarray) -> float:
        """
        Calculate Von Neumann entropy of quantum state.
        
        S = -Tr(ρ log₂ ρ)
        """
        # Eigenvalues of density matrix
        eigenvalues = np.linalg.eigvalsh(density_matrix)
        eigenvalues = eigenvalues[eigenvalues > 1e-10]  # Remove near-zeros
        
        return -np.sum(eigenvalues * np.log2(eigenvalues))
    
    @staticmethod
    def image_complexity(image: np.ndarray) -> float:
        """
        Estimate image complexity (inverse of compressibility).
        
        Returns value in [0, 1] where 1 = maximum complexity (random noise).
        """
        # Use local variance as complexity proxy
        from scipy.ndimage import uniform_filter
        
        local_mean = uniform_filter(image, size=3)
        local_sqr_mean = uniform_filter(image**2, size=3)
        local_var = local_sqr_mean - local_mean**2
        
        # Average local variance normalized
        return np.clip(np.mean(local_var) * 10, 0, 1)


if __name__ == "__main__":
    # Demo
    print("Testing Metrics Calculator...")
    
    # Create test images
    original = np.random.rand(32, 32)
    reconstructed = original + 0.1 * np.random.randn(32, 32)
    reconstructed = np.clip(reconstructed, 0, 1)
    
    calc = MetricsCalculator()
    
    print(f"PSNR: {calc.psnr(original, reconstructed):.2f} dB")
    print(f"SSIM: {calc.ssim(original, reconstructed):.4f}")
    print(f"Hellinger Fidelity: {calc.hellinger_fidelity(original, reconstructed):.4f}")
    print(f"Spectral Fidelity: {calc.spectral_fidelity(original, reconstructed):.4f}")
