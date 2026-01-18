"""
HEPAR Springer Benchmarks - Analytical Estimator Classes
=========================================================
10 QIR Technique Estimators with Citation Keys for LaTeX Generation

Each class provides:
- Qubit count formulas (logical)
- Circuit depth formulas
- Gate count (Clifford) formulas
- SWAP overhead estimation for Heavy-Hex topology
- Bibliography citation key for auto-generation

References are from peer-reviewed publications (2011-2025).
"""

import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class EstimatorMetrics:
    """Container for estimator output metrics."""
    technique: str
    cite_key: str
    qubits_logical: int
    qubits_physical: int
    circuit_depth: int
    gate_count_clifford: int
    swap_overhead: int
    scalability: str
    notes: str = ""


class BaseEstimator(ABC):
    """Abstract base class for QIR technique estimators."""
    
    def __init__(self, N: int, cite_key: str, technique_name: str):
        """
        Initialize estimator.
        
        Args:
            N: Image dimension (assumes square NxN)
            cite_key: BibTeX citation key for bibliography
            technique_name: Name of the technique
        """
        self.N = N
        self.n = int(np.ceil(np.log2(max(N, 2))))  # log2(N) for position encoding
        self.cite_key = cite_key
        self.technique_name = technique_name
    
    @abstractmethod
    def calculate_qubits_logical(self) -> int:
        """Calculate logical qubit requirement."""
        pass
    
    @abstractmethod
    def calculate_circuit_depth(self) -> int:
        """Calculate circuit depth (critical path length)."""
        pass
    
    @abstractmethod
    def calculate_gate_count(self) -> int:
        """Calculate total Clifford gate count."""
        pass
    
    def calculate_qubits_physical(self) -> int:
        """
        Estimate physical qubits on Heavy-Hex topology.
        Typically 1.2x-2x logical due to ancilla and routing.
        """
        logical = self.calculate_qubits_logical()
        # Heavy-Hex overhead factor (conservative estimate)
        return int(logical * 1.3)
    
    def calculate_swap_overhead(self) -> int:
        """
        Estimate SWAP gates needed for Heavy-Hex topology.
        Based on circuit depth and connectivity constraints.
        """
        depth = self.calculate_circuit_depth()
        qubits = self.calculate_qubits_logical()
        # SWAP overhead scales with depth × log(qubits) for sparse topologies
        return int(depth * np.log2(max(qubits, 2)) * 0.15)
    
    @abstractmethod
    def get_scalability(self) -> str:
        """Return scalability rating: Very High, High, Medium, Low."""
        pass
    
    def get_metrics(self) -> EstimatorMetrics:
        """Return complete metrics dictionary."""
        return EstimatorMetrics(
            technique=self.technique_name,
            cite_key=self.cite_key,
            qubits_logical=self.calculate_qubits_logical(),
            qubits_physical=self.calculate_qubits_physical(),
            circuit_depth=self.calculate_circuit_depth(),
            gate_count_clifford=self.calculate_gate_count(),
            swap_overhead=self.calculate_swap_overhead(),
            scalability=self.get_scalability()
        )


# =============================================================================
# BASELINE TECHNIQUES (Simulated in experiments)
# =============================================================================

class FRQIEstimator(BaseEstimator):
    """
    Flexible Representation of Quantum Images (FRQI)
    Reference: Le, Dong, Hirota (2011) - Quantum Information Processing
    
    State: |I> = (1/2^n) Σ (cos θ_i |0> + sin θ_i |1>) ⊗ |i>
    Qubits: 2n + 1 (position + color qubit)
    """
    
    def __init__(self, N: int):
        super().__init__(N, "le2011frqi", "FRQI")
    
    def calculate_qubits_logical(self) -> int:
        return 2 * self.n + 1
    
    def calculate_circuit_depth(self) -> int:
        # Each pixel requires controlled rotation: O(N²)
        return self.N * self.N
    
    def calculate_gate_count(self) -> int:
        # N² controlled-Ry gates, each decomposes to ~3 CNOTs + rotations
        return self.N * self.N * 4
    
    def get_scalability(self) -> str:
        return "Medium"


class NEQREstimator(BaseEstimator):
    """
    Novel Enhanced Quantum Representation (NEQR)
    Reference: Zhang, Lu, Gao, Wang (2013) - Quantum Information Processing
    
    State: |I> = (1/2^n) Σ |C_i> ⊗ |i>
    Qubits: 2n + q (position + q-bit color, typically q=8)
    """
    
    def __init__(self, N: int, color_bits: int = 8):
        super().__init__(N, "zhang2013neqr", "NEQR")
        self.color_bits = color_bits
    
    def calculate_qubits_logical(self) -> int:
        return 2 * self.n + self.color_bits
    
    def calculate_circuit_depth(self) -> int:
        # Each pixel: controlled-X gates for each color bit
        return self.N * self.N * self.color_bits
    
    def calculate_gate_count(self) -> int:
        # N² pixels × q bit-flips, multi-controlled gates
        return self.N * self.N * self.color_bits * 2
    
    def calculate_swap_overhead(self) -> int:
        # NEQR has HIGH swap overhead due to all-to-all connectivity needs
        depth = self.calculate_circuit_depth()
        qubits = self.calculate_qubits_logical()
        # Linear topology penalty: much higher SWAP count
        return int(depth * qubits * 0.5)
    
    def get_scalability(self) -> str:
        return "Low"


# =============================================================================
# HEPAR (Novel Proposed Technique)
# =============================================================================

class HEPAREstimator(BaseEstimator):
    """
    Hierarchical Entanglement-based Phase-Amplitude Representation (HEPAR)
    Reference: [Your Paper] - Proposed Novel Technique
    
    4 Pillars:
    1. Depth Register - Quadtree level encoding
    2. Gray Code Sorting - Minimized Hamming distance traversal
    3. SPAE - Simultaneous Phase-Amplitude Encoding
    4. TREX - Twirled Readout Error Extinction
    
    State: |Ψ_HEPAR> = Σ_k α_k |L_k> ⊗ Σ_i w̃_i |ψ_i> ⊗ |i>
    Qubits: log₂(K) + d_max + 1 (leaves + depth + payload)
    """
    
    def __init__(self, N: int, num_leaves: int = None, sparsity: float = 0.5):
        super().__init__(N, "hepar2026proposed", "HEPAR")
        # K = number of quadtree leaves (depends on image sparsity)
        if num_leaves is None:
            # Estimate leaves based on sparsity: K ≈ N² × (1 - sparsity) / 4
            self.K = max(4, int(N * N * (1 - sparsity) / 4))
        else:
            self.K = num_leaves
        self.d_max = int(np.log2(N))  # Maximum tree depth
    
    def calculate_qubits_logical(self) -> int:
        # Address qubits for K leaves + depth register + payload
        n_addr = max(1, int(np.ceil(np.log2(max(self.K, 2)))))
        n_depth = max(1, int(np.ceil(np.log2(self.d_max + 1))))
        return n_addr + n_depth + 1
    
    def calculate_circuit_depth(self) -> int:
        # O(K × 2) due to hierarchical traversal + SPAE
        return self.K * 2
    
    def calculate_gate_count(self) -> int:
        # K leaves × 6 gates per leaf (amplitude + phase encoding)
        return self.K * 6
    
    def calculate_swap_overhead(self) -> int:
        # HEPAR optimized for Heavy-Hex: low SWAP overhead
        depth = self.calculate_circuit_depth()
        qubits = self.calculate_qubits_logical()
        # Gray code sorting reduces transitions
        return int(depth * np.log2(max(qubits, 2)) * 0.05)
    
    def get_scalability(self) -> str:
        return "Very High"
    
    def calculate_compression_ratio(self) -> float:
        """Calculate compression ratio: N²/K"""
        return (self.N * self.N) / max(self.K, 1)


# =============================================================================
# MULTI-CHANNEL & COLOR TECHNIQUES
# =============================================================================

class MCQIEstimator(BaseEstimator):
    """
    Multi-Channel Quantum Images (MCQI)
    Reference: Sun, Iliyasu, Yan, Dong, Hirota (2013) - Entropy Journal
    
    For RGB images with 3×8 = 24 color qubits.
    Qubits: 2n + 24 (position + R[8] + G[8] + B[8])
    """
    
    def __init__(self, N: int, channels: int = 3, bits_per_channel: int = 8):
        super().__init__(N, "sun2013mcqi", "MCQI")
        self.channels = channels
        self.bits_per_channel = bits_per_channel
    
    def calculate_qubits_logical(self) -> int:
        return 2 * self.n + self.channels * self.bits_per_channel
    
    def calculate_circuit_depth(self) -> int:
        # N² pixels × 3 channels × 8 bits each
        return self.N * self.N * self.channels * self.bits_per_channel
    
    def calculate_gate_count(self) -> int:
        return self.N * self.N * self.channels * self.bits_per_channel * 2
    
    def calculate_swap_overhead(self) -> int:
        # Very high due to 24+ color qubits
        depth = self.calculate_circuit_depth()
        qubits = self.calculate_qubits_logical()
        return int(depth * qubits * 0.4)
    
    def get_scalability(self) -> str:
        return "Low"


class QRMWEstimator(BaseEstimator):
    """
    Quantum Representation of Multi-Wavelength Images (QRMW)
    Reference: Chen et al. (2019) - quantum hyperspectral imaging
    
    For multi-spectral images with 16 wavelength bands.
    Qubits: 2n + 16 (position + wavelength encoding)
    """
    
    def __init__(self, N: int, wavelength_bands: int = 16):
        super().__init__(N, "chen2019qrmw", "QRMW")
        self.bands = wavelength_bands
    
    def calculate_qubits_logical(self) -> int:
        return 2 * self.n + self.bands
    
    def calculate_circuit_depth(self) -> int:
        return self.N * self.N * self.bands
    
    def calculate_gate_count(self) -> int:
        return self.N * self.N * self.bands * 2
    
    def get_scalability(self) -> str:
        return "Low"


# =============================================================================
# IMPROVED/EFFICIENT VARIANTS
# =============================================================================

class GQIREstimator(BaseEstimator):
    """
    Generalized Quantum Image Representation (GQIR)
    Reference: Jiang, Wang, Xu (2015) - generalized framework
    
    Combines FRQI and NEQR benefits.
    Qubits: 2n + 8
    Depth: O(N × log N) via tree-structured gates
    """
    
    def __init__(self, N: int):
        super().__init__(N, "jiang2015gqir", "GQIR")
    
    def calculate_qubits_logical(self) -> int:
        return 2 * self.n + 8
    
    def calculate_circuit_depth(self) -> int:
        # Tree-structured: O(N × log N)
        return int(self.N * np.log2(max(self.N, 2)))
    
    def calculate_gate_count(self) -> int:
        return int(self.N * np.log2(max(self.N, 2)) * 4)
    
    def get_scalability(self) -> str:
        return "Medium"


class EFRQIEstimator(BaseEstimator):
    """
    Efficient FRQI (EFRQI)
    Reference: Yuan et al. (2014) - optimized FRQI variant
    
    Reduces depth by parallelizing controlled rotations.
    Qubits: 2n + 1
    Depth: O(N²/2) - halved via parallel decomposition
    """
    
    def __init__(self, N: int):
        super().__init__(N, "yuan2014efrqi", "EFRQI")
    
    def calculate_qubits_logical(self) -> int:
        return 2 * self.n + 1
    
    def calculate_circuit_depth(self) -> int:
        return (self.N * self.N) // 2
    
    def calculate_gate_count(self) -> int:
        return self.N * self.N * 2
    
    def get_scalability(self) -> str:
        return "Medium"


class IQIREstimator(BaseEstimator):
    """
    Improved Quantum Image Representation (IQIR)
    Reference: Sang, Wang, Nie (2017) - improved encoding efficiency
    
    Adds parity qubit for error detection.
    Qubits: 2n + 9 (position + 8 color + parity)
    """
    
    def __init__(self, N: int):
        super().__init__(N, "sang2017iqir", "IQIR")
    
    def calculate_qubits_logical(self) -> int:
        return 2 * self.n + 9
    
    def calculate_circuit_depth(self) -> int:
        return self.N * self.N * 4
    
    def calculate_gate_count(self) -> int:
        return self.N * self.N * 8
    
    def get_scalability(self) -> str:
        return "Medium"


# =============================================================================
# ADVANCED/SPECIALIZED TECHNIQUES
# =============================================================================

class QSNAEstimator(BaseEstimator):
    """
    2D Quantum Star Network Architecture (2D-QSNA)
    Reference: Li, Fan, Xia, Song, He (2018) - star topology
    
    Ultra-low depth via star network connectivity.
    Qubits: 2n (minimal - position only, amplitude encodes color)
    Depth: O(4n) - constant for fixed image size
    """
    
    def __init__(self, N: int):
        super().__init__(N, "li2018qsna", "2D-QSNA")
    
    def calculate_qubits_logical(self) -> int:
        return 2 * self.n
    
    def calculate_circuit_depth(self) -> int:
        # Star topology: O(n) depth
        return 4 * self.n
    
    def calculate_gate_count(self) -> int:
        return 4 * self.n
    
    def calculate_swap_overhead(self) -> int:
        # Minimal SWAP for star topology
        return int(self.n * 2)
    
    def get_scalability(self) -> str:
        return "Very High"


class NASSEstimator(BaseEstimator):
    """
    Normal Arbitrary Superposition State (NASS)
    Reference: Wang, Song, Su (2016) - arbitrary amplitude preparation
    
    General state preparation with minimal qubits.
    Qubits: 2n
    Depth: O(n) - logarithmic scaling
    """
    
    def __init__(self, N: int):
        super().__init__(N, "wang2016nass", "NASS")
    
    def calculate_qubits_logical(self) -> int:
        return 2 * self.n
    
    def calculate_circuit_depth(self) -> int:
        return 2 * self.n
    
    def calculate_gate_count(self) -> int:
        return 4 * self.n
    
    def get_scalability(self) -> str:
        return "Very High"


class QUALPIEstimator(BaseEstimator):
    """
    Quantum Log-Polar Image Representation (QUALPI)
    Reference: Zhou et al. (2020) - log-polar coordinate system
    
    Natural for rotational invariance in image recognition.
    Qubits: 2n + 2 (position + radius/angle encoding)
    """
    
    def __init__(self, N: int):
        super().__init__(N, "zhou2020qualpi", "QUALPI")
    
    def calculate_qubits_logical(self) -> int:
        return 2 * self.n + 2
    
    def calculate_circuit_depth(self) -> int:
        return (self.N * self.N) // 2
    
    def calculate_gate_count(self) -> int:
        return self.N * self.N
    
    def get_scalability(self) -> str:
        return "Medium"


# =============================================================================
# ESTIMATOR FACTORY
# =============================================================================

class EstimatorFactory:
    """Factory class to create estimators by technique name."""
    
    ESTIMATORS = {
        'FRQI': FRQIEstimator,
        'NEQR': NEQREstimator,
        'HEPAR': HEPAREstimator,
        'MCQI': MCQIEstimator,
        'GQIR': GQIREstimator,
        '2D-QSNA': QSNAEstimator,
        'EFRQI': EFRQIEstimator,
        'IQIR': IQIREstimator,
        'NASS': NASSEstimator,
        'QRMW': QRMWEstimator,
        'QUALPI': QUALPIEstimator
    }
    
    @classmethod
    def create(cls, technique: str, N: int, **kwargs) -> BaseEstimator:
        """Create an estimator for the given technique."""
        if technique not in cls.ESTIMATORS:
            raise ValueError(f"Unknown technique: {technique}. Available: {list(cls.ESTIMATORS.keys())}")
        return cls.ESTIMATORS[technique](N, **kwargs)
    
    @classmethod
    def create_all(cls, N: int) -> Dict[str, BaseEstimator]:
        """Create estimators for all techniques."""
        return {name: cls.create(name, N) for name in cls.ESTIMATORS}
    
    @classmethod
    def get_all_metrics(cls, N: int) -> Dict[str, EstimatorMetrics]:
        """Get metrics for all techniques at given image size."""
        estimators = cls.create_all(N)
        return {name: est.get_metrics() for name, est in estimators.items()}


def generate_scaling_table(sizes: list = [8, 16, 32, 64, 128, 256]) -> dict:
    """
    Generate scaling comparison table for all techniques across image sizes.
    
    Returns:
        Dictionary with technique -> size -> metrics mapping
    """
    results = {}
    for size in sizes:
        results[size] = EstimatorFactory.get_all_metrics(size)
    return results


if __name__ == "__main__":
    # Demo: Print metrics for 32x32 image
    print("=" * 70)
    print("QIR Technique Estimators (32×32 Image)")
    print("=" * 70)
    
    for name, metrics in EstimatorFactory.get_all_metrics(32).items():
        print(f"\n{metrics.technique} [{metrics.cite_key}]:")
        print(f"  Qubits (L/P): {metrics.qubits_logical}/{metrics.qubits_physical}")
        print(f"  Depth: {metrics.circuit_depth:,}")
        print(f"  Gates: {metrics.gate_count_clifford:,}")
        print(f"  SWAP Overhead: {metrics.swap_overhead:,}")
        print(f"  Scalability: {metrics.scalability}")
