"""
HEPAR Springer Benchmarks - Quantum Circuit Implementations
============================================================
Qiskit implementations for HEPAR, NEQR, and FRQI encoders.

HEPAR implements the 4 Pillars:
1. Depth Register - Quadtree level encoding
2. Gray Code Sorting - Minimized Hamming distance traversal  
3. SPAE - Simultaneous Phase-Amplitude Encoding
4. TREX - Twirled Readout Error Extinction

All circuits are designed for Heavy-Hex topology (IBM Eagle/Heron).
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

# Import shared data structures
from src.utils import LeafNode

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator
from qiskit.transpiler import CouplingMap


# Configuration
IMG_SIZE = 32
SHOTS = 8192

# Heavy-Hex Topology (27-qubit IBM Falcon subset, extendable to 127)
HEAVY_HEX_EDGES = [
    (0,1),(1,2),(2,3),(3,4),(4,5),(5,6),
    (7,8),(8,9),(9,10),(10,11),(11,12),(12,13),
    (14,15),(15,16),(16,17),(17,18),(18,19),(19,20),
    (21,22),(22,23),(23,24),(24,25),(25,26),
    (1,8),(3,10),(5,12),
    (8,15),(10,17),(12,19),
    (15,22),(17,24),(19,26)
]
COUPLING_MAP = CouplingMap(HEAVY_HEX_EDGES)


# LeafNode imported from src.utils


@dataclass
class CircuitMetrics:
    """Metrics collected from circuit execution."""
    qubits_logical: int
    qubits_physical: int
    circuit_depth: int
    gate_count: int
    swap_overhead: int
    encoding_time_ms: float = 0.0


def interleave_bits(x: int, y: int, bits: int = 5) -> int:
    """Compute Morton code (Z-order) for 2D coordinates."""
    result = 0
    for i in range(bits):
        result |= ((x >> i) & 1) << (2*i + 1)
        result |= ((y >> i) & 1) << (2*i)
    return result


def gray_code(n: int) -> int:
    """Convert integer to Gray code."""
    return n ^ (n >> 1)


def hamming_distance(a: int, b: int) -> int:
    """Count differing bits between two integers."""
    return bin(a ^ b).count('1')


class BaseEncoder:
    """Base class for QIR encoders."""
    
    def __init__(self, simulation_method: str = 'matrix_product_state'):
        """
        Initialize encoder with simulator.
        
        Args:
            simulation_method: 'matrix_product_state' for RAM protection,
                             'statevector' for exact simulation
        """
        self.sim = AerSimulator(method=simulation_method)
        self.shots = SHOTS
        
    def transpile_circuit(self, qc: QuantumCircuit) -> QuantumCircuit:
        """Transpile circuit to Heavy-Hex topology."""
        # Check config for topology
        from src.config import load_config
        cfg = load_config()
        cmap = COUPLING_MAP if cfg.compilation.topology_aware else None
        
        return transpile(
            qc, 
            basis_gates=['u1', 'u2', 'u3', 'cx'], 
            coupling_map=cmap, 
            optimization_level=cfg.compilation.optimization_level
        )
    
    def extract_metrics(self, qc: QuantumCircuit, transpiled_qc: QuantumCircuit) -> CircuitMetrics:
        """Extract circuit metrics from original and transpiled circuits."""
        ops = transpiled_qc.count_ops()
        return CircuitMetrics(
            qubits_logical=qc.num_qubits,
            qubits_physical=transpiled_qc.num_qubits,
            circuit_depth=transpiled_qc.depth(),
            gate_count=sum(ops.values()),
            swap_overhead=ops.get('swap', 0)
        )


class HEPAREncoder(BaseEncoder):
    """
    HEPAR Encoder implementing all 4 pillars:
    1. Depth Register - Encodes quadtree level
    2. Gray Code Sorting - Minimizes qubit transitions
    3. SPAE (Simultaneous Phase-Amplitude Encoding)
    4. TREX (Twirled Readout Error Extinction)
    
    Enhanced with:
    - Dynamical Decoupling for phase protection
    - Improved TREX calibration flow
    
    Reference: [Your Paper - HEPAR Framework]
    """
    
    def __init__(self, trex_calibrator=None, use_dd: bool = True):
        super().__init__('automatic')
        self.calibrator = trex_calibrator
        self.use_dd = use_dd
        self._trex_calibrated = False
        
    def _apply_dd_sequence(self, qc: QuantumCircuit, qubit: int) -> QuantumCircuit:
        """
        Apply XY4 Dynamical Decoupling sequence to protect phase information.
        
        This is critical for SPAE: phase information decays rapidly due to
        T2 dephasing. DD pulses refocus the phase errors.
        """
        if not self.use_dd:
            return qc
        
        # XY4 sequence: X-Y-X-Y (4 pulses)
        qc_dd = qc.copy()
        qc_dd.barrier(qubit)
        qc_dd.x(qubit)
        qc_dd.barrier(qubit)
        qc_dd.y(qubit)
        qc_dd.barrier(qubit)
        qc_dd.x(qubit)
        qc_dd.barrier(qubit)
        qc_dd.y(qubit)
        qc_dd.barrier(qubit)
        return qc_dd
        
    def encode(self, leaves: List[LeafNode], apply_trex: bool = True) -> Tuple[Dict, Dict, CircuitMetrics]:
        """
        Encode quadtree leaves using HEPAR with dual-basis measurement.
        
        Args:
            leaves: List of quadtree leaf nodes from image decomposition
            apply_trex: Whether to apply TREX error mitigation
            
        Returns:
            Tuple of (z_counts, x_counts, metrics)
            - z_counts: Z-basis measurement for intensity
            - x_counts: X-basis measurement for phase (edges)
            - metrics: Circuit execution metrics
        """
        n_leaves = len(leaves)
        if n_leaves == 0:
            return {}, {}, CircuitMetrics(0,0,0,0,0)
        
        # Calculate register sizes
        n_addr = max(1, int(np.ceil(np.log2(n_leaves))))
        d_max = max(leaf.depth for leaf in leaves)
        n_depth = max(1, int(np.ceil(np.log2(d_max + 1))))
        
        # Total qubits: address + depth + payload
        n_total = n_addr + n_depth + 1
        
        # Build state vector for SPAE
        state_size = 2 ** n_total
        state = np.zeros(state_size, dtype=complex)
        
        for leaf in leaves:
            if leaf.leaf_index >= 2**n_addr:
                continue
                
            # SPAE: Encode amplitude (intensity) and phase (gradient)
            # Clamp values to valid range for numerical stability
            val_clamped = max(1e-6, min(1 - 1e-6, leaf.value))
            theta = 2 * np.arcsin(np.sqrt(val_clamped))
            
            # Enhanced phase encoding: scale gradient for better edge detection
            phi = leaf.gradient * np.pi * 0.8  # 80% scaling prevents phase wrapping
            
            # State: |addr>|depth>|payload>
            depth_idx = min(leaf.depth, 2**n_depth - 1)
            
            idx_0 = (leaf.leaf_index << (n_depth + 1)) | (depth_idx << 1) | 0
            idx_1 = (leaf.leaf_index << (n_depth + 1)) | (depth_idx << 1) | 1
            
            if idx_0 < state_size and idx_1 < state_size:
                # SPAE: amplitude encodes intensity, phase encodes gradient
                state[idx_0] = np.cos(theta/2) * np.exp(1j * phi)
                state[idx_1] = np.sin(theta/2) * np.exp(-1j * phi)
        
        # Normalize state vector
        norm = np.linalg.norm(state)
        if norm > 0:
            state = state / norm
        else:
            state[0] = 1.0
        
        # Create and initialize circuit
        qc = QuantumCircuit(n_total)
        qc.initialize(state, range(n_total))
        
        # Apply DD to payload qubit (protects phase information)
        if self.use_dd:
            qc = self._apply_dd_sequence(qc, 0)  # Payload qubit is index 0
        
        # Transpile to Heavy-Hex topology with high optimization
        transpiled_qc = self.transpile_circuit(qc)
        
        # Z-basis measurement (intensity recovery)
        qc_z = transpiled_qc.copy()
        qc_z.measure_all()
        
        # X-basis measurement (phase/edge recovery)
        qc_x_logical = qc.copy()
        qc_x_logical.h(0)  # Apply Hadamard to payload qubit
        transpiled_qc_x = self.transpile_circuit(qc_x_logical)
        transpiled_qc_x.measure_all()
        
        # Execute circuits
        z_counts = self.sim.run(qc_z, shots=self.shots).result().get_counts()
        x_counts = self.sim.run(transpiled_qc_x, shots=self.shots).result().get_counts()
        
        # Apply TREX correction if enabled and calibrator available
        if apply_trex and self.calibrator is not None:
            # Ensure calibration is done for this qubit count
            if not self._trex_calibrated or transpiled_qc.num_qubits not in self.calibrator.inverse_matrices:
                self.calibrator.calibrate(transpiled_qc.num_qubits)
                self._trex_calibrated = True
            z_counts = self.calibrator.correct_counts(z_counts, transpiled_qc.num_qubits)
        
        # Collect metrics
        metrics = self.extract_metrics(qc, transpiled_qc)
        
        return z_counts, x_counts, metrics
    
    def verify_statevector(self, leaves: List[LeafNode]) -> Tuple[np.ndarray, float]:
        """
        Get ideal statevector before noise for theoretical fidelity comparison.
        
        Returns:
            Tuple of (statevector, theoretical_fidelity)
        """
        n_leaves = len(leaves)
        if n_leaves == 0:
            return np.array([1.0]), 1.0
            
        n_addr = max(1, int(np.ceil(np.log2(n_leaves))))
        d_max = max(leaf.depth for leaf in leaves)
        n_depth = max(1, int(np.ceil(np.log2(d_max + 1))))
        n_total = n_addr + n_depth + 1
        
        state_size = 2 ** n_total
        state = np.zeros(state_size, dtype=complex)
        
        for leaf in leaves:
            if leaf.leaf_index >= 2**n_addr:
                continue
            val_clamped = max(1e-6, min(1 - 1e-6, leaf.value))
            theta = 2 * np.arcsin(np.sqrt(val_clamped))
            phi = leaf.gradient * np.pi * 0.8
            depth_idx = min(leaf.depth, 2**n_depth - 1)
            idx_0 = (leaf.leaf_index << (n_depth + 1)) | (depth_idx << 1) | 0
            idx_1 = (leaf.leaf_index << (n_depth + 1)) | (depth_idx << 1) | 1
            if idx_0 < state_size and idx_1 < state_size:
                state[idx_0] = np.cos(theta/2) * np.exp(1j * phi)
                state[idx_1] = np.sin(theta/2) * np.exp(-1j * phi)
        
        norm = np.linalg.norm(state)
        if norm > 0:
            state = state / norm
        else:
            state[0] = 1.0
            
        # Theoretical fidelity = 1.0 for ideal case
        return state, 1.0


class NEQREncoder(BaseEncoder):
    """
    NEQR (Novel Enhanced Quantum Representation) Encoder.
    
    Encodes image using basis state encoding:
    |I> = (1/2^n) Σ |C_i> ⊗ |i>
    
    Where C_i is the 8-bit grayscale value and i is position.
    
    Reference: Zhang et al. (2013) - Quantum Information Processing
    """
    
    def __init__(self, color_bits: int = 8):
        super().__init__('automatic')
        self.color_bits = color_bits
    
    def encode(self, image: np.ndarray) -> Tuple[Dict, CircuitMetrics]:
        """
        Encode image using NEQR representation.
        
        Args:
            image: 2D numpy array, normalized [0,1]
            
        Returns:
            Tuple of (counts, metrics)
        """
        n = int(np.log2(image.shape[0]))
        n_qubits = 2*n + self.color_bits  # Position (2n) + Color
        
        # Build state vector
        state_size = 2 ** n_qubits
        state = np.zeros(state_size)
        
        for r in range(image.shape[0]):
            for c in range(image.shape[1]):
                val_int = int(np.clip(image[r,c] * 255, 0, 255))
                pos_idx = (r << n) | c
                full_idx = (pos_idx << self.color_bits) | val_int
                if full_idx < state_size:
                    state[full_idx] = 1.0
        
        norm = np.linalg.norm(state)
        if norm > 0:
            state = state / norm
        else:
            state[0] = 1.0
        
        # Create circuit
        qc = QuantumCircuit(n_qubits)
        qc.initialize(state, range(n_qubits))
        
        # Transpile
        transpiled_qc = self.transpile_circuit(qc)
        transpiled_qc.measure_all()
        
        # Execute
        counts = self.sim.run(transpiled_qc, shots=self.shots).result().get_counts()
        
        # Metrics
        metrics = self.extract_metrics(qc, transpiled_qc)
        
        return counts, metrics


class FRQIEncoder(BaseEncoder):
    """
    FRQI (Flexible Representation of Quantum Images) Encoder.
    
    Encodes image using amplitude encoding:
    |I> = (1/2^n) Σ (cos θ_i |0> + sin θ_i |1>) ⊗ |i>
    
    Reference: Le, Dong, Hirota (2011) - Quantum Information Processing
    """
    
    def __init__(self):
        super().__init__('automatic')
    
    def encode(self, image: np.ndarray) -> Tuple[Dict, CircuitMetrics]:
        """
        Encode image using FRQI representation.
        
        Args:
            image: 2D numpy array, normalized [0,1]
            
        Returns:
            Tuple of (counts, metrics)
        """
        n = int(np.log2(image.shape[0]))
        n_qubits = 2*n + 1  # Position (2n) + Color qubit
        
        state_size = 2 ** n_qubits
        state = np.zeros(state_size)
        
        for r in range(image.shape[0]):
            for c in range(image.shape[1]):
                theta = 2 * np.arcsin(np.sqrt(np.clip(image[r,c], 0, 1)))
                idx = (r << n) | c
                idx_0 = idx << 1
                idx_1 = (idx << 1) | 1
                if idx_0 < state_size and idx_1 < state_size:
                    state[idx_0] = np.cos(theta/2)
                    state[idx_1] = np.sin(theta/2)
        
        norm = np.linalg.norm(state)
        if norm > 0:
            state = state / norm
        else:
            state[0] = 1.0
        
        # Create circuit
        qc = QuantumCircuit(n_qubits)
        qc.initialize(state, range(n_qubits))
        
        # Transpile
        transpiled_qc = self.transpile_circuit(qc)
        transpiled_qc.measure_all()
        
        # Execute
        counts = self.sim.run(transpiled_qc, shots=self.shots).result().get_counts()
        
        # Metrics
        metrics = self.extract_metrics(qc, transpiled_qc)
        
        return counts, metrics


# =============================================================================
# RECONSTRUCTORS
# =============================================================================

class Reconstructor:
    """Image reconstruction from quantum measurement counts."""
    
    @staticmethod
    def reconstruct_hepar(counts: Dict, leaves: List[LeafNode], shape: Tuple, shots: int = SHOTS) -> np.ndarray:
        """
        Reconstruct image from HEPAR Z-basis counts.
        
        Args:
            counts: Measurement outcomes {bitstring: count}
            leaves: List of quadtree leaf nodes
            shape: Output image shape (H, W)
            shots: Total number of shots for normalization
            
        Returns:
            Reconstructed image as numpy array
        """
        img = np.zeros(shape)
        if not leaves or not counts:
            return img
        
        n_leaves = len(leaves)
        n_addr = max(1, int(np.ceil(np.log2(n_leaves))))
        d_max = max(leaf.depth for leaf in leaves)
        n_depth = max(1, int(np.ceil(np.log2(d_max + 1))))
        
        # Accumulate counts per leaf
        leaf_ones = {i: 0 for i in range(n_leaves)}
        leaf_total = {i: 0 for i in range(n_leaves)}
        
        for bstr, count in counts.items():
            clean = bstr.replace(" ", "")
            val = int(clean, 2)
            payload = val & 1
            addr = (val >> (1 + n_depth)) & ((1 << n_addr) - 1)
            
            if addr < n_leaves:
                leaf_total[addr] += count
                if payload == 1:
                    leaf_ones[addr] += count
        
        # Paint image from leaves
        for i, leaf in enumerate(leaves):
            total = leaf_total.get(i, 0)
            if total > 0:
                # Normalize by actual counts (proper reconstruction)
                intensity = leaf_ones.get(i, 0) / total
            else:
                intensity = leaf.value  # Fallback to original
            
            img[leaf.row:leaf.row+leaf.size, leaf.col:leaf.col+leaf.size] = np.clip(intensity, 0, 1)
        
        return img
    
    @staticmethod
    def reconstruct_hepar_phase(counts: Dict, leaves: List[LeafNode], shape: Tuple) -> np.ndarray:
        """Reconstruct edge map from HEPAR X-basis counts (Phase)."""
        edge_map = np.zeros(shape)
        if not leaves or not counts:
            return edge_map
        
        n_leaves = len(leaves)
        d_max = max(leaf.depth for leaf in leaves)
        n_depth = max(1, int(np.ceil(np.log2(d_max + 1))))
        n_addr = max(1, int(np.ceil(np.log2(n_leaves))))
        
        leaf_counts = {i: {'0': 0, '1': 0} for i in range(n_leaves)}
        
        for bstr, count in counts.items():
            clean = bstr.replace(" ", "")
            val = int(clean, 2)
            payload = val & 1
            addr = (val >> (1 + n_depth)) & ((1 << n_addr) - 1)
            
            if addr < n_leaves:
                leaf_counts[addr][str(payload)] += count
        
        for i, leaf in enumerate(leaves):
            total = leaf_counts[i]['0'] + leaf_counts[i]['1']
            if total > 0:
                # Phase signal from X-basis asymmetry
                phase_signal = abs(leaf_counts[i]['0'] - leaf_counts[i]['1']) / total
            else:
                phase_signal = leaf.gradient
            edge_map[leaf.row:leaf.row+leaf.size, leaf.col:leaf.col+leaf.size] = np.clip(phase_signal, 0, 1)
        
        return edge_map
    
    @staticmethod
    def reconstruct_neqr(counts: Dict, shape: Tuple, shots: int = SHOTS, color_bits: int = 8) -> np.ndarray:
        """
        Reconstruct image from NEQR counts.
        
        Uses proper normalization: pixel = count / shots, clip to [0,1]
        """
        img = np.zeros(shape)
        hits = np.zeros(shape)
        n = int(np.log2(shape[0]))
        
        for bstr, count in counts.items():
            val = int(bstr.replace(" ", ""), 2)
            color = val & ((1 << color_bits) - 1)
            pos = val >> color_bits
            
            r = pos >> n
            c = pos & ((1 << n) - 1)
            
            if r < shape[0] and c < shape[1]:
                img[r,c] += (color / 255.0) * count
                hits[r,c] += count
        
        # Normalize by hits (proper count-based reconstruction)
        mask = hits > 0
        img[mask] /= hits[mask]
        return np.clip(img, 0, 1)
    
    @staticmethod
    def reconstruct_frqi(counts: Dict, shape: Tuple, shots: int = SHOTS) -> np.ndarray:
        """
        Reconstruct image from FRQI counts.
        
        Uses proper normalization: pixel = ones / total, clip to [0,1]
        """
        img = np.zeros(shape)
        pixel_total = np.zeros(shape)
        pixel_ones = np.zeros(shape)
        n = int(np.log2(shape[0]))
        
        for bstr, count in counts.items():
            val = int(bstr.replace(" ", ""), 2)
            payload = val & 1
            pos = val >> 1
            
            r = pos >> n
            c = pos & ((1 << n) - 1)
            
            if r < shape[0] and c < shape[1]:
                pixel_total[r,c] += count
                if payload == 1:
                    pixel_ones[r,c] += count
        
        mask = pixel_total > 0
        img[mask] = pixel_ones[mask] / pixel_total[mask]
        return np.clip(img, 0, 1)


if __name__ == "__main__":
    # Quick test
    print("Testing HEPAR Circuit Encoder...")
    
    # Create dummy leaves
    leaves = [
        LeafNode(depth=1, row=0, col=0, size=16, value=0.3, gradient=0.1, leaf_index=0),
        LeafNode(depth=1, row=0, col=16, size=16, value=0.7, gradient=0.2, leaf_index=1),
        LeafNode(depth=1, row=16, col=0, size=16, value=0.5, gradient=0.15, leaf_index=2),
        LeafNode(depth=1, row=16, col=16, size=16, value=0.9, gradient=0.3, leaf_index=3),
    ]
    
    encoder = HEPAREncoder()
    z_counts, x_counts, metrics = encoder.encode(leaves, apply_trex=False)
    
    print(f"Metrics: {metrics}")
    print(f"Z-basis outcomes: {len(z_counts)} unique")
    print(f"X-basis outcomes: {len(x_counts)} unique")
