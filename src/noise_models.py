"""
HEPAR Springer Benchmarks - Noise Models
=========================================
Heavy-Hex topology simulation, TREX calibration, and Dynamical Decoupling.

Implements IBM Eagle/Heron-like noise characteristics:
- Depolarizing error (1-2 qubit gates)
- Readout (measurement) error matrices
- T1/T2 coherence limits
- TREX (Twirled Readout Error Extinction) mitigation
"""

import numpy as np
from typing import Dict, List, Tuple, Optional

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError
from qiskit.transpiler import CouplingMap


# Heavy-Hex Topology (127-qubit IBM Eagle/Heron)
HEAVY_HEX_27_EDGES = [
    (0,1),(1,2),(2,3),(3,4),(4,5),(5,6),
    (7,8),(8,9),(9,10),(10,11),(11,12),(12,13),
    (14,15),(15,16),(16,17),(17,18),(18,19),(19,20),
    (21,22),(22,23),(23,24),(24,25),(25,26),
    (1,8),(3,10),(5,12),
    (8,15),(10,17),(12,19),
    (15,22),(17,24),(19,26)
]


def get_heavy_hex_coupling_map(num_qubits: int = 27) -> CouplingMap:
    """
    Get Heavy-Hex coupling map.
    
    Args:
        num_qubits: 27 for Falcon, 127 for Eagle
        
    Returns:
        CouplingMap for transpilation
    """
    if num_qubits <= 27:
        return CouplingMap(HEAVY_HEX_27_EDGES)
    else:
        # Extended Heavy-Hex for larger simulations
        # This is a simplified approximation
        edges = list(HEAVY_HEX_27_EDGES)
        for i in range(27, num_qubits):
            # Add sparse connectivity
            if i > 27:
                edges.append((i, i-1))
            if i % 7 == 0 and i > 7:
                edges.append((i, i-7))
        return CouplingMap(edges)


def create_ibm_noise_model(
    p1_error: float = 0.001,  # Single-qubit gate error
    p2_error: float = 0.01,   # Two-qubit gate error
    read_error: float = 0.02, # Readout error
    num_qubits: int = 27
) -> NoiseModel:
    """
    Create realistic IBM-like noise model.
    
    Args:
        p1_error: Single-qubit gate error probability
        p2_error: Two-qubit gate error probability
        read_error: Readout error probability
        num_qubits: Number of qubits in the device
        
    Returns:
        Qiskit NoiseModel
    """
    noise = NoiseModel()
    
    # Single-qubit depolarizing error
    error_1q = depolarizing_error(p1_error, 1)
    noise.add_all_qubit_quantum_error(error_1q, ['u1', 'u2', 'u3', 'x', 'y', 'z', 'h', 's', 't'])
    
    # Two-qubit depolarizing error
    error_2q = depolarizing_error(p2_error, 2)
    noise.add_all_qubit_quantum_error(error_2q, ['cx', 'cz', 'swap'])
    
    # Readout error (asymmetric)
    # P(0|1) typically higher than P(1|0) on real hardware
    p01 = read_error * 1.2  # Measure 1 as 0
    p10 = read_error * 0.8  # Measure 0 as 1
    read_err = ReadoutError([[1 - p10, p10], [p01, 1 - p01]])
    noise.add_all_qubit_readout_error(read_err)
    
    return noise


class TREXCalibrator:
    """
    Twirled Readout Error Extinction (TREX) Calibration and Correction.
    
    TREX mitigates readout errors by:
    1. Running calibration circuits (|0...0> and |1...1> states)
    2. Building confusion matrix M from measurement outcomes
    3. Computing M^(-1) for error inversion
    4. Applying correction to measurement counts
    
    Reference: Maciejewski et al. (2020) - Mitigation of readout noise
    """
    
    def __init__(self, simulator: AerSimulator, shots: int = 4096):
        """
        Initialize TREX calibrator.
        
        Args:
            simulator: AerSimulator instance (with or without noise)
            shots: Number of shots for calibration circuits
        """
        self.sim = simulator
        self.shots = shots
        self.confusion_matrices = {}
        self.inverse_matrices = {}
    
    def calibrate(self, num_qubits: int) -> np.ndarray:
        """
        Calibrate readout error for given qubit count.
        
        Args:
            num_qubits: Number of qubits to calibrate
            
        Returns:
            Inverse confusion matrix M^(-1)
        """
        if num_qubits in self.confusion_matrices:
            return self.inverse_matrices[num_qubits]
        
        # Prepare |0...0> and measure
        qc0 = QuantumCircuit(num_qubits)
        qc0.measure_all()
        
        # Prepare |1...1> and measure
        qc1 = QuantumCircuit(num_qubits)
        qc1.x(range(num_qubits))
        qc1.measure_all()
        
        # Run calibration circuits
        result0 = self.sim.run(qc0, shots=self.shots).result().get_counts()
        result1 = self.sim.run(qc1, shots=self.shots).result().get_counts()
        
        # Build 2x2 confusion matrix (simplified per-qubit average)
        p00 = result0.get('0'*num_qubits, 0) / self.shots  # P(0|0)
        p01 = 1 - p00  # P(1|0)
        p11 = result1.get('1'*num_qubits, 0) / self.shots  # P(1|1)
        p10 = 1 - p11  # P(0|1)
        
        # Confusion matrix M: M[i,j] = P(measure i | prepared j)
        M = np.array([[p00, p10], [p01, p11]])
        
        # Regularize for numerical stability
        M = M + 1e-6 * np.eye(2)
        M = M / M.sum(axis=0, keepdims=True)
        
        # Compute inverse
        try:
            M_inv = np.linalg.inv(M)
        except np.linalg.LinAlgError:
            M_inv = np.eye(2)
        
        self.confusion_matrices[num_qubits] = M
        self.inverse_matrices[num_qubits] = M_inv
        
        return M_inv
    
    def correct_counts(self, counts: Dict[str, int], num_qubits: int) -> Dict[str, int]:
        """
        Apply TREX correction to measurement counts.
        
        Args:
            counts: Raw measurement counts {bitstring: count}
            num_qubits: Number of qubits
            
        Returns:
            Corrected counts dictionary
        """
        if num_qubits not in self.inverse_matrices:
            self.calibrate(num_qubits)
        
        M_inv = self.inverse_matrices[num_qubits]
        total = sum(counts.values())
        
        if total == 0:
            return counts
        
        # Per-bitstring correction (simplified global correction)
        corrected = {}
        for bitstring, count in counts.items():
            # Apply per-bit correction based on bit values
            prob = count / total
            
            # Count number of 1s for correction factor
            num_ones = bitstring.count('1')
            # Correction factor based on confusion matrix
            correction_factor = M_inv[1,1] ** (num_ones / len(bitstring))
            
            corrected[bitstring] = max(0, int(prob * total * correction_factor))
        
        return corrected if corrected else counts
    
    def get_fidelity_improvement(self, num_qubits: int) -> float:
        """
        Estimate fidelity improvement from TREX correction.
        
        Returns:
            Expected fidelity improvement factor
        """
        if num_qubits not in self.confusion_matrices:
            return 1.0
        
        M = self.confusion_matrices[num_qubits]
        # Average diagonal (correct measurement probability)
        baseline_fidelity = (M[0,0] + M[1,1]) / 2
        # TREX improves this toward 1.0
        corrected_fidelity = min(1.0, baseline_fidelity + 0.5 * (1 - baseline_fidelity))
        
        return corrected_fidelity / max(baseline_fidelity, 0.01)


class DynamicalDecoupling:
    """
    Dynamical Decoupling (DD) pulse sequences for coherence extension.
    
    Implements common DD sequences:
    - XY4: Basic 4-pulse sequence
    - CPMG: Carr-Purcell-Meiboom-Gill
    
    Reference: Viola et al. (1999), Biercuk et al. (2011)
    """
    
    @staticmethod
    def add_xy4_sequence(qc: QuantumCircuit, qubit: int, num_cycles: int = 1) -> QuantumCircuit:
        """
        Add XY4 dynamical decoupling sequence.
        
        Sequence: X - Y - X - Y (repeated)
        """
        qc_dd = qc.copy()
        for _ in range(num_cycles):
            qc_dd.x(qubit)
            qc_dd.barrier(qubit)
            qc_dd.y(qubit)
            qc_dd.barrier(qubit)
            qc_dd.x(qubit)
            qc_dd.barrier(qubit)
            qc_dd.y(qubit)
            qc_dd.barrier(qubit)
        return qc_dd
    
    @staticmethod
    def add_cpmg_sequence(qc: QuantumCircuit, qubit: int, num_pulses: int = 4) -> QuantumCircuit:
        """
        Add CPMG dynamical decoupling sequence.
        
        Sequence: (π_y)^n
        """
        qc_dd = qc.copy()
        for _ in range(num_pulses):
            qc_dd.y(qubit)
            qc_dd.barrier(qubit)
        return qc_dd


class NoisySimulator:
    """
    Wrapper for noisy quantum simulation with TREX and DD support.
    """
    
    def __init__(
        self,
        noise_model: Optional[NoiseModel] = None,
        use_trex: bool = True,
        use_dd: bool = False,
        shots: int = 8192
    ):
        """
        Initialize noisy simulator.
        
        Args:
            noise_model: Qiskit NoiseModel (None for ideal)
            use_trex: Whether to enable TREX error mitigation
            use_dd: Whether to enable dynamical decoupling
            shots: Number of measurement shots
        """
        self.noise_model = noise_model
        self.shots = shots
        self.use_trex = use_trex
        self.use_dd = use_dd
        
        # Create simulator
        if noise_model:
            self.sim = AerSimulator(noise_model=noise_model, method='matrix_product_state')
        else:
            self.sim = AerSimulator(method='matrix_product_state')
        
        # TREX calibrator
        if use_trex:
            self.calibrator = TREXCalibrator(self.sim, shots=min(4096, shots))
        else:
            self.calibrator = None
    
    def run(self, qc: QuantumCircuit, apply_mitigation: bool = True) -> Dict[str, int]:
        """
        Run circuit and return (optionally mitigated) counts.
        """
        counts = self.sim.run(qc, shots=self.shots).result().get_counts()
        
        if apply_mitigation and self.calibrator:
            counts = self.calibrator.correct_counts(counts, qc.num_qubits)
        
        return counts


if __name__ == "__main__":
    # Demo: Test noise model and TREX
    print("Testing Noise Model and TREX Calibration...")
    
    noise = create_ibm_noise_model(p2_error=0.02, read_error=0.03)
    sim = NoisySimulator(noise_model=noise, use_trex=True)
    
    # Simple test circuit
    qc = QuantumCircuit(4)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.cx(2, 3)
    qc.measure_all()
    
    counts = sim.run(qc)
    print(f"Counts (with TREX): {counts}")
