# HEPAR Formal Theory Documentation
# ==================================
# Mathematical foundations for Springer publication

## Definition 1: HEPAR Quantum Image State

The HEPAR state for an image I is defined as:

$$
|\Psi_{\text{HEPAR}}\rangle = \frac{1}{\sqrt{|L|}} \sum_{i \in L} w_i \left( \cos\frac{\theta_i}{2}|0\rangle + e^{i\phi_i}\sin\frac{\theta_i}{2}|1\rangle \right) \otimes |p_i\rangle
$$

Where:
- **L** = Quadtree leaf set (adaptive spatial decomposition)
- **p_i** = Variable-length spatial path encoding position
- **w_i** = Area-normalized weight: $w_i = \sqrt{s_i / N^2}$ where $s_i$ is block size
- **θ_i** = Intensity angle: $\theta_i = 2\arcsin(\sqrt{I_i})$
- **φ_i** = Phase encoding gradient: $\phi_i = \pi \cdot \nabla I_i$

---

## The 4 Pillars of HEPAR

### Pillar 1: Depth Register
Encodes quadtree level in dedicated qubits, enabling variable-resolution blocks.

$$n_{\text{depth}} = \lceil \log_2(d_{\max} + 1) \rceil$$

### Pillar 2: Gray Code Sorting
Minimizes Hamming distance between consecutive leaf indices.

$$\text{Savings} = \sum_{i} H(i, i+1)_{\text{unsorted}} - H(i, i+1)_{\text{Gray}}$$

### Pillar 3: SPAE (Simultaneous Phase-Amplitude Encoding)
Dual-basis measurement extracts both intensity (Z-basis) and edges (X-basis).

### Pillar 4: TREX (Twirled Readout Error Extinction)
Calibration-based readout error mitigation via confusion matrix inversion.

---

## Complexity Bounds

### Theorem 1 (Qubit Complexity)
$$Q_{\text{HEPAR}} = \lceil \log_2 |L| \rceil + \lceil \log_2(d_{\max}+1) \rceil + 1$$

### Theorem 2 (Gate Complexity)
$$G_{\text{HEPAR}} = O(|L| \cdot (2n + \log |L|))$$

### Theorem 3 (Depth Scaling)
$$D_{\text{HEPAR}} = O(|L|)$$

---

## Worst-Case Analysis (Proposition)

> **Proposition (Worst-Case Degradation).**
> For images with entropy $H(I) \to \log N^2$, the number of quadtree leaves 
> $|L| \to N^2$, and HEPAR degrades to FRQI/NEQR-level complexity.

This occurs for:
- Gaussian noise images
- High-frequency textures
- Images with uniform high variance

**Implication:** HEPAR provides advantage only for structured/sparse images.

---

## Non-Claims (Reviewer Armor)

> **Remark (Non-Claims).**
> HEPAR does not claim:
> - Exponential quantum speedup
> - Full image reconstruction efficiency for all image classes
> - Universal quantum advantage
> 
> Its goal is *feasibility under NISQ constraints* for structured images with
> exploitable spatial redundancy.

---

## State Fidelity Definition

$$F = |\langle \psi_{\text{ideal}} | \psi_{\text{noisy}} \rangle|^2$$

For mixed states:
$$F(\rho, \sigma) = \left( \text{Tr}\sqrt{\sqrt{\rho}\sigma\sqrt{\rho}} \right)^2$$
