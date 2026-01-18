# Complexity Bounds
# =================

## HEPAR vs Competitors: Formal Comparison

---

## Qubit Complexity

| Technique | Qubits | HEPAR Advantage |
|-----------|--------|-----------------|
| FRQI      | $2n + 1$ | When $|L| < 2^{2n}/2^{n_{\text{depth}}}$ |
| NEQR      | $2n + 8$ | When $|L| < 2^{2n+8}/2^{n_{\text{addr}}+1}$ |
| MCQI      | $2n + 24$ | Almost always (unless |L| > N²) |
| HEPAR     | $\lceil\log_2|L|\rceil + n_{\text{depth}} + 1$ | — |

---

## Crossover Analysis

### Theorem (HEPAR-NEQR Crossover)

HEPAR uses fewer qubits than NEQR when:

$$|L| < 2^{2n + 8 - n_{\text{depth}} - 1}$$

For typical images with 80% homogeneous regions:
- 32×32: Crossover at |L| ≈ 256 (HEPAR wins)
- 64×64: Crossover at |L| ≈ 1024 (HEPAR wins)
- 256×256: Crossover at |L| ≈ 16384 (HEPAR wins if sparse)

---

## Circuit Depth Bounds

$$D_{\text{HEPAR}} = O(|L|) \ll O(N^2) = D_{\text{NEQR}}$$

For sparse medical images (|L| ≈ 50):
- HEPAR depth: ~100
- NEQR depth: ~8192

**→ 80× reduction**

---

## NISQ Feasibility Constraint

IBM Eagle limit: 127 qubits

| Technique | Max Image Size (127 qubits) |
|-----------|----------------------------|
| NEQR      | 32×32 (barely)             |
| MCQI      | 16×16                      |
| HEPAR     | 256×256 (if |L| < 64)      |
