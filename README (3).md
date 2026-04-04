# MFVM — Madelung Flow Virtual Machine

**Divergence Lab / Kai Cobbs**

```
pip install mfvm  # coming soon — clone from source for now
```

---

## What This Is

MFVM is a field-computing framework. The fundamental data structure is an **amplitude field**: a pair (ρ, S) where ρ is a probability density and S is a phase field, together defining a complex-valued field ψ = √ρ · exp(iS). These fields evolve under exact, GPU-native operations.

The framework originated from work on quantum simulation. The Madelung equations — a hydrodynamic reformulation of the Schrödinger equation due to Madelung (1926) — show that any quantum wavefunction can be written as a probability fluid with a phase field driving its velocity. This is the mathematical backbone of the library. Quantum gates, Hamiltonian evolution, entanglement, interference, and ground-state projection are all natively expressible in (ρ, S) arithmetic.

What emerged from building this is that the quantum framing, while intellectually generative, is not the right lens for most applications. The amplitude field is a general-purpose computational primitive. It supports:

- **Distributional memory** — encode a matrix as amplitude fields, query it many times with no recomputation (QKMM)
- **Parallel scoring** — R independent fields evolving simultaneously; one oracle pass extracts an (R, N) probability matrix
- **Global belief propagation** — soft oracles encode continuous scores into phases; diffusion produces non-separable, globally coherent output distributions
- **Ground-state optimization** — imaginary-time projection finds the minimum-energy configuration of any potential landscape via split-operator evolution
- **Non-destructive readout** — the state is never collapsed; ρ can be read arbitrarily many times without disturbing S

The quantum simulation capabilities remain fully intact and are validated (Bell states, GHZ, QFT, Grover). But the design intention going forward is **field computing**: amplitude fields as a computational medium, not as a quantum hardware emulator.

---

## Installation

```bash
git clone https://github.com/HavensGuide/DissipativeValue
cd DissipativeValue
pip install -e .
```

**Requirements:** Python ≥ 3.9, PyTorch ≥ 2.0. A CUDA GPU is strongly recommended; all operations fall back to CPU automatically.

---

## Core Concepts

### The Amplitude Field

Every computation in MFVM is expressed over a pair of tensors:

| Symbol | Shape | Meaning |
|--------|-------|---------|
| ρ | `(dim,)` float32 | Probability density — non-negative, sums to 1 |
| S | `(dim,)` float32 | Phase field — angles in ℝ |
| ψ | `(dim,)` complex64 | Amplitude — `√ρ · exp(iS)`, computed on demand |

`dim` is the size of the configuration space. For quantum-register semantics, `dim = 2^n_qubits`. For general field computation, `dim` is the size of your problem space (number of candidates, grid points, etc.).

### Two Modes of Operation

**Discrete (qubit-register) mode** — `mode='qubit_register'`

Indices are computational basis states `{0,1}^N`. Gates, measurement, entanglement entropy, and CHSH tests operate here. This is the standard quantum computing interface.

**Continuous (position-space) mode** — `mode='position_space'`

Indices are spatial positions on a 1D periodic grid. Madelung PDEs, the quantum potential, osmotic velocity, and imaginary-time ground-state projection operate here. This is the field-computing interface: define a potential landscape V(x), evolve the amplitude field, read out the ground-state distribution.

Setting `mode=None` (the default) disables mode checks for backward compatibility.

---

## Quick Start

### Field-Computing Interface

```python
from mfvm import MadelungState, HamiltonianEvolver
import torch

# A 128-point position-space amplitude field
state = MadelungState(n_qubits=7, mode='position_space', hbar_eff=1.0)

evolver = HamiltonianEvolver(state, dt=0.01, derivative_mode='spectral')

# Real-time evolution under a double-well potential
history = evolver.evolve(
    n_steps=1000,
    V=HamiltonianEvolver.double_well(depth=1.0, width=1.0),
    record_every=50,
)

# Ground-state projection via imaginary-time (split-operator method)
# Updates both ρ and S — converges to the ground state wavefunction
E0 = evolver.ground_state_energy(
    V=HamiltonianEvolver.harmonic(omega=1.0),
    n_steps=500,
    tol=1e-6,
)
print(f"Ground state energy: {E0:.4f}")  # → ~0.5 for harmonic oscillator

# Non-destructive readout — never modifies the state
print(state.rho)   # probability density
print(state.S)     # phase field
```

### Parallel Field Evaluation

```python
from mfvm import BatchedAmplitudeState
import torch

# 1024 independent registers over 200 candidates each
state = BatchedAmplitudeState(n_registers=1024, dim=256)
state.init_uniform(200)

# Boolean oracle: shape (1024, 200) or (1024, 256) bool
mask = compute_oracle_mask(...)   # your vectorized test
state.oracle(mask)
state.diffusion()

# Full (1024, 200) probability matrix — one pass, all registers
probs = state.field(valid_range=200)

# Soft oracle: continuous scores → phase encoding → global interference
scores = compute_scores(...)      # (1024, 200) float
state.oracle_soft(scores)
state.diffusion()
```

### Quantum Circuit Interface

```python
from mfvm import MFVirtualMachine

vm = MFVirtualMachine(n_qubits=3)

# Gate-based circuit
vm.gate.hadamard(0)
vm.gate.cnot(0, 1)
vm.gate.t_gate(2)

# Non-destructive measurement — no shot noise
probs = vm.query.measure_probs()
corr  = vm.query.correlation(0, 1)
chsh  = vm.query.chsh_value(0, 1)   # should exceed 2.0 for entangled states

# Grover search
result = vm.run_search(target=42, n_qubits=8)

# State persists automatically to .mfvm file
vm.save()
```

### Amplitude Search (GateLibrary)

```python
from mfvm import MFVirtualMachine
import torch

vm = MFVirtualMachine(n_qubits=8)   # 256-element search space

# Pre-computed boolean mask — O(1) GPU scatter oracle
mask = torch.zeros(256, dtype=torch.bool)
mask[[42, 117, 200]] = True          # mark targets

result = vm.gate.amplitude_search(mask, n_iter=1, top_k=3)
# result['rho']          — full (256,) probability field, non-destructive
# result['top_indices']  — [42, 117, 200] ranked by amplitude
# result['top_probs']    — corresponding probabilities
# result['oracle_calls'] — 1

# Unlike Grover, the full ρ distribution is always available after every step —
# no measurement collapse, no shot noise, no counting step required.
# Caller verifies or re-ranks top candidates directly from the amplitude field.
#
# Note: this advantage is specific to MFVM as a classical simulator.
# On physical quantum hardware, measurement collapses the wavefunction
# and the full ρ distribution is not accessible.
```

### Amplitude-Encoded Matrix Memory (QKMM)

```python
from mfvm import QKMMMemory
import torch

A = torch.randn(256, 128)   # matrix to encode — write once
B = torch.randn(64,  128)   # query batch

mem = QKMMMemory(mode='exact')
mem.encode(A)                # O(M·N) write — amortized across all queries

result = mem.query(B)        # O(M·K·G) per query, A never re-encoded
result = mem.query(B2)       # same encoding, different query
result = mem.query(B3)

# Batched: n independent query batches in one bmm kernel
B_batch = torch.randn(8, 64, 128)
results = mem.query_batched(B_batch)   # (8, 256, 64)

# Top-k via amplitude search — beats full query() when k << M
# Phase 1: coarse oracle over M rows via amplitude field  — O(M·G)
# Phase 2: exact re-score of top candidates only          — O(k·N) vs O(M·N)
result = mem.query_topk(B, k=8)
# result['indices']  — (K, k) top row indices into A per query
# result['scores']   — (K, k) exact inner product scores
# result['rho']      — (K, M) full amplitude field (non-destructive readout)

# Benchmark: amortized cost vs classical matmul
report = mem.benchmark(M=256, N=128, K=256)
print(f"Break-even at Q={report['breakeven_Q']:.1f} queries")
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  MFVirtualMachine  (top-level orchestrator)             │
│  run_circuit │ run_hamiltonian │ run_search              │
├──────────────┬──────────────────┬───────────────────────┤
│  GateLibrary │ HamiltonianEvol. │  QueryEngine          │
│  H, CNOT, T  │ step / evolve    │  expectation          │
│  QFT, Grover │ imag_time_evolve │  chsh_value           │
│  Bell, GHZ   │ ground_state_E   │  measure_probs        │
│  amplitude_  │                  │                       │
│  search      │                  │                       │
├──────────────┴──────────────────┴───────────────────────┤
│  MadelungState              BatchedAmplitudeState       │
│  (ρ, S) single register     (ρ, S) × R registers        │
│  mode A / mode B            field / oracle / diffusion  │
├─────────────────────────────────────────────────────────┤
│  QKMMMemory                                             │
│  Amplitude-encoded matrix — write once, query many      │
│  query / query_batched / query_topk                     │
├─────────────────────────────────────────────────────────┤
│  GPU LAYER — PyTorch / CUDA                             │
│  FFT · bmm · svd · scatter — all native tensor ops      │
└─────────────────────────────────────────────────────────┘
```

---

## Key Design Decisions

**Non-destructive readout.** Reading ρ or S never modifies the state. This is categorically different from physical quantum hardware, where measurement collapses the wavefunction. In MFVM, `peek()`, `measure_probs()`, and `expectation()` are free operations that can be called any number of times at any point in a computation.

**ψ dirty-bit cache.** Computing ψ = √ρ · exp(iS) is relatively expensive. `MadelungState` caches the result and invalidates it only when ρ or S actually changes. Full assignments (`state.rho = tensor`) invalidate automatically via property setters. In-place modifications require an explicit `state._invalidate_psi()` call — gates in `GateLibrary` follow this contract throughout.

**Split-operator imaginary time.** The old ground-state energy estimator re-weighted ρ by exp(−E_loc·dτ) but left S frozen — it couldn't drive the phase toward the real ground-state wavefunction. The current `imaginary_time_step()` uses the Strang split-operator method: half-potential step in position space, full kinetic step in Fourier space (exp(−k²·dτ/2m)), half-potential step. Both ρ and S are updated. This is the standard algorithm used in computational quantum chemistry and is second-order accurate in dτ.

**Mode enforcement.** `MadelungState` accepts `mode='position_space'` or `mode='qubit_register'`. Calling `quantum_potential` or `osmotic_velocity` on a qubit-register state raises `RuntimeError` — in register mode, array indices are bit-strings with no spatial adjacency, so finite-difference stencils are physically meaningless. `HamiltonianEvolver` warns at construction if attached to a register-mode state. `imaginary_time_step` raises on register-mode states entirely. Setting `mode=None` (default) disables all checks for backward compatibility.

**Entanglement visibility.** `remove_qubit()` uses SVD to find the dominant Schmidt component of the reduced state. When the pre-trace state was entangled, the post-trace state is genuinely mixed — but `(ρ, S)` can only represent a pure state. A `warnings.warn` fires whenever the dominant Schmidt weight falls below 0.99, reporting the discarded fraction and pointing to `reduced_density_matrix()` for the exact 2×2 result.

---

## Theoretical Background

The (ρ, S) representation is the **Madelung decomposition** of a complex amplitude field ψ = √ρ · e^{iS}. Under this substitution, the Schrödinger equation becomes two coupled fluid equations:

```
∂ρ/∂t = −∇·(ρ ∇S/m)                  [continuity]
∂S/∂t = −(∇S)²/2m − V − Q             [Hamilton-Jacobi]
```

where Q = −(ħ²/2m)(∇²√ρ / √ρ) is the **quantum potential** — the term that encodes all genuinely quantum behavior (tunneling, zero-point energy, interference).

This formulation was first written down by Madelung (1926) and later given a rigorous stochastic mechanics foundation by Nelson (1966), who showed that any conservative diffusion process admits a quantum mechanical representation. The information-geometric interpretation — where ρ is a point on a statistical manifold and S encodes the osmotic velocity — connects to the Fisher–Rao metric and score-based diffusion models.

MFVM implements both the discrete (gate) and continuous (PDE) faces of this structure on GPU hardware, and extends it to batched field evaluation and amplitude-encoded memory. The quantum simulation aspects were the original construction scaffolding. The more general claim is that **amplitude fields are a useful computational primitive independent of their quantum origin** — the interference structure, the non-destructive readout, and the global coherence of the diffusion operator are properties of the mathematical object, not of any physical system.

---

## Validated Circuits

The following are confirmed to produce correct results on CPU and CUDA:

| Circuit | Validation |
|---------|------------|
| Bell state | \|Φ+⟩ = (\|00⟩+\|11⟩)/√2, CHSH > 2√2 |
| GHZ state | (\|000⟩+\|111⟩)/√2 for N qubits |
| QFT | n-qubit quantum Fourier transform |
| Grover search | single and multi-target, with function oracle |
| Harmonic oscillator ground state | E0 → ħω/2 under imaginary-time projection |

---

## Citation

If you use MFVM in research, please cite the foundational paper, where it was utilized in an economic setting:

> Cobbs, K. (2025). *The Dissipative Structure of Value: Madelung Flow Regression Applied to NYC Residential Real Estate.* SSRN Working Paper. https://doi.org/10.2139/ssrn.5975594

---

## License

MIT. See `LICENSE`.

---

*Divergence Lab is an independent research venture. MFVM is developed and maintained by Kai Cobbs.*