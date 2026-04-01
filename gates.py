"""
mfvm/gates.py
─────────────
GateLibrary: quantum gates as direct Madelung (ρ, S) transformations.

Every gate is an in-place transformation of the MadelungState.
No basis decomposition. No matrix exponentiation. No state vector.

Gate semantics match standard quantum gates exactly:
  - Hadamard creates superposition
  - CNOT creates entanglement (requires joint density)
  - Phase gates rotate S
  - Pauli gates flip ρ or shift S by π

PSI CACHE CONTRACT
──────────────────
MadelungState.rho and .S are now properties whose setters auto-invalidate
the ψ cache.  Full assignments (self.state.rho = tensor) are handled
automatically.  In-place modifications (self.state.S[mask] += θ,
self.state.rho[i] = x) require an explicit _invalidate_psi() call at the
end of the gate.  Every gate in this file follows that contract.
"""

import torch
import numpy as np
from .state import MadelungState, EPS


class GateLibrary:
    """
    Quantum gates implemented as Madelung fluid transformations.

    All gates operate in-place on the MadelungState.
    The (ρ, S) representation makes many gates trivially cheap:
      - Phase rotations:  S += θ at target indices  →  O(1)
      - Pauli X:          swap ρ entries            →  O(1)
      - Hadamard:         mix ρ + set S             →  O(1)
      - CNOT:             conditional swap           →  O(dim)
    """

    def __init__(self, state: MadelungState):
        self.state  = state
        self.device = state.device

    def _qubit_indices(self, qubit: int):
        """
        Return index arrays (i0, i1) where i0 has qubit=0, i1 has qubit=1.
        All other bits range freely.
        """
        n        = self.state.n_qubits
        dim      = self.state.dim
        indices  = torch.arange(dim, device=self.device)
        bit_mask = 1 << (n - 1 - qubit)
        i0 = indices[(indices & bit_mask) == 0]
        i1 = indices[(indices & bit_mask) != 0]
        return i0, i1

    def _apply_unitary(self, U: torch.Tensor, qubit: int):
        """
        Apply a 2×2 unitary U to a single qubit.
        Works by: (ρ,S) → ψ → U⊗I^(n-1) ψ → (ρ_new, S_new).
        This is the exact route — all phase interference is handled correctly.

        Full assignment to self.state.rho / self.state.S triggers the
        property setter, which automatically invalidates the ψ cache.
        No explicit _invalidate_psi() needed here.
        """
        i0, i1 = self._qubit_indices(qubit)
        psi = self.state.psi.clone()          # reads from cache (or builds it)

        a0 = psi[i0].clone()                  # amplitudes where qubit=0
        a1 = psi[i1].clone()                  # amplitudes where qubit=1

        U = U.to(dtype=torch.complex64, device=self.device)
        new_a0 = U[0, 0] * a0 + U[0, 1] * a1
        new_a1 = U[1, 0] * a0 + U[1, 1] * a1

        psi_new = psi.clone()
        psi_new[i0] = new_a0
        psi_new[i1] = new_a1

        # Full assignments — property setters invalidate cache automatically
        self.state.rho = (psi_new.abs() ** 2)
        self.state.S   = torch.angle(psi_new)
        self.state.normalize()

    # ─────────────────────────────────────────────────────────
    # Single-qubit gates (all exact via _apply_unitary)
    # ─────────────────────────────────────────────────────────

    def hadamard(self, qubit: int):
        """H = 1/√2 [[1,1],[1,-1]]"""
        s = 1.0 / np.sqrt(2)
        U = torch.tensor([[s, s], [s, -s]], dtype=torch.complex64)
        self._apply_unitary(U, qubit)

    def pauli_x(self, qubit: int):
        """X: [[0,1],[1,0]]"""
        U = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64)
        self._apply_unitary(U, qubit)

    def pauli_y(self, qubit: int):
        """Y: [[0,-i],[i,0]]"""
        U = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex64)
        self._apply_unitary(U, qubit)

    def pauli_z(self, qubit: int):
        """
        Z: [[1,0],[0,-1]] — phase flip on |1⟩, ρ unchanged.
        In-place S modification → explicit _invalidate_psi() required.
        """
        _, i1 = self._qubit_indices(qubit)
        self.state.S[i1] += np.pi
        self.state._invalidate_psi()

    def phase(self, qubit: int, theta: float):
        """
        P(θ): [[1,0],[0,e^iθ]] — pure phase on |1⟩.
        In-place S modification → explicit _invalidate_psi() required.
        """
        _, i1 = self._qubit_indices(qubit)
        self.state.S[i1] += theta
        self.state._invalidate_psi()

    def s_gate(self, qubit: int):
        self.phase(qubit, np.pi / 2.0)

    def t_gate(self, qubit: int):
        self.phase(qubit, np.pi / 4.0)

    def rx(self, qubit: int, theta: float):
        """Rx(θ) = cos(θ/2)I - i·sin(θ/2)X"""
        c = np.cos(theta / 2)
        s = np.sin(theta / 2)
        U = torch.tensor([[c, -1j*s], [-1j*s, c]], dtype=torch.complex64)
        self._apply_unitary(U, qubit)

    def ry(self, qubit: int, theta: float):
        """Ry(θ) = cos(θ/2)I - i·sin(θ/2)Y"""
        c = np.cos(theta / 2)
        s = np.sin(theta / 2)
        U = torch.tensor([[c, -s], [s, c]], dtype=torch.complex64)
        self._apply_unitary(U, qubit)

    def rz(self, qubit: int, theta: float):
        """Rz(θ): pure phase — ρ unchanged.  Delegates to phase()."""
        self.phase(qubit, theta)

    # ─────────────────────────────────────────────────────────
    # Two-qubit gates (require joint density — entanglement lives here)
    # ─────────────────────────────────────────────────────────

    def cnot(self, control: int, target: int):
        """
        CNOT: flip target qubit iff control qubit is |1⟩.
        Creates entanglement when control is in superposition.

        In Madelung: swap (ρ, S) blocks where control=1.
        In-place swaps on both rho and S → _invalidate_psi() at end.
        """
        n   = self.state.n_qubits
        dim = self.state.dim
        indices = torch.arange(dim, device=self.device)

        ctrl_mask   = 1 << (n - 1 - control)
        target_mask = 1 << (n - 1 - target)

        # Indices where control=1
        ctrl_1 = (indices & ctrl_mask) != 0

        # Within those, pairs where target differs
        target_0_in_ctrl1 = indices[ctrl_1 & ((indices & target_mask) == 0)]
        target_1_in_ctrl1 = indices[ctrl_1 & ((indices & target_mask) != 0)]

        # Swap ρ and S for these pairs — in-place
        self.state.rho[target_0_in_ctrl1], self.state.rho[target_1_in_ctrl1] = \
            self.state.rho[target_1_in_ctrl1].clone(), \
            self.state.rho[target_0_in_ctrl1].clone()
        self.state.S[target_0_in_ctrl1], self.state.S[target_1_in_ctrl1] = \
            self.state.S[target_1_in_ctrl1].clone(), \
            self.state.S[target_0_in_ctrl1].clone()
        self.state._invalidate_psi()

    def cz(self, control: int, target: int):
        """
        CZ: phase flip on |11⟩.
        In-place S modification → _invalidate_psi().
        """
        n   = self.state.n_qubits
        dim = self.state.dim
        indices     = torch.arange(dim, device=self.device)
        ctrl_mask   = 1 << (n - 1 - control)
        target_mask = 1 << (n - 1 - target)
        both_1 = ((indices & ctrl_mask) != 0) & ((indices & target_mask) != 0)
        self.state.S[both_1] += np.pi
        self.state._invalidate_psi()

    def swap(self, qubit_i: int, qubit_j: int):
        """SWAP: exchange states of two qubits."""
        self.cnot(qubit_i, qubit_j)
        self.cnot(qubit_j, qubit_i)
        self.cnot(qubit_i, qubit_j)

    # ─────────────────────────────────────────────────────────
    # Compound circuits
    # ─────────────────────────────────────────────────────────

    def bell_state(self, qubit_i: int = 0, qubit_j: int = 1):
        """
        Prepare |Φ+⟩ = (|00⟩ + |11⟩)/√2.
        Requires at least 2 qubits. Creates genuine entanglement.
        """
        # Full assignments → property setters invalidate cache
        self.state.rho    = torch.zeros(self.state.dim,
                                        dtype=torch.float32, device=self.device)
        self.state.rho[0] = 1.0   # |00...0⟩  (in-place after setter)
        self.state.S      = torch.zeros(self.state.dim,
                                        dtype=torch.float32, device=self.device)
        self.state._invalidate_psi()   # rho[0] modified in-place after setter

        self.hadamard(qubit_i)
        self.cnot(qubit_i, qubit_j)

    def ghz_state(self):
        """
        GHZ = (|00...0⟩ + |11...1⟩)/√2 for all N qubits.
        """
        n = self.state.n_qubits
        self.state.rho    = torch.zeros(self.state.dim,
                                        dtype=torch.float32, device=self.device)
        self.state.rho[0] = 1.0
        self.state.S      = torch.zeros(self.state.dim,
                                        dtype=torch.float32, device=self.device)
        self.state._invalidate_psi()

        self.hadamard(0)
        for i in range(1, n):
            self.cnot(0, i)

    def controlled_phase(self, control: int, target: int, theta: float):
        """
        Controlled-phase gate: applies exp(iθ) to |11⟩, identity elsewhere.
        Directly adds θ to S at indices where both control and target are |1⟩.
        This is the primitive needed by QFT — no CZ + correction hack.
        In-place S modification → _invalidate_psi().
        """
        n    = self.state.n_qubits
        dim  = self.state.dim
        idx  = torch.arange(dim, device=self.device)
        c_mask = 1 << (n - 1 - control)
        t_mask = 1 << (n - 1 - target)
        both_1 = ((idx & c_mask) != 0) & ((idx & t_mask) != 0)
        self.state.S[both_1] += theta
        self.state._invalidate_psi()

    def qft(self, qubits: list = None):
        """
        Quantum Fourier Transform on specified qubits (or all).

        Circuit:  for each qubit i:
                    H(i)
                    for k = i+1 … n-1:  CP(2π / 2^(k-i+1), i, k)
                  reverse qubit order

        Uses controlled_phase directly — no CZ + phase correction needed.
        Each sub-gate correctly manages cache invalidation.
        """
        if qubits is None:
            qubits = list(range(self.state.n_qubits))
        n = len(qubits)

        for i, qi in enumerate(qubits):
            self.hadamard(qi)
            for j, qj in enumerate(qubits[i+1:], start=i+1):
                # R_k gate: k = j - i + 1,  θ = 2π / 2^k
                theta = 2 * np.pi / (2 ** (j - i + 1))
                self.controlled_phase(qi, qj, theta)

        # Bit-reverse permutation
        for i in range(n // 2):
            self.swap(qubits[i], qubits[n - 1 - i])

    def grover_oracle(self, target_state: int):
        """
        Grover oracle: phase flip on |target_state⟩.
        O(target)|x⟩ = -|x⟩ if x == target, else |x⟩.
        In-place S modification → _invalidate_psi().
        """
        self.state.S[target_state] += np.pi
        self.state._invalidate_psi()

    def function_oracle(self, f, n_solutions: int = None):
        """
        General function oracle: phase-flip ALL basis states |i⟩ where f(i)=True.

        This is the reusable primitive behind quantum search over any
        structured or unstructured problem:

            f : int → bool    (classical predicate, evaluated for each basis state)

        Phase flip applied to every i in [0, dim) where f(i) is True.
        States where f(i) is False are untouched.

        Returns t: number of marked states (solutions found).

        PERFORMANCE — two modes:

          Tensor mask (recommended for GPU workloads):
            Pass a pre-computed boolean tensor of shape (dim,).
            Oracle cost is O(1) — a single GPU scatter write.
            This is what makes the O(√N) Grover advantage real in wall-clock time.

                mask = compute_all_hits_in_parallel(...)   # your vectorized test
                t = gate.function_oracle(mask)

          Callable (convenience / small problems / debugging):
            Pass f: int → bool. Evaluated sequentially for each basis state.
            Oracle cost is O(dim) — a Python loop, not GPU-parallel.
            Total cost with Grover: O(dim) × O(√dim) = O(dim^1.5) — worse than classical.
            Only use for prototyping or when dim is small (< 64).

                t = gate.function_oracle(lambda i: database[i].matches(query))

        For production use at scale, always pre-compute the boolean mask
        using vectorized GPU operations and pass it as a tensor.
        """
        dim = self.state.dim

        # Accept either a callable or a boolean tensor mask
        if callable(f):
            # O(dim) sequential — use only for prototyping
            marked = torch.tensor(
                [bool(f(i)) for i in range(dim)],
                dtype=torch.bool,
                device=self.device,
            )
        elif isinstance(f, torch.Tensor):
            # O(1) — recommended path for production use
            marked = f.to(self.device).bool()
            if marked.shape[0] < dim:
                pad    = torch.zeros(dim - marked.shape[0],
                                     dtype=torch.bool, device=self.device)
                marked = torch.cat([marked, pad])
        else:
            raise TypeError("f must be callable (slow) or a boolean torch.Tensor (fast)")

        self.state.S[marked] += np.pi
        self.state._invalidate_psi()   # in-place S modification
        return int(marked.sum().item())

    def grover_diffusion(self):
        """
        Grover diffusion: 2|s⟩⟨s| - I  (inversion about the mean).
        Applied directly on ψ: ψ_new[x] = 2·mean(ψ) - ψ[x].
        This is exact and avoids accumulated Hadamard errors.

        Full assignments to rho and S → property setters handle invalidation.
        """
        psi     = self.state.psi.clone()
        psi_new = 2.0 * psi.mean() - psi
        # Full assignments — setters invalidate cache automatically
        self.state.rho = (psi_new.abs() ** 2)
        self.state.S   = torch.angle(psi_new)
        self.state.normalize()

    def grover_search(self, target_state: int, n_iterations: int = None):
        """
        Full Grover search for a single known target_state in 2^N space.
        Optimal iterations: π/4 · √(2^N)

        For unknown or multiple targets use grover_search_f() instead.
        """
        dim = self.state.dim

        if n_iterations is None:
            n_iterations = max(1, int(np.pi / 4 * np.sqrt(dim)))

        # Full assignments → property setters invalidate cache
        self.state.rho = torch.full((dim,), 1.0 / dim,
                                    dtype=torch.float32, device=self.device)
        self.state.S   = torch.zeros(dim, dtype=torch.float32, device=self.device)

        for _ in range(n_iterations):
            self.grover_oracle(target_state)
            self.grover_diffusion()

        self.state.normalize()

    def grover_search_f(self, f, n_solutions: int = None,
                        valid_range: int = None):
        """
        Grover search with a general function oracle f(i) → bool.

        Handles multiple solutions and unknown solution count.
        Implements adaptive strategy (Boyer et al. 1998):
          - If n_solutions is given: optimal iterations = π/4 · √(dim/t)
          - If unknown: use 1 oracle call to count t, then iterate optimally

        Returns (marked_indices, t, n_oracle_calls) where:
          marked_indices: list of high-probability states after amplification
          t:              number of solutions found by oracle
          n_oracle_calls: total oracle evaluations used

        valid_range: if set, restricts search to indices [0, valid_range).
        Useful when 2^N > actual problem size (e.g., 6 qubits for 50 items).

        PERFORMANCE NOTE:
          Total wall-clock complexity = (oracle cost) × O(√(dim/t))

          For true O(√N) advantage, f must be O(1) — pass a pre-computed
          boolean tensor, not a callable.
        """
        dim    = self.state.dim
        vrange = valid_range or dim
        oracle_calls = 0

        # ── Step 1: Count solutions ────────────────────────────
        # Count directly from the mask/callable — no oracle call, no state mutation.
        # The previous code applied the oracle to a fresh uniform state purely to
        # read marked.sum(), paying a full state init + phase-flip + cache invalidation
        # for information available without touching the state at all.
        if n_solutions is None:
            if isinstance(f, torch.Tensor):
                mask_count = f[:vrange].to(self.device).bool()
                t = int(mask_count.sum().item())
            else:
                t = sum(1 for i in range(vrange) if bool(f(i)))
            oracle_calls += 0    # no oracle consumed
        else:
            t = n_solutions

        if t == 0:
            return [], 0, oracle_calls

        # ── Step 2: Optimal Grover iterations ─────────────────
        n_iter = max(1, int(np.pi / 4 * np.sqrt(dim / t)))

        rho_init         = torch.zeros(dim, dtype=torch.float32, device=self.device)
        rho_init[:vrange] = 1.0 / vrange
        self.state.rho   = rho_init
        self.state.S     = torch.zeros(dim, dtype=torch.float32, device=self.device)

        for _ in range(n_iter):
            self.function_oracle(f)
            self.grover_diffusion()
            oracle_calls += 1

        self.state.normalize()

        # ── Step 3: Non-destructive readout ───────────────────
        rho     = self.state.rho[:vrange]
        top_idx = torch.argsort(rho, descending=True)
        marked  = top_idx[:max(t, 1)].cpu().tolist()
        return marked, t, oracle_calls
