"""
mfvm/query.py
─────────────
QueryEngine: GPU-saturating parallel queries against a single MadelungState.

The key departure from physical quantum hardware:
  - Physical QC: each measurement collapses the state → need N shots → N copies
  - MFVM: state is never disturbed → broadcast 1 state across N parallel queries

This eliminates shot noise entirely and keeps memory at O(state) + O(batch),
not O(N × state).
"""

import torch
from typing import Callable, Optional, Union
from .state import MadelungState, EPS


class QueryEngine:
    """
    Parallel query executor for MadelungState.

    Two query modes:
    ─────────────────────────────────────────────────────
    Mode A: parallel_state
        One state, N different operators queried simultaneously.
        Use case: expectation values of N observables in one pass.

    Mode B: parallel_operator
        One operator, N copies of state in parallel batch.
        Use case: GPU saturation when a single query is too small
                  to fill the GPU's execution units.
    ─────────────────────────────────────────────────────

    In both modes, the underlying MadelungState is never modified.
    """

    def __init__(self, state: MadelungState):
        self.state  = state
        self.device = state.device

    # ─────────────────────────────────────────────────────────
    # Core: expectation value (single operator, no parallelism)
    # ─────────────────────────────────────────────────────────

    def expectation(self, operator: torch.Tensor) -> float:
        """
        ⟨O⟩ = Σ_x ρ(x) · O(x)

        operator: (dim,) diagonal observable or (dim, dim) full matrix.
        Non-destructive: ρ is read, never written.
        """
        if operator.dim() == 1:
            return float(torch.dot(self.state.rho, operator.to(self.device)))
        elif operator.dim() == 2:
            psi = self.state.psi
            return float(torch.real(torch.conj(psi) @ operator.to(self.device) @ psi))
        else:
            raise ValueError(f"Operator must be 1D or 2D, got shape {operator.shape}")

    def correlation(self, qubit_i: int, qubit_j: int) -> float:
        """
        Two-qubit correlation ⟨ZᵢZⱼ⟩ computed directly from joint density.
        No measurement required — exact, no sampling noise.
        """
        rho_matrix = self.state.rho.view([2] * self.state.n_qubits)
        # Z eigenvalues: |0⟩ → +1, |1⟩ → -1
        z_i = torch.tensor([1.0, -1.0], device=self.device)
        z_j = torch.tensor([1.0, -1.0], device=self.device)

        # Build Z_i ⊗ Z_j operator on the full space
        result = 0.0
        for xi in range(2):
            for xj in range(2):
                # Index into the n_qubit tensor: fix qubits i,j, sum over rest
                idx = [slice(None)] * self.state.n_qubits
                idx[qubit_i] = xi
                idx[qubit_j] = xj
                prob = float(rho_matrix[tuple(idx)].sum())
                result += prob * z_i[xi] * z_j[xj]
        return result

    # ─────────────────────────────────────────────────────────
    # Mode A: N different operators, 1 state
    # ─────────────────────────────────────────────────────────

    def parallel_state(self, operators: torch.Tensor) -> torch.Tensor:
        """
        Query N diagonal operators against the same state simultaneously.

        operators: (N, dim) — N diagonal observables
        returns:   (N,)     — expectation values

        GPU kernel processes all N operators in one batched matmul.
        State is broadcast, never copied.
        """
        ops = operators.to(self.device)  # (N, dim)
        rho = self.state.rho.unsqueeze(0)  # (1, dim) — broadcast, no copy
        return torch.sum(ops * rho, dim=1)  # (N,)

    def parallel_state_full(self, operators: torch.Tensor) -> torch.Tensor:
        """
        Query N full matrix operators against the same state simultaneously.

        operators: (N, dim, dim)
        returns:   (N,) real expectation values
        """
        ops = operators.to(self.device)  # (N, dim, dim)
        psi = self.state.psi.unsqueeze(0)  # (1, dim)
        # ⟨ψ|O|ψ⟩ for each operator: batched
        Opsi = torch.bmm(ops, psi.unsqueeze(-1))  # (N, dim, 1)
        ev = torch.conj(psi.unsqueeze(-1)).transpose(-2,-1) @ Opsi  # (N, 1, 1)
        return ev.real.squeeze(-1).squeeze(-1)  # (N,)

    # ─────────────────────────────────────────────────────────
    # Mode B: 1 operator, batch of parallel executions
    # (GPU saturation for small states)
    # ─────────────────────────────────────────────────────────

    def parallel_operator(
        self,
        operator: torch.Tensor,
        n_parallel: int = 256,
    ) -> torch.Tensor:
        """
        Run the same operator query n_parallel times simultaneously.
        Uses broadcast — one state in memory, n_parallel GPU threads.

        This is the GPU saturation pattern: when a single query
        underutilizes the GPU, batching forces a fully parallel kernel.

        operator:    (dim,) diagonal or (dim,dim) full
        n_parallel:  batch width — tune for GPU utilization
        returns:     (n_parallel,) — all identical (confirms no state mutation)
        """
        if operator.dim() == 1:
            op = operator.to(self.device).unsqueeze(0).expand(n_parallel, -1)  # (B, dim)
            rho = self.state.rho.unsqueeze(0).expand(n_parallel, -1)           # (B, dim) broadcast
            return torch.sum(op * rho, dim=1)  # (B,)
        elif operator.dim() == 2:
            op  = operator.to(self.device).unsqueeze(0).expand(n_parallel, -1, -1)
            psi = self.state.psi.unsqueeze(0).expand(n_parallel, -1)
            Opsi = torch.bmm(op, psi.unsqueeze(-1))
            ev = torch.conj(psi.unsqueeze(-1)).transpose(-2,-1) @ Opsi
            return ev.real.squeeze(-1).squeeze(-1)

    # ─────────────────────────────────────────────────────────
    # Measurement (non-destructive — returns probabilities, no collapse)
    # ─────────────────────────────────────────────────────────

    def measure_probs(self, qubit_idx: Optional[int] = None) -> torch.Tensor:
        """
        Return measurement probabilities without collapsing the state.

        qubit_idx=None: full distribution over all 2^N basis states
        qubit_idx=k:    marginal probability of qubit k being |0⟩ or |1⟩
        """
        if qubit_idx is None:
            return self.state.rho.clone()
        else:
            rdm = self.state.reduced_density_matrix(qubit_idx)
            return torch.diag(rdm)

    def sample(self, n_shots: int = 1024, qubit_idx: Optional[int] = None) -> torch.Tensor:
        """
        Draw samples from ρ without disturbing the state.
        Unlike physical QC, this is purely classical sampling from
        the stored distribution — the state is never collapsed.

        Returns: (n_shots,) integer indices into the basis
        """
        probs = self.measure_probs(qubit_idx)
        return torch.multinomial(probs, n_shots, replacement=True)

    # ─────────────────────────────────────────────────────────
    # Utility: CHSH inequality test (Bell test)
    # ─────────────────────────────────────────────────────────

    def chsh_value(self, qubit_i: int = 0, qubit_j: int = 1) -> float:
        """
        CHSH = |⟨AB⟩ - ⟨AB'⟩ + ⟨A'B⟩ + ⟨A'B'⟩|
        Classical bound: 2.0
        Quantum bound:   2√2 ≈ 2.828

        Standard CHSH settings:
          A  = Z_i,           A' = X_i
          B  = (Z_j+X_j)/√2, B' = (Z_j-X_j)/√2

        Implementation: compute each of the four two-qubit correlations
        ⟨O_i ⊗ O_j⟩ directly from the full joint density matrix |ψ⟩⟨ψ|.

        Each correlation ⟨O_i ⊗ O_j⟩ = ⟨ψ|(O_i ⊗ O_j)|ψ⟩ is evaluated
        by contracting the 2^N wavefunction against the Kronecker product
        observable — no basis rotation, no sampling, no state collapse.
        """
        if self.state.n_qubits < 2:
            raise ValueError("CHSH requires at least 2 qubits.")

        n   = self.state.n_qubits
        psi = self.state.psi          # (2^N,) complex — non-destructive read

        # Build single-qubit Pauli matrices
        I = torch.eye(2,     dtype=torch.complex64, device=self.device)
        Z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64, device=self.device)
        X = torch.tensor([[0, 1], [1,  0]], dtype=torch.complex64, device=self.device)
        s = 1.0 / (2.0 ** 0.5)
        B_mat  = s * (Z + X)   # (Z+X)/√2
        Bp_mat = s * (Z - X)   # (Z-X)/√2

        def two_qubit_corr(O_i: torch.Tensor, O_j: torch.Tensor) -> float:
            """
            ⟨ψ|(O_i ⊗ O_j ⊗ I^(n-2))|ψ⟩
            Evaluated by reshaping ψ into a per-qubit tensor and contracting.
            """
            psi_grid = psi.view([2] * n)                    # (2,2,...,2)

            # Contract qubit_j with O_j: sum over qubit_j index
            # Result shape: same as psi_grid but qubit_j axis is contracted
            psi_j = torch.tensordot(O_j, psi_grid, dims=([1], [qubit_j]))
            # tensordot puts the new (output) axis first → move it back
            psi_j = torch.moveaxis(psi_j, 0, qubit_j)      # restore axis order

            # Contract qubit_i with O_i
            psi_ij = torch.tensordot(O_i, psi_j,   dims=([1], [qubit_i]))
            psi_ij = torch.moveaxis(psi_ij, 0, qubit_i)

            # ⟨ψ|O|ψ⟩ = psi† · (O|ψ⟩)
            return float(torch.real(torch.dot(psi.conj(), psi_ij.reshape(-1))))

        AB  = two_qubit_corr(Z,  B_mat)    # ⟨Z_i B_j⟩
        ABp = two_qubit_corr(Z,  Bp_mat)   # ⟨Z_i B'_j⟩
        ApB = two_qubit_corr(X,  B_mat)    # ⟨X_i B_j⟩
        ApBp= two_qubit_corr(X,  Bp_mat)   # ⟨X_i B'_j⟩

        return abs(AB - ABp + ApB + ApBp)
