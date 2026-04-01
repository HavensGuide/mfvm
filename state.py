"""
mfvm/state.py  —  MadelungState: the persistent (ρ, S) object.

REPRESENTATION NOTE
───────────────────
This module stores quantum states as (ρ, S) pairs where ρ = |ψ|² and
ψ = √ρ · exp(iS).  Two distinct use-modes exist and must not be conflated:

  (A) Position-space wavefunction on a 1D ring of 2^N grid points.
      Here indices 0…2^N-1 are spatial positions with periodic topology.
      quantum_potential and osmotic_velocity are physically meaningful.
      torch.roll finite differences are correct.

  (B) Computational-basis qubit register on {0,1}^N.
      Indices are bit-strings; there is NO natural spatial adjacency.
      quantum_potential / osmotic_velocity have no well-defined meaning
      in this mode — they are exposed for position-space use only.

All multi-qubit operations (partial trace, entanglement entropy) treat
the state as a computational-basis register (mode B).

PSI CACHE
─────────
ψ = √ρ · exp(iS) is expensive to recompute at every gate call.  A
dirty-bit cache is maintained automatically:

  - self.rho = ...  (full assignment)  →  cache invalidated via property setter
  - self.S   = ...  (full assignment)  →  cache invalidated via property setter
  - self.rho[i] = ...  (in-place)     →  call self._invalidate_psi() explicitly
  - self.S[i]   = ...  (in-place)     →  call self._invalidate_psi() explicitly

Gates in gates.py call _invalidate_psi() after all in-place S/ρ writes.
Pass psi_cache=False to disable (useful for debugging or profiling).
"""

import warnings
import torch
import numpy as np
import time
from pathlib import Path

EPS          = 1e-10
MFVM_VERSION = "0.1.0"
DEFAULT_FILENAME = "mfvm_state.mfvm"

_VALID_MODES = frozenset({"position_space", "qubit_register", None})


class MadelungState:

    def __init__(self, n_qubits=1, hbar_eff=1.0, m_eff=1.0,
                 device=None, description="", psi_cache=True, mode=None):
        """
        n_qubits:   number of qubits in the register
        hbar_eff:   effective ħ (scale quantum pressure)
        m_eff:      effective mass
        device:     torch.device (defaults to CUDA if available)
        description: free-text label saved with the state
        psi_cache:  enable ψ dirty-bit cache (default True).
                    Set False to always recompute ψ — useful for debugging
                    or when profiling to isolate cache effects.
        mode:       'position_space' — continuous wavefunction on a 1D
                        periodic grid (mode A).  quantum_potential and
                        osmotic_velocity are physically meaningful.
                        torch.roll finite differences are correct.
                    'qubit_register' — computational-basis register on
                        {0,1}^N (mode B).  Indices are bit-strings with
                        no spatial adjacency.  quantum_potential /
                        osmotic_velocity raise RuntimeError in this mode.
                    None (default) — unspecified, no mode checks performed.
                        Backward-compatible default.
        """
        if mode not in _VALID_MODES:
            raise ValueError(
                f"mode must be 'position_space', 'qubit_register', or None; got {mode!r}"
            )
        self.n_qubits    = n_qubits
        self.hbar_eff    = hbar_eff
        self.mode        = mode
        self.m_eff       = m_eff
        self.device      = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.description = description
        self._created_at = time.time()
        self._psi_cache_enabled = psi_cache
        self._psi_cache  = None          # populated lazily by the psi property

        dim      = 2 ** n_qubits
        self._rho = torch.ones(dim,  dtype=torch.float32, device=self.device) / dim
        self._S   = torch.zeros(dim, dtype=torch.float32, device=self.device)

    # ── rho / S as properties so full assignment auto-invalidates cache ───

    @property
    def rho(self) -> torch.Tensor:
        return self._rho

    @rho.setter
    def rho(self, value: torch.Tensor):
        self._rho = value
        self._psi_cache = None

    @property
    def S(self) -> torch.Tensor:
        return self._S

    @S.setter
    def S(self, value: torch.Tensor):
        self._S = value
        self._psi_cache = None

    def _invalidate_psi(self):
        """
        Manually invalidate the ψ cache.

        Must be called after any IN-PLACE modification of rho or S
        (e.g. self.rho[i] = x, self.S[mask] += θ).

        Full assignments (self.rho = tensor) invalidate automatically
        through the property setter and do not require this call.
        """
        self._psi_cache = None

    # ── mode guard ────────────────────────────────────────────

    def _assert_position_space(self, method_name: str):
        """
        Raise RuntimeError if the state is configured as a qubit register.

        quantum_potential and osmotic_velocity are only physically meaningful
        for position-space wavefunctions on a 1D periodic grid (mode A),
        where indices represent spatial positions with periodic topology.

        In mode='qubit_register' the indices are bit-strings with no
        spatial adjacency — applying torch.roll finite differences produces
        numerically defined but physically meaningless values with no
        warning.  This guard makes the misuse explicit.

        If mode=None (unspecified), the check is skipped for backward
        compatibility — the old behaviour is preserved.
        """
        if self.mode == 'qubit_register':
            raise RuntimeError(
                f"{method_name} is only valid for position-space states "
                f"(mode='position_space').  This state is mode='qubit_register': "
                f"indices are bit-strings with no spatial adjacency, so "
                f"torch.roll finite differences are physically meaningless here.  "
                f"For qubit-register observables use reduced_density_matrix() "
                f"or entanglement_entropy() instead."
            )

    # ── wavefunction ──────────────────────────────────────────

    @property
    def psi(self) -> torch.Tensor:
        """
        ψ = √ρ · exp(iS).

        Result is cached between mutations (dirty-bit pattern).
        Cache is invalidated by:
          - assigning to self.rho or self.S  (automatic via property setter)
          - in-place modifications           (caller must call _invalidate_psi())

        Disable caching with psi_cache=False at construction time.
        """
        if not self._psi_cache_enabled:
            return torch.sqrt(self._rho + EPS) * torch.exp(1j * self._S)

        if self._psi_cache is None:
            self._psi_cache = torch.sqrt(self._rho + EPS) * torch.exp(1j * self._S)
        return self._psi_cache

    @property
    def dim(self) -> int:
        return 2 ** self.n_qubits

    # ── position-space Madelung quantities (mode A only) ──────

    @property
    def quantum_potential(self) -> torch.Tensor:
        """
        Q = -(ħ²/2m) ∇²√ρ / √ρ

        Valid ONLY when state is a position-space wavefunction on a
        1D periodic grid (mode A).  Meaningless for computational-basis
        registers (mode B) — indices have no spatial adjacency.
        Raises RuntimeError if mode='qubit_register'.
        """
        self._assert_position_space('quantum_potential')
        sqrt_rho  = torch.sqrt(self._rho + EPS)
        laplacian = torch.roll(sqrt_rho, 1) + torch.roll(sqrt_rho, -1) - 2 * sqrt_rho
        return -(self.hbar_eff ** 2 / (2 * self.m_eff)) * laplacian / sqrt_rho

    @property
    def osmotic_velocity(self) -> torch.Tensor:
        """
        u = (ħ/2m) ∇log ρ = (ħ/2m) ∇ρ / ρ

        Valid ONLY in position-space mode (mode A).
        Raises RuntimeError if mode='qubit_register'.
        """
        self._assert_position_space('osmotic_velocity')
        grad_rho = torch.roll(self._rho, -1) - torch.roll(self._rho, 1)
        return (self.hbar_eff / (2 * self.m_eff)) * grad_rho / (self._rho + EPS)

    @property
    def current_velocity(self) -> torch.Tensor:
        """v = (ħ/m) ∇S.  Valid in position-space mode (mode A)."""
        grad_S = torch.roll(self._S, -1) - torch.roll(self._S, 1)
        return (self.hbar_eff / self.m_eff) * grad_S

    # ── multi-qubit register operations (mode B) ─────────────

    def add_qubit(self, initial_p0: float = 0.5, initial_S1: float = 0.0):
        """
        Extend register by one qubit (product state, no entanglement added).

        initial_p0:  probability of |0⟩ for the new qubit (default 0.5)
        initial_S1:  phase of |1⟩ for the new qubit (default 0.0)

        ρ uses the standard Kronecker product (correct for probabilities).
        S uses an ADDITIVE tensor product:
            S_combined[2i + j] = S_old[i] + S_new[j]
        This preserves the existing phase structure — torch.kron(S_old, S_new)
        would compute S_old[i] * S_new[j], which zeros S_old whenever S_new=0.
        """
        p0 = float(np.clip(initial_p0, EPS, 1 - EPS))
        p1 = 1.0 - p0
        new_rho = torch.tensor([p0, p1],              dtype=torch.float32, device=self.device)
        new_S   = torch.tensor([0.0, initial_S1],     dtype=torch.float32, device=self.device)

        # ρ: standard Kronecker product is correct for probability densities
        self.rho = torch.kron(self._rho, new_rho)     # setter invalidates cache

        # S: additive tensor product — phases add for product states
        # S_combined[2i + j] = S_old[i] + S_new[j]
        n_old           = len(self._S)
        S_old_expanded  = self._S.repeat_interleave(2)   # (2*n_old,): S_old[i] repeated twice
        S_new_tiled     = new_S.repeat(n_old)             # (2*n_old,): new_S tiled n_old times
        self.S          = S_old_expanded + S_new_tiled    # setter invalidates cache

        self.n_qubits += 1

    def remove_qubit(self, qubit_idx: int):
        """
        Reduce register by tracing out qubit_idx.

        Partial trace on ρ (diagonal of |ψ⟩⟨ψ|) is exact:
            ρ_reduced[i] = Σ_b |ψ[i with qubit_idx=b]|²

        After tracing out a qubit the remaining state is generally MIXED
        (it cannot be written as a pure |ψ'⟩ unless the qubit was
        unentangled).  We keep ρ correct and set S = angle of the
        largest-Schmidt-coefficient component, which is the best pure-state
        approximation.  The mixed-state caveat is noted below.
        """
        if self.n_qubits <= 1:
            raise ValueError("Cannot remove the last qubit.")

        psi      = self.psi                                   # (2^N,) complex
        n        = self.n_qubits
        psi_grid = psi.view([2] * n)                          # (2, 2, …, 2)

        # Move target qubit to axis 0, reshape to (2, 2^(N-1))
        psi_grid  = psi_grid.movedim(qubit_idx, 0)
        psi_mat   = psi_grid.reshape(2, -1)                   # (2, 2^(N-1))

        # Reduced density matrix (2×2) via partial trace: ρ_q = psi_mat @ psi_mat†
        rdm       = torch.mm(psi_mat, psi_mat.conj().T)       # (2, 2) complex

        # Reduced state for the REMAINING qubits via Schmidt decomposition
        # SVD: psi_mat = U Σ Vᴴ  →  remaining state ≈ dominant Schmidt vector
        try:
            U, s, Vh = torch.linalg.svd(psi_mat, full_matrices=False)
        except Exception:
            # fallback: marginalise ρ, zero S — compute fully before assigning
            rho_mat  = self._rho.view([2] * n)
            rho_marg = rho_mat.sum(dim=qubit_idx).reshape(-1)
            rho_marg = torch.clamp(rho_marg, min=0.0)
            rho_marg = rho_marg / (rho_marg.sum() + EPS)
            self.rho = rho_marg                           # one assignment → one cache invalidation
            self.S   = torch.zeros_like(self._rho)
            self.n_qubits -= 1
            return

        # Dominant Schmidt vector gives the best pure-state approx to
        # the post-trace state.  s[0] is the largest singular value.
        dominant = s[0] * Vh[0]                               # (2^(N-1),) complex
        norm      = torch.sqrt((dominant.abs() ** 2).sum() + EPS)
        dominant  = dominant / norm

        # Warn if the pre-trace state was entangled: the post-trace state
        # is genuinely mixed, but (ρ, S) can only represent a pure state.
        # We keep the dominant Schmidt component; the rest is discarded.
        if s.shape[0] > 1:
            probs = s ** 2
            probs = probs / (probs.sum() + EPS)
            dominant_frac = float(probs[0])
            if dominant_frac < 0.99:
                warnings.warn(
                    f"remove_qubit({qubit_idx}): pre-trace state was entangled "
                    f"(dominant Schmidt weight = {dominant_frac:.3f}).  "
                    f"The exact post-trace state is mixed, but (ρ, S) can only "
                    f"represent a pure state.  Keeping the dominant Schmidt "
                    f"component and discarding {1 - dominant_frac:.3f} of the norm.  "
                    f"For the exact 2×2 reduced density matrix use "
                    f"reduced_density_matrix({qubit_idx}).",
                    stacklevel=2,
                )

        self.rho      = (dominant.abs() ** 2).float()         # setter invalidates cache
        self.S        = torch.angle(dominant).float()
        self.n_qubits -= 1

    def reduced_density_matrix(self, qubit_idx: int) -> torch.Tensor:
        """
        2×2 reduced density matrix for qubit_idx.
        Exact: ρ_q = Tr_{others}(|ψ⟩⟨ψ|).
        Returns a (2, 2) complex Hermitian matrix (not just the diagonal).
        """
        psi      = self.psi
        n        = self.n_qubits
        psi_grid = psi.view([2] * n).movedim(qubit_idx, 0)   # qubit first
        psi_mat  = psi_grid.reshape(2, -1)                    # (2, 2^(N-1))
        rdm      = torch.mm(psi_mat, psi_mat.conj().T)        # (2, 2) complex
        return rdm / (rdm.diagonal().real.sum() + EPS)

    def entanglement_entropy(self, qubit_i: int, qubit_j: int) -> float:
        """
        Von Neumann entanglement entropy of the bipartition
        {qubit_i, qubit_j} | rest.

        Correct algorithm (Schmidt decomposition on the pure state):
          1. Reshape ψ as (4, 2^(N-2)) — subsystem AB × environment
          2. SVD: U Σ Vᴴ  →  Schmidt coefficients σ_k = Σ_kk
          3. S_ent = -Σ_k σ_k² log σ_k²

        This equals the von Neumann entropy of the reduced density matrix
        ρ_AB = Tr_env(|ψ⟩⟨ψ|) without forming that 4×4 matrix explicitly.

        Returns 0 for product states and log(4) ≈ 1.386 for maximally
        entangled 2-qubit states.
        """
        psi  = self.psi                                       # (2^N,) complex
        n    = self.n_qubits

        if n < 2:
            return 0.0

        # Reshape: move qubits i, j to front, then fold into (4, rest)
        psi_grid = psi.view([2] * n)
        # Permute: [qubit_i, qubit_j, everything else]
        other = [k for k in range(n) if k not in (qubit_i, qubit_j)]
        psi_grid = psi_grid.permute([qubit_i, qubit_j] + other)
        psi_mat  = psi_grid.reshape(4, -1)                    # (4, 2^(N-2))

        # Schmidt decomposition
        try:
            _, s, _ = torch.linalg.svd(psi_mat, full_matrices=False)
        except Exception:
            return 0.0

        # Schmidt coefficients are singular values (already ≥ 0)
        # Normalise (should be ~1 for normalised ψ)
        probs = (s ** 2)
        probs = probs / (probs.sum() + EPS)
        return float(-torch.sum(probs * torch.log(probs + EPS)))

    def mutual_information(self, qubit_i: int, qubit_j: int) -> float:
        """
        MI(i,j) = S(i) + S(j) - S(ij)

        S(i), S(j): von Neumann entropy of individual reduced density matrices.
        S(ij):      entanglement entropy of the joint subsystem (via Schmidt).
        MI > 0 iff qubits i and j are correlated (classically or quantumly).
        """
        def vn_entropy_rdm(rdm: torch.Tensor) -> float:
            """von Neumann entropy of a 2×2 density matrix."""
            try:
                eigvals = torch.linalg.eigvalsh(rdm.real.float())
            except Exception:
                eigvals = rdm.diagonal().real.float()
            eigvals = torch.clamp(eigvals, min=0.0)
            eigvals = eigvals / (eigvals.sum() + EPS)
            return float(-torch.sum(eigvals * torch.log(eigvals + EPS)))

        rdm_i = self.reduced_density_matrix(qubit_i)
        rdm_j = self.reduced_density_matrix(qubit_j)
        S_i   = vn_entropy_rdm(rdm_i)
        S_j   = vn_entropy_rdm(rdm_j)
        S_ij  = self.entanglement_entropy(qubit_i, qubit_j)
        return max(0.0, S_i + S_j - S_ij)

    # ── normalization ─────────────────────────────────────────

    def normalize(self):
        rho_c = torch.clamp(self._rho, min=0.0)
        self.rho = rho_c / (rho_c.sum() + EPS)    # setter invalidates cache

    def is_valid(self) -> bool:
        return (bool(torch.all(self._rho >= 0))
                and abs(float(self._rho.sum()) - 1.0) < 1e-4
                and self._rho.shape[0] == 2 ** self.n_qubits)

    # ── persistence ───────────────────────────────────────────

    def save(self, path: str = DEFAULT_FILENAME, qkmm=None):
        checkpoint = {
            "version":          MFVM_VERSION,
            "n_qubits":         self.n_qubits,
            "hbar_eff":         self.hbar_eff,
            "m_eff":            self.m_eff,
            "description":      self.description,
            "created_at":       self._created_at,
            "saved_at":         time.time(),
            "psi_cache":        self._psi_cache_enabled,
            "mode":             self.mode,
            "rho":              self._rho.cpu(),
            "S":                self._S.cpu(),
        }
        if qkmm is not None and qkmm.psi_A is not None:
            checkpoint["qkmm"] = {
                "psi_A":             qkmm.psi_A.cpu(),
                "norms_A":           qkmm.norms_A.cpu(),
                "M":                 qkmm.M,
                "N":                 qkmm.N,
                "G":                 qkmm.G,
                "mode":              qkmm.mode,
                "approx_oversample": qkmm.approx_oversample,
                "use_real":          qkmm.use_real,
            }
        torch.save(checkpoint, path)

    @classmethod
    def load(cls, path: str, device=None):
        device     = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)

        state                     = cls.__new__(cls)
        state.device              = device
        state.n_qubits            = checkpoint["n_qubits"]
        state.hbar_eff            = checkpoint["hbar_eff"]
        state.m_eff               = checkpoint["m_eff"]
        state.description         = checkpoint.get("description", "")
        state._created_at         = checkpoint.get("created_at", 0.0)
        state._psi_cache_enabled  = checkpoint.get("psi_cache", True)
        state.mode                = checkpoint.get("mode", None)
        state._psi_cache          = None
        state._rho                = checkpoint["rho"].to(device)
        state._S                  = checkpoint["S"].to(device)
        return state, checkpoint.get("qkmm", None)

    @classmethod
    def load_or_init(cls, path=DEFAULT_FILENAME, n_qubits=1,
                     device=None, **kwargs):
        if Path(path).exists():
            print(f"[MFVM] Loading existing state from '{path}'")
            return cls.load(path, device=device)
        else:
            print(f"[MFVM] Initializing fresh {n_qubits}-qubit state -> '{path}'")
            state = cls(n_qubits=n_qubits, device=device, **kwargs)
            state.save(path)
            return state, None

    # ── display ───────────────────────────────────────────────

    def __repr__(self):
        v     = "✓" if self.is_valid() else "✗"
        cache = "on" if self._psi_cache_enabled else "off"
        mode  = self.mode or "unspecified"
        return (f"MadelungState(n_qubits={self.n_qubits}, dim={self.dim}, "
                f"device={self.device}, ħ_eff={self.hbar_eff}, "
                f"mode={mode}, valid={v}, psi_cache={cache})")

    def summary(self):
        cache = "enabled" if self._psi_cache_enabled else "disabled"
        mode  = self.mode or "unspecified"
        print(f"{'-'*50}")
        print(f"  MFVM State  —  {self.n_qubits} qubits  (dim={self.dim})")
        print(f"  Device: {self.device}   ħ_eff={self.hbar_eff}   m_eff={self.m_eff}")
        print(f"  Mode: {mode}")
        print(f"  ρ sum={float(self._rho.sum()):.6f}   "
              f"valid={'Yes' if self.is_valid() else 'NO'}")
        print(f"  ψ cache: {cache}")
        if self.description:
            print(f"  {self.description}")
        print(f"{'-'*50}")
