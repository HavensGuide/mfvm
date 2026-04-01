"""
mfvm/parallel_state.py
──────────────────────
BatchedAmplitudeState: R independent (ρ, S) registers evolving simultaneously.

This is the parallel analogue of MadelungState. Where MadelungState holds
one amplitude field of shape (dim,), BatchedAmplitudeState holds R
independent fields of shape (R, dim).

USE CASES
─────────
Any workload requiring the same circuit applied to many independent
inputs simultaneously:

  Rendering:        R pixels × N scene primitives
  Database search:  R queries × N items
  Optimization:     R independent Grover searches in parallel
  Sampling:         R independent Hamiltonian evolutions

FIELD EVALUATION PARADIGM
──────────────────────────
The core operation is field evaluation, not search:

  1. init_uniform(N)           — superposition over N candidates
  2. oracle(mask)              — hard phase-flip on Boolean mask (R, N) bool
     oracle_soft(scores)       — continuous phase encoding of real-valued scores
  3. diffusion()               — inversion about mean over valid candidates only
  4. field(N)                  — read full (R, N) probability matrix

After k oracle+diffusion steps, field(N) returns a complete probability
distribution over all N candidates for all R queries simultaneously.
This is not Grover search — it is distributional scoring: all R queries
share the amplitude dynamics, extracting O(R×N) information per pass.

The mechanism is closer to belief propagation than to quantum search:
one oracle+diffusion round computes a global amplitude correlation
equivalent to one round of BP on a fully connected graph, in O(N) rather
than O(N²). oracle_soft() enables the continuous version — one-pass
global scoring where each candidate's output depends on the interference
pattern with all other candidates simultaneously, not just its own score.

ORACLE MODES
────────────
Hard oracle (oracle):      binary phase flip — Boolean match/no-match.
                           S[r, i] += π  where mask[r, i] is True.

Soft oracle (oracle_soft): continuous phase encoding of real scores.
                           S[r, i] += scores[r, i] / hbar_eff
                           Enables the one-pass global amplitude correlation:
                           ρ_new[r,i] = |2μ_r - ψ[r,i]|² where μ_r encodes
                           the coherent average over all candidates'
                           score-weighted amplitudes simultaneously.
                           This is structurally different from softmax —
                           each candidate's output depends on the full field
                           geometry, not just its own score.

DIFFUSION PADDING FIX
─────────────────────
dim is always a power of 2 (≥ N_candidates). Indices [N, dim) are padding.
The mean in diffusion must be computed over [0, valid_range) only —
including padding indices dilutes μ by (dim - N)/dim near-zero terms,
systematically weakening amplification. _valid_range is stored by
init_uniform() and used automatically by diffusion().

HBAR_EFF
────────
hbar_eff controls the phase scale for oracle_soft() and the strength of
quantum pressure. Smaller hbar_eff → sharper phase differences between
candidates → stronger amplification contrast after diffusion. Larger
hbar_eff → softer, more diffuse output distribution. The default (1.0)
matches MadelungState. For MFR-fusion use cases, set hbar_eff to the
value inferred from the empirical trajectory.

PSI CACHE CONTRACT
──────────────────
Mirrors MadelungState exactly:
  - rho and S are properties whose setters auto-invalidate the ψ cache
  - In-place modifications require _invalidate_psi() explicitly
  - psi_cache=False disables caching (useful for profiling)

RELATIONSHIP TO MadelungState
──────────────────────────────
BatchedAmplitudeState is operationally independent — it does not inherit
from MadelungState because the batch dimension changes the contract for
every method (normalization, partial trace, entanglement entropy all
have different semantics for batched registers).

However the (ρ, S) representation, the psi cache pattern, and the
Madelung fluid interpretation are identical. A single-register
BatchedAmplitudeState(1, dim) is equivalent to MadelungState(n_qubits)
where dim = 2^n_qubits.
"""

import torch
import math
from typing import Optional

from .state import MadelungState, EPS


class BatchedAmplitudeState:
    """
    R independent (ρ, S) Madelung amplitude registers.

    Shape: rho (R, dim) float32, S (R, dim) float32, psi (R, dim) complex64.

    Each row is fully independent — no entanglement across the batch
    dimension. GPU kernel calls process all R registers simultaneously.

    Parameters
    ──────────
    n_registers: int
        Number of independent amplitude registers (R).
    dim: int
        Size of each register. For quantum-register semantics, dim = 2^n_qubits.
        For general field evaluation, dim = next_power_of_2(N_candidates).
    device: torch.device, optional
        Defaults to CUDA if available.
    psi_cache: bool
        Enable dirty-bit ψ cache (default True). Set False to always
        recompute ψ — useful for debugging or isolation profiling.

    Example
    ───────
    >>> state = BatchedAmplitudeState(n_registers=576, dim=64)
    >>> state.init_uniform(N=50)                    # 576 registers, each uniform over 50 items
    >>> mask = compute_oracle_mask(rays, spheres)   # (576, 64) bool
    >>> state.oracle(mask)
    >>> state.diffusion()
    >>> probs = state.field(N=50)                   # (576, 50) probability matrix
    """

    def __init__(
        self,
        n_registers: int,
        dim:         int,
        device:      torch.device = None,
        psi_cache:   bool         = True,
        hbar_eff:    float        = 1.0,
    ):
        self.R             = n_registers
        self.dim           = dim
        self.hbar_eff      = hbar_eff
        self.device        = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self._psi_cache_en = psi_cache
        self._psi_cache:   Optional[torch.Tensor] = None

        # Tracks valid_range set by init_uniform() so diffusion() can compute
        # the global mean over real candidates only, excluding padding indices.
        self._valid_range: Optional[int] = None

        self._rho = torch.zeros(n_registers, dim, dtype=torch.float32, device=self.device)
        self._S   = torch.zeros(n_registers, dim, dtype=torch.float32, device=self.device)

    # ── rho / S properties — setters auto-invalidate psi ─────

    @property
    def rho(self) -> torch.Tensor:
        return self._rho

    @rho.setter
    def rho(self, value: torch.Tensor):
        self._rho       = value
        self._psi_cache = None

    @property
    def S(self) -> torch.Tensor:
        return self._S

    @S.setter
    def S(self, value: torch.Tensor):
        self._S         = value
        self._psi_cache = None

    def _invalidate_psi(self):
        """
        Manually invalidate the ψ cache.

        Must be called after any IN-PLACE modification of rho or S
        (e.g. self._S[mask] += θ). Full assignments through the
        property setters invalidate automatically.
        """
        self._psi_cache = None

    # ── ψ — cached ────────────────────────────────────────────

    @property
    def psi(self) -> torch.Tensor:
        """ψ = √ρ · exp(iS), shape (R, dim). Cached between mutations."""
        if not self._psi_cache_en:
            return torch.sqrt(self._rho + EPS) * torch.exp(1j * self._S)
        if self._psi_cache is None:
            self._psi_cache = (
                torch.sqrt(self._rho + EPS) * torch.exp(1j * self._S)
            )
        return self._psi_cache

    # ── Initialization ────────────────────────────────────────

    def init_uniform(self, valid_range: int):
        """
        Uniform superposition over [0, valid_range) for all R registers.
        Indices [valid_range, dim) are zeroed out (padding for power-of-2 dim).

        Stores valid_range so diffusion() can compute the global mean over
        real candidates only — padding indices would otherwise dilute μ by
        (dim - valid_range)/dim near-zero terms, weakening amplification.

        Full assignment → property setter invalidates cache automatically.
        """
        rho = torch.zeros(self.R, self.dim, dtype=torch.float32, device=self.device)
        rho[:, :valid_range] = 1.0 / valid_range
        self.rho = rho
        self.S   = torch.zeros_like(rho)
        self._valid_range = valid_range

    def init_from_state(self, state: MadelungState):
        """
        Broadcast a single MadelungState into all R registers.
        Useful when all queries start from the same initial condition.
        Sets _valid_range = state.dim (no padding when broadcasting a full state).
        """
        assert state.dim == self.dim, (
            f"dim mismatch: state.dim={state.dim}, self.dim={self.dim}")
        self.rho = state.rho.unsqueeze(0).expand(self.R, -1).clone()
        self.S   = state.S.unsqueeze(0).expand(self.R, -1).clone()
        self._valid_range = state.dim

    # ── Core field operations ─────────────────────────────────

    def oracle(self, mask: torch.Tensor):
        """
        Hard oracle: phase-flip marked candidates in-place.

        mask: (R, dim) bool — True at [r, i] flips candidate i for register r.
              Can also be (R, N) bool where N <= dim — automatically zero-padded.

        Boolean match/no-match. For continuous score encoding use oracle_soft().
        In-place S modification → _invalidate_psi() called here.
        """
        if mask.shape[1] < self.dim:
            # Pad mask to full dim so indexing is safe
            pad  = torch.zeros(self.R, self.dim - mask.shape[1],
                               dtype=torch.bool, device=self.device)
            mask = torch.cat([mask.to(self.device), pad], dim=1)
        self._S[mask] += math.pi
        self._invalidate_psi()

    def oracle_soft(self, scores: torch.Tensor, hbar_eff: float = None):
        """
        Soft oracle: encode continuous scores as phases.

        scores:   (R, N) or (R, dim) float — real-valued score per register
                  per candidate. Higher score → larger phase → stronger
                  post-diffusion amplification relative to low-score candidates.
        hbar_eff: phase scale (default: self.hbar_eff).
                  S[r, i] += scores[r, i] / hbar_eff

        WHY THIS MATTERS
        ────────────────
        Hard oracle produces binary phase contrast: marked = +π, unmarked = 0.
        After diffusion, output depends only on whether each candidate was
        marked — the scoring is Boolean.

        Soft oracle produces continuous phase contrast: each candidate's
        amplitude carries a score-proportional phase. The global mean μ_r
        becomes a Fourier-type integral over the score distribution:

            μ_r = (1/N) Σᵢ √ρ[r,i] · exp(i · scores[r,i] / ħ)

        After diffusion:
            ρ_new[r,i] = |2μ_r - ψ[r,i]|²

        This is a global amplitude correlation — each candidate's output
        depends on its phase alignment with the coherent field average,
        not just its own score. The output distribution is non-separable:
        permuting scores changes every candidate's output probability, even
        candidates whose own score didn't change.

        This is structurally different from softmax (which is separable)
        and from hard Grover (which is Boolean). It is the natural primitive
        for one-pass global belief propagation over a continuous score field.

        In-place S modification → _invalidate_psi() called here.
        """
        h = hbar_eff if hbar_eff is not None else self.hbar_eff
        sc = scores.to(self.device)
        N  = sc.shape[1]
        self._S[:, :N] += sc / h
        self._invalidate_psi()

    def diffusion(self):
        """
        Grover diffusion: ψ_new[r, x] = 2·mean_x(ψ[r, :valid]) - ψ[r, x]
        Applied to all R registers in parallel — one batched kernel.

        PADDING FIX: the mean is computed over [:, :_valid_range] only.
        Including padding indices (initialized to zero by init_uniform)
        dilutes μ by (dim - N)/dim near-zero terms — a systematic weakening
        of amplification that scales with the padding fraction. For dim=1024,
        N=600, the uncorrected dilution is 41%.

        If _valid_range is not set (e.g. manual rho/S assignment without
        init_uniform), falls back to mean over the full dim — same as the
        previous behavior.

        Full assignment → property setter invalidates cache.
        """
        vr  = self._valid_range or self.dim
        psi = self.psi                                     # (R, dim) complex

        # Mean over valid candidates only — excludes padding
        mean_psi = psi[:, :vr].mean(dim=1, keepdim=True)  # (R, 1)
        psi_new  = 2.0 * mean_psi - psi                   # (R, dim)

        self.rho = psi_new.abs() ** 2
        self.S   = torch.angle(psi_new)
        self._normalize()

    def _normalize(self):
        """Per-register L1 normalization. In-place → _invalidate_psi()."""
        rho_c = torch.clamp(self._rho, min=0.0)
        totals = rho_c.sum(dim=1, keepdim=True) + EPS
        self._rho = rho_c / totals
        self._invalidate_psi()

    # ── Readout ───────────────────────────────────────────────

    def field(self, valid_range: int) -> torch.Tensor:
        """
        Non-destructive readout: return ρ[:, :valid_range] as (R, N).

        State is never modified — no collapse, no re-preparation.
        Returns a complete probability distribution over all N candidates
        for all R registers simultaneously.

        This is the key output of field evaluation: O(R×N) information
        extracted in one pass, compared to O(1) per classical query.
        """
        return self._rho[:, :valid_range].clone()

    def field_topk(self, valid_range: int, k: int = 1) -> tuple:
        """
        Return top-k candidates per register by probability.

        Probabilities are renormalized within valid_range before ranking so
        values are comparable across registers and consistent with field_expected().
        Without renormalization, absolute probabilities may not sum to 1 within
        valid_range if padding dilution or float32 drift has occurred.

        Returns:
          values  (R, k) float — renormalized probabilities within valid_range
          indices (R, k) int   — candidate indices
        """
        probs  = self._rho[:, :valid_range]
        norm   = probs.sum(dim=1, keepdim=True) + EPS
        probs_n = probs / norm                             # renormalized within valid_range
        return torch.topk(probs_n, min(k, valid_range), dim=1)

    def field_entropy(self, valid_range: int) -> torch.Tensor:
        """
        Per-register Shannon entropy over valid candidates.

        Returns: (R,) float — entropy in nats for each register.

        Entropy = 0   → all mass on one candidate (fully concentrated)
        Entropy = ln N → uniform over all candidates (no information gained)

        Use as a convergence diagnostic for multi-pass belief refinement:
        monitor field_entropy() across oracle+diffusion rounds and stop
        when entropy stabilizes or drops below a threshold. Decreasing
        entropy means the field is concentrating onto fewer candidates —
        the belief is sharpening.

            for round in range(max_rounds):
                state.oracle(mask)
                state.diffusion()
                H = state.field_entropy(N)     # (R,) per-register entropy
                if H.max() < threshold:
                    break                       # all registers have converged
        """
        probs  = self._rho[:, :valid_range]
        norm   = probs.sum(dim=1, keepdim=True) + EPS
        p      = probs / norm                              # (R, N) normalized
        return -(p * torch.log(p + EPS)).sum(dim=1)       # (R,) entropy in nats

    def field_expected(
        self,
        values: torch.Tensor,   # (N,) or (R, N) — values to average
        valid_range: int,
    ) -> torch.Tensor:
        """
        Expected value under the field distribution.

        Returns (R,) — probability-weighted average of values for each register.
        Useful for: expected depth, expected color, expected normal, etc.

        This is the distributional rendering primitive: instead of argmax
        (classical closest hit), return the weighted mixture of all candidates.
        """
        probs = self._rho[:, :valid_range]               # (R, N)
        norm  = probs.sum(dim=1, keepdim=True) + EPS
        prob_n = probs / norm

        if values.dim() == 1:
            # Same values for all registers
            return (prob_n * values.unsqueeze(0)).sum(dim=1)
        else:
            # Per-register values
            return (prob_n * values).sum(dim=1)

    # ── Single-register extraction ────────────────────────────

    def to_madelung_state(self, register_idx: int) -> MadelungState:
        """
        Extract a single register as a MadelungState.
        Useful for debugging or single-register post-processing.
        """
        n_qubits = int(math.log2(self.dim))
        state    = MadelungState(n_qubits=n_qubits, device=self.device)
        state.rho = self._rho[register_idx].clone()
        state.S   = self._S[register_idx].clone()
        return state

    # ── Validity ──────────────────────────────────────────────

    def is_valid(self) -> bool:
        """All registers normalized and non-negative."""
        return (
            bool(torch.all(self._rho >= 0)) and
            float((self._rho.sum(dim=1) - 1.0).abs().max()) < 1e-3
        )

    # ── Display ───────────────────────────────────────────────

    def __repr__(self) -> str:
        cache = "on" if self._psi_cache_en else "off"
        valid = "✓" if self.is_valid() else "✗"
        vr    = self._valid_range if self._valid_range is not None else "unset"
        return (
            f"BatchedAmplitudeState("
            f"R={self.R}, dim={self.dim}, "
            f"valid_range={vr}, "
            f"hbar_eff={self.hbar_eff}, "
            f"device={self.device}, "
            f"psi_cache={cache}, valid={valid})"
        )

    def summary(self):
        vr = self._valid_range if self._valid_range is not None else "unset"
        print(f"{'─'*50}")
        print(f"  BatchedAmplitudeState")
        print(f"  R={self.R}  dim={self.dim}  valid_range={vr}  device={self.device}")
        print(f"  hbar_eff={self.hbar_eff}  ψ cache: {'enabled' if self._psi_cache_en else 'disabled'}")
        print(f"  ρ mean={float(self._rho.mean()):.6f}  "
              f"valid={'Yes' if self.is_valid() else 'NO'}")
        if self._valid_range is not None:
            H = self.field_entropy(self._valid_range)
            print(f"  entropy: mean={float(H.mean()):.4f}  "
                  f"min={float(H.min()):.4f}  max={float(H.max()):.4f}  (nats)")
        print(f"{'─'*50}")
