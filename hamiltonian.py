"""
mfvm/hamiltonian.py
───────────────────
HamiltonianEvolver: direct Madelung fluid evolution.

This is the second computational mode of the MFVM — as distinct from
gate-based circuits as continuous-time dynamics is from discrete logic.

Instead of decomposing H into gates (Trotter), we solve the Madelung
PDEs directly:

    Continuity:      ∂ρ/∂t + ∇·(ρ ∇S/m) = 0
    Hamilton-Jacobi: ∂S/∂t + (∇S)²/2m + V + Q = 0

where Q = -(ħ²/2m)(∇²√ρ / √ρ) is the quantum potential.

No Trotter error. No gate overhead. Exact for the chosen timestep.

Applications:
  - VQE: minimize ⟨H⟩ = ∫ ρ·(V + Q) dx by evolving toward ground state
  - Molecular simulation: V = force field potential
  - Optimization: V = objective function (quantum tunneling through barriers)
  - Field theories: V = interaction terms of the target theory

REAL-TIME EVOLUTION
───────────────────
step() / evolve() integrate the Madelung PDEs forward in real time.
Integrators: RK4 (default, 4th-order) or Euler (fast, 1st-order).

IMAGINARY-TIME EVOLUTION (ground state)
────────────────────────────────────────
imaginary_time_step() / imaginary_time_evolve() / ground_state_energy()
project the wavefunction onto the ground state via:

    ψ(τ+dτ) ∝ exp(-H·dτ) ψ(τ)

Implementation: Strang split-operator method (2nd-order):
    1. ψ₁ = ψ · exp(-V·dτ/2)                     [half potential, position space]
    2. ψ₂ = IFFT(FFT(ψ₁) · exp(-k²·dτ/(2m)))     [full kinetic, Fourier space]
    3. ψ₃ = ψ₂ · exp(-V·dτ/2)                     [half potential, position space]
    4. ψ₃ /= ‖ψ₃‖

Both ρ AND S are updated at each step.  The kinetic step in Fourier space
drives S toward the real, nodeless ground-state phase — this is the key
improvement over the previous local density re-weighting (which left S
frozen and required a pre-relaxation call to evolve()).

Requires mode='position_space': imaginary-time projection requires a
physical spatial grid.  Raises RuntimeError for mode='qubit_register'.

DERIVATIVE MODES
────────────────
Two derivative backends for real-time step() / evolve():

  'finite_diff' (default):
      Uses torch.roll central differences — O(N), second-order accurate.
      Fast for small grids, low memory.  Appropriate for gate-mode (mode B)
      where spatial adjacency is not physically meaningful anyway.

  'spectral':
      Uses FFT-based derivatives — O(N log N), spectrally exact.
      Exact for band-limited fields on periodic grids (mode A).
      The quantum potential Q is a spectral quantity by nature; spectral
      derivatives eliminate finite-difference truncation error entirely.
      Use for position-space simulation (mode A) or high-accuracy VQE.

      Gradient:  ∇f = IFFT(ik · FFT(f))
      Laplacian: ∇²f = IFFT(-k² · FFT(f))

      where k = 2π · fftfreq(N) are the wavenumbers on the periodic grid.

Note: imaginary_time_step() ALWAYS uses FFT for the kinetic sub-step,
regardless of derivative_mode — this is independent of the real-time
derivative backend choice.

  User configuration:
      evolver = HamiltonianEvolver(state, derivative_mode='spectral')
      evolver = HamiltonianEvolver(state, derivative_mode='finite_diff')
"""

import warnings
import torch
import numpy as np
from typing import Callable, Optional, Union
from .state import MadelungState, EPS


class HamiltonianEvolver:
    """
    Direct Madelung PDE evolution for a MadelungState.

    The quantum potential Q is computed analytically from ρ at each step.
    External potential V is supplied by the user as a tensor or callable.

    Time integration: 4th-order Runge-Kutta (default) or Euler (fast/approximate).
    Derivative backend: finite differences (default) or spectral (FFT).
    """

    DERIVATIVE_MODES = ("finite_diff", "spectral")

    def __init__(
        self,
        state:           MadelungState,
        dt:              float = 0.01,
        integrator:      str   = "rk4",
        derivative_mode: str   = "finite_diff",
    ):
        """
        state:           MadelungState to evolve
        dt:              timestep (tune based on hbar_eff and V magnitude)
        integrator:      'rk4' (accurate, default) | 'euler' (fast, approximate)
        derivative_mode: 'finite_diff' (default) | 'spectral'
                         See module docstring for full description.
                         Short guide:
                           Mode B (qubit register) → 'finite_diff'
                           Mode A (position-space, VQE, molecule) → 'spectral'
        """
        if derivative_mode not in self.DERIVATIVE_MODES:
            raise ValueError(
                f"derivative_mode must be one of {self.DERIVATIVE_MODES}, "
                f"got '{derivative_mode}'"
            )

        # Warn if attached to a qubit-register state.
        # Real-time step() / evolve() still run — the user may intentionally
        # want amplitude dynamics on the register — but the spatial interpretation
        # (periodic grid, torch.roll adjacency) does not hold for bit-string indices.
        # imaginary_time_step() / imaginary_time_evolve() will raise RuntimeError
        # in this mode because ground-state projection requires a physical grid.
        if getattr(state, 'mode', None) == 'qubit_register':
            warnings.warn(
                "HamiltonianEvolver attached to a qubit-register state "
                "(mode='qubit_register').  The Madelung PDEs treat array "
                "indices as spatially adjacent grid points — this is correct "
                "for position-space wavefunctions (mode A) but produces "
                "physically uninterpretable results for computational-basis "
                "registers (mode B) where indices are bit-strings.  "
                "If you intended continuous position-space evolution, use "
                "MadelungState(..., mode='position_space') and derivative_mode='spectral'.",
                stacklevel=2,
            )

        self.state           = state
        self.dt              = dt
        self.integrator      = integrator
        self.derivative_mode = derivative_mode
        self.device          = state.device
        self.t               = 0.0
        self._energy_history = []

        # Pre-compute wavenumbers for spectral mode — rebuilt if dim changes
        self._k:     Optional[torch.Tensor] = None
        self._k2:    Optional[torch.Tensor] = None
        self._k_dim: Optional[int]          = None
        if derivative_mode == "spectral":
            self._build_wavenumbers(state.dim)

    # ─────────────────────────────────────────────────────────
    # Spectral infrastructure
    # ─────────────────────────────────────────────────────────

    def _build_wavenumbers(self, N: int):
        """
        Pre-compute wavenumber arrays for the spectral derivative backend.

        k  = 2π · fftfreq(N)   — wavenumbers for gradient ∇f = IFFT(ik·FFT(f))
        k² = k²                 — for Laplacian ∇²f = IFFT(-k²·FFT(f))

        Stored as complex tensors to avoid repeated casting during evolution.
        Re-built only when N changes (grid resize).
        """
        k        = torch.fft.fftfreq(N, device=self.device) * 2 * torch.pi
        self._k  = (1j * k).to(torch.complex64)               # imaginary wavenumbers
        self._k2 = (k ** 2).to(torch.complex64)               # k² for Laplacian
        self._k_dim = N

    def _ensure_wavenumbers(self, N: int):
        """Rebuild wavenumber cache if grid size changed."""
        if self._k_dim != N:
            self._build_wavenumbers(N)

    # ─────────────────────────────────────────────────────────
    # Derivative operators — dispatched by derivative_mode
    # ─────────────────────────────────────────────────────────

    def _gradient(self, f: torch.Tensor) -> torch.Tensor:
        """
        ∇f on the periodic 1D grid.

        CONVENTION: both modes return f[i+1] - f[i-1]  (2 × proper central diff).
        This matches the original torch.roll code exactly so the two modes are
        drop-in replaceable without re-tuning dt or hbar_eff.

        finite_diff: f[i+1] - f[i-1] via torch.roll  — O(N), 2nd order
        spectral:    IFFT(2ik · FFT(f))               — O(N log N), exact
                     (factor 2 aligns with finite_diff convention)
        """
        if self.derivative_mode == "finite_diff":
            return torch.roll(f, -1) - torch.roll(f, 1)

        # Spectral path — multiply by 2 to match finite_diff convention
        self._ensure_wavenumbers(f.shape[0])
        F = torch.fft.fft(f.to(torch.complex64))
        return torch.fft.ifft(2 * self._k * F).real

    def _laplacian(self, f: torch.Tensor) -> torch.Tensor:
        """
        ∇²f on the periodic 1D grid.

        finite_diff: three-point stencil via torch.roll  — O(N), 2nd order
        spectral:    IFFT(-k² · FFT(f))                 — O(N log N), exact
        """
        if self.derivative_mode == "finite_diff":
            return torch.roll(f, 1) + torch.roll(f, -1) - 2 * f

        # Spectral path
        self._ensure_wavenumbers(f.shape[0])
        F = torch.fft.fft(f.to(torch.complex64))
        return torch.fft.ifft(-self._k2 * F).real

    # ─────────────────────────────────────────────────────────
    # Potential
    # ─────────────────────────────────────────────────────────

    def _get_V(self, V: Union[torch.Tensor, Callable, None]) -> torch.Tensor:
        """Resolve potential to a (dim,) tensor."""
        if V is None:
            return torch.zeros(self.state.dim, device=self.device)
        elif callable(V):
            return V(self.state).to(self.device)
        else:
            return V.to(self.device).float()

    # ─────────────────────────────────────────────────────────
    # Madelung equations — right-hand sides
    # ─────────────────────────────────────────────────────────

    def _quantum_potential(self, rho: torch.Tensor) -> torch.Tensor:
        """
        Q = -(ħ²/2m) · ∇²√ρ / √ρ

        Laplacian computed via the configured derivative backend.
        Spectral mode eliminates finite-difference truncation error,
        which matters especially when rho has sharp features.
        """
        sqrt_rho = torch.sqrt(rho + EPS)
        lap      = self._laplacian(sqrt_rho)
        return -(self.state.hbar_eff ** 2 / (2 * self.state.m_eff)) * lap / sqrt_rho

    def _rhs(self, rho: torch.Tensor, S: torch.Tensor, V: torch.Tensor):
        """
        Madelung right-hand sides:
          dρ/dt = -∇·(ρ ∇S/m)
          dS/dt = -(∇S)²/2m - V - Q

        Both gradient and divergence use the configured derivative backend.
        """
        hbar, m = self.state.hbar_eff, self.state.m_eff

        # Velocity field: v = ħ/m · ∇S
        grad_S = self._gradient(S)
        v      = (hbar / m) * grad_S

        # Continuity: dρ/dt = -∇·(ρ v) = -∇(ρ v)  [1D: divergence = gradient]
        rho_v     = rho * v
        div_rho_v = self._gradient(rho_v)
        drho_dt   = -div_rho_v

        # Hamilton-Jacobi: dS/dt = -(∇S)²/2m - V - Q
        Q     = self._quantum_potential(rho)
        dS_dt = -(grad_S ** 2) / (2 * m) - V - Q

        return drho_dt, dS_dt

    # ─────────────────────────────────────────────────────────
    # Time stepping
    # ─────────────────────────────────────────────────────────

    def step(self, V: Union[torch.Tensor, Callable, None] = None):
        """Advance by one timestep dt."""
        V_tensor = self._get_V(V)
        rho = self.state.rho.clone()
        S   = self.state.S.clone()

        if self.integrator == "euler":
            drho, dS = self._rhs(rho, S, V_tensor)
            self.state.rho = rho + self.dt * drho
            self.state.S   = S   + self.dt * dS

        elif self.integrator == "rk4":
            # k1
            k1_rho, k1_S = self._rhs(rho, S, V_tensor)
            # k2
            k2_rho, k2_S = self._rhs(
                rho + 0.5 * self.dt * k1_rho,
                S   + 0.5 * self.dt * k1_S,
                V_tensor
            )
            # k3
            k3_rho, k3_S = self._rhs(
                rho + 0.5 * self.dt * k2_rho,
                S   + 0.5 * self.dt * k2_S,
                V_tensor
            )
            # k4
            k4_rho, k4_S = self._rhs(
                rho + self.dt * k3_rho,
                S   + self.dt * k3_S,
                V_tensor
            )
            self.state.rho = rho + (self.dt / 6) * (k1_rho + 2*k2_rho + 2*k3_rho + k4_rho)
            self.state.S   = S   + (self.dt / 6) * (k1_S   + 2*k2_S   + 2*k3_S   + k4_S)

        # Enforce non-negativity and normalization
        rho_new = self.state.rho
        rho_new = torch.clamp(rho_new, min=0.0)
        total   = rho_new.sum()
        if total > EPS:
            self.state.rho = rho_new / total
        else:
            # Recovery: reset to uniform if evolution blew up
            self.state.rho = torch.ones_like(rho_new) / rho_new.numel()
        self.t += self.dt

    def evolve(
        self,
        n_steps:      int,
        V:            Union[torch.Tensor, Callable, None] = None,
        record_every: int = 10,
    ) -> list:
        """
        Evolve for n_steps timesteps.
        Returns energy history sampled every record_every steps.
        """
        V_tensor = self._get_V(V)
        history  = []

        for step_i in range(n_steps):
            self.step(V_tensor)
            if step_i % record_every == 0:
                E = self.energy(V_tensor)
                history.append({"t": self.t, "energy": E})

        self._energy_history.extend(history)
        return history

    # ─────────────────────────────────────────────────────────
    # Energy
    # ─────────────────────────────────────────────────────────

    def energy(self, V: Union[torch.Tensor, Callable, None] = None) -> float:
        """
        Total energy: E = ⟨T⟩ + ⟨V⟩ + ⟨Q⟩

        Kinetic:   T = ∫ ρ (∇S)²/2m dx
        Potential: V = ∫ ρ V(x) dx
        Quantum:   Q = ∫ ρ Q(x) dx

        Gradient uses the configured derivative backend.
        """
        V_tensor = self._get_V(V)
        rho = self.state.rho
        S   = self.state.S
        m   = self.state.m_eff

        grad_S = self._gradient(S)
        T      = float(torch.sum(rho * grad_S ** 2 / (2 * m)))
        Vexp   = float(torch.sum(rho * V_tensor))
        Q      = self._quantum_potential(rho)
        Qexp   = float(torch.sum(rho * Q))

        return T + Vexp + Qexp

    def imaginary_time_step(
        self,
        V:    Union[torch.Tensor, Callable, None] = None,
        dtau: Optional[float] = None,
    ):
        """
        Single imaginary-time step via the split-operator method.

        Imaginary-time evolution (t → -iτ) drives the wavefunction toward
        the ground state by suppressing excited-state components:
            ψ(τ+dτ) ∝ exp(-H·dτ) ψ(τ)

        This implementation uses the Strang-splitting (second-order accurate):
            ψ₁ = ψ  · exp(-V·dτ/2)          [half potential step, position space]
            ψ₂ = IFFT(FFT(ψ₁) · exp(-k²·dτ/(2m)))  [full kinetic step, Fourier space]
            ψ₃ = ψ₂ · exp(-V·dτ/2)          [half potential step, position space]
            ψ₃ → ψ₃ / ‖ψ₃‖                  [renormalize]

        Both ρ = |ψ₃|² and S = arg(ψ₃) are updated.  The kinetic step in
        Fourier space is the critical difference from the previous
        ground_state_energy: it damps high-momentum components and drives
        S toward the (real, nodeless) ground-state phase, whereas a pure
        local density re-weighting leaves S frozen at its current value.

        The Fourier kinetic step is ALWAYS spectral regardless of the
        derivative_mode setting — this is the exact and natural representation
        of exp(-T·dτ) and does not depend on finite-difference adjacency.

        Raises RuntimeError if mode='qubit_register': ground-state projection
        requires a physical spatial grid; bit-string indices have no kinetic
        energy structure.

        Parameters
        ----------
        V    : potential (tensor, callable, or None)
        dtau : imaginary timestep (default: self.dt)
        """
        if getattr(self.state, 'mode', None) == 'qubit_register':
            raise RuntimeError(
                "imaginary_time_step() requires a position-space state "
                "(mode='position_space').  Ground-state projection via "
                "exp(-T·dτ) in Fourier space requires a physical spatial "
                "grid; bit-string indices carry no kinetic energy structure."
            )

        dtau     = dtau if dtau is not None else self.dt
        V_tensor = self._get_V(V)

        # Ensure wavenumbers exist (may not be built if derivative_mode='finite_diff')
        self._ensure_wavenumbers(self.state.dim)

        psi = self.state.psi.clone()           # complex64, shape (dim,)

        # ── Half potential step ───────────────────────────────
        V_c   = V_tensor.to(torch.complex64)
        psi   = psi * torch.exp(-0.5 * dtau * V_c)

        # Renormalize between sub-steps to keep numerics stable
        psi   = psi / (torch.sqrt((psi.abs() ** 2).sum()) + EPS)

        # ── Full kinetic step in Fourier space ────────────────
        # exp(-T·dτ)|ψ⟩ = IFFT(exp(-k²·dτ/(2m)) · FFT(ψ))
        # This correctly damps high-momentum components and rotates S
        # toward the real ground-state phase — something a position-space
        # density re-weighting cannot do.
        k2_real = self._k2.real                            # (dim,) float, k² values
        psi_hat = torch.fft.fft(psi)
        psi_hat = psi_hat * torch.exp(
            -dtau * k2_real.to(torch.complex64) / (2.0 * self.state.m_eff)
        )
        psi     = torch.fft.ifft(psi_hat)

        # ── Half potential step ───────────────────────────────
        psi     = psi * torch.exp(-0.5 * dtau * V_c)

        # ── Normalize ─────────────────────────────────────────
        norm    = torch.sqrt((psi.abs() ** 2).sum() + EPS)
        psi     = psi / norm

        # ── Decompose back to (ρ, S) ─────────────────────────
        # Full assignments → property setters invalidate psi cache
        self.state.rho = (psi.abs() ** 2).float()
        self.state.S   = torch.angle(psi).float()
        self.t        += dtau

    def imaginary_time_evolve(
        self,
        n_steps:      int,
        V:            Union[torch.Tensor, Callable, None] = None,
        dtau:         Optional[float]                     = None,
        record_every: int                                 = 10,
        tol:          Optional[float]                     = None,
    ) -> list:
        """
        Evolve under imaginary time for n_steps steps, returning energy history.

        Each step is a second-order split-operator imaginary-time step (see
        imaginary_time_step).  Energy is sampled every record_every steps
        using the real-time energy() method, which measures ⟨T⟩ + ⟨V⟩ + ⟨Q⟩
        on the current (ρ, S).

        Early stopping: if tol is set, stops when the energy change between
        consecutive recorded steps falls below tol.

        Parameters
        ----------
        n_steps      : maximum number of imaginary-time steps
        V            : potential (tensor, callable, or None)
        dtau         : imaginary timestep (default: self.dt)
        record_every : energy sampling interval
        tol          : convergence tolerance on ΔE (default: no early stopping)

        Returns
        -------
        list of {'tau': float, 'energy': float} dicts, one per recorded step
        """
        V_tensor = self._get_V(V)
        history  = []
        prev_E   = None

        for step_i in range(n_steps):
            self.imaginary_time_step(V_tensor, dtau=dtau)

            if step_i % record_every == 0:
                E = self.energy(V_tensor)
                history.append({"tau": self.t, "energy": E})

                if tol is not None and prev_E is not None:
                    if abs(E - prev_E) < tol:
                        break
                prev_E = E

        self._energy_history.extend(history)
        return history

    def ground_state_energy(
        self,
        V:      Union[torch.Tensor, Callable, None] = None,
        n_steps: int   = 500,
        dtau:    Optional[float] = None,
        tol:     float = 1e-6,
    ) -> float:
        """
        Estimate the ground state energy by imaginary-time projection.

        Runs imaginary_time_evolve() until convergence (|ΔE| < tol) or
        n_steps steps, then returns the final energy.  The state is updated
        in place — on return, (ρ, S) represent the best ground-state
        approximation reached.

        For a quick single-step estimate without evolving the state, use
        energy() on a pre-relaxed state instead.

        Parameters
        ----------
        V       : potential (tensor, callable, or None)
        n_steps : maximum imaginary-time steps (default 500)
        dtau    : imaginary timestep (default: self.dt)
        tol     : convergence threshold on |ΔE| between recorded steps
        """
        history = self.imaginary_time_evolve(
            n_steps=n_steps, V=V, dtau=dtau,
            record_every=10, tol=tol,
        )
        return history[-1]["energy"] if history else self.energy(V)

    # ─────────────────────────────────────────────────────────
    # Standard potentials
    # ─────────────────────────────────────────────────────────

    @staticmethod
    def double_well(depth: float = 1.0, width: float = 1.0) -> Callable:
        """V(x) = depth · (x² - width²)²  — double well potential."""
        def V(state: MadelungState) -> torch.Tensor:
            dim = state.dim
            x   = torch.linspace(-2, 2, dim, device=state.device)
            return depth * (x**2 - width**2)**2
        return V

    @staticmethod
    def harmonic(omega: float = 1.0) -> Callable:
        """V(x) = ½ω²x²  — harmonic oscillator."""
        def V(state: MadelungState) -> torch.Tensor:
            dim = state.dim
            x   = torch.linspace(-3, 3, dim, device=state.device)
            return 0.5 * omega**2 * x**2
        return V

    @staticmethod
    def optimization_landscape(objective: Callable) -> Callable:
        """
        Wrap any scalar objective function as a quantum potential.
        The Madelung fluid will tunnel through local minima.
        """
        def V(state: MadelungState) -> torch.Tensor:
            dim = state.dim
            x   = torch.linspace(0, 1, dim, device=state.device)
            return torch.tensor([objective(xi.item()) for xi in x],
                                device=state.device, dtype=torch.float32)
        return V

    # ─────────────────────────────────────────────────────────
    # Display
    # ─────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"HamiltonianEvolver("
            f"integrator={self.integrator}, "
            f"derivative_mode={self.derivative_mode}, "
            f"dt={self.dt}, t={self.t:.4f})"
        )
