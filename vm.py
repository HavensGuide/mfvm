"""
mfvm/vm.py
──────────
MFVirtualMachine: the top-level orchestrator.

This is the primary interface for all MFVM computation.

Two usage modes:
────────────────────────────────────────────────────────────────────────
Mode 1: VM (execute programs)

    vm = MFVirtualMachine()
    vm.run_circuit([('hadamard', 0), ('cnot', 0, 1)])
    vm.run_hamiltonian(V=HamiltonianEvolver.double_well(), n_steps=500)
    result = vm.run_search(target=42, n_qubits=8)

Mode 2: Toolkit (compose primitives)

    vm = MFVirtualMachine(n_qubits=3)
    vm.gate.hadamard(0)
    vm.gate.cnot(0, 1)
    probs = vm.query.measure_probs()
    E = vm.query.expectation(operator)
    vm.state.save()
────────────────────────────────────────────────────────────────────────

Auto-persistence: MFVirtualMachine() checks the working directory for
an existing .mfvm state file. If found, loads it. If not, initializes
fresh and saves. State is explicitly saved on vm.save() or vm.checkpoint().

CONFIGURATION OPTIONS (constructor kwargs)
──────────────────────────────────────────
psi_cache: bool = True
    Enable ψ dirty-bit cache in MadelungState.  Recomputing ψ = √ρ·exp(iS)
    at every gate call is expensive for deep circuits.  The cache stores the
    last computed ψ and invalidates it whenever ρ or S changes.
    Set False to always recompute (useful for debugging or profiling).

derivative_mode: str = 'finite_diff'
    Backend for spatial derivatives in HamiltonianEvolver.
      'finite_diff' — torch.roll central differences, O(N), 2nd-order.
                      Appropriate for mode-B (qubit register) or fast iteration.
      'spectral'    — FFT-based, O(N log N), spectrally exact.
                      Use for position-space simulation (mode A), VQE, or
                      any case where ρ has sharp features.

qkmm_use_real: bool | None = None
    Dtype policy for QKMM inner products.
      None  — auto: float32 when inputs are real + exact mode (recommended).
      True  — always float32 (halves memory, skips complex arithmetic).
      False — always complex64 (needed for phase-encoded or complex matrices).
"""

import torch
import numpy as np
import time
import os
from pathlib import Path
from typing import Union, Callable, Optional, List, Tuple

from .state       import MadelungState, DEFAULT_FILENAME, EPS
from .query       import QueryEngine
from .gates       import GateLibrary
from .qkmm        import QKMMMemory
from .hamiltonian import HamiltonianEvolver


class MFVirtualMachine:
    """
    Madelung Flow Virtual Machine.

    The fundamental unit of computation is the v-qubit: a (ρ, S) pair
    on a two-point configuration space. This is not a von Neumann
    architecture — memory and compute are unified in the Madelung fluid.

    Key properties:
    - Non-destructive observation: reading state never collapses it
    - GPU-native: all tensors on CUDA, parallelism via broadcast
    - Dynamically extensible: add/remove qubits on the fly
    - Persistent: auto-saves to .mfvm file, auto-loads on init
    - Dual interface: VM mode (run programs) + toolkit mode (compose primitives)
    """

    def __init__(
        self,
        n_qubits:        int            = 1,
        state_file:      str            = DEFAULT_FILENAME,
        device:          torch.device   = None,
        hbar_eff:        float          = 1.0,
        m_eff:           float          = 1.0,
        description:     str            = "",
        dt:              float          = 0.01,
        psi_cache:       bool           = True,
        derivative_mode: str            = "finite_diff",
        qkmm_use_real:   Optional[bool] = None,
    ):
        """
        n_qubits:        initial qubit count (ignored if loading existing state)
        state_file:      path to .mfvm persistence file
        device:          torch.device (defaults to CUDA if available)
        hbar_eff:        effective ħ (scales quantum pressure)
        m_eff:           effective mass
        description:     free-text label attached to the state
        dt:              Hamiltonian timestep
        psi_cache:       enable ψ dirty-bit cache (see module docstring)
        derivative_mode: 'finite_diff' | 'spectral' (see module docstring)
        qkmm_use_real:   dtype policy for QKMM (see module docstring)
        """
        self.device          = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_file      = state_file
        self.dt              = dt
        self.derivative_mode = derivative_mode
        self.qkmm_use_real   = qkmm_use_real

        # ── Load or initialize state ──────────────────────────
        self.state, _qkmm_data = MadelungState.load_or_init(
            path        = state_file,
            n_qubits    = n_qubits,
            device      = self.device,
            hbar_eff    = hbar_eff,
            m_eff       = m_eff,
            description = description,
            psi_cache   = psi_cache,
        )

        # ── Attach subsystems ────────────────────────────────
        self.gate        = GateLibrary(self.state)
        self.query       = QueryEngine(self.state)
        self.hamiltonian = HamiltonianEvolver(
            self.state,
            dt=dt,
            derivative_mode=derivative_mode,
        )

        # ── Restore pre-encoded QKMM matrix if saved in prior session ──
        self._qkmm: Optional[QKMMMemory] = None
        if _qkmm_data is not None:
            self._qkmm = QKMMMemory(
                device             = self.device,
                mode               = _qkmm_data["mode"],
                approx_oversample  = _qkmm_data["approx_oversample"],
                use_real           = _qkmm_data.get("use_real", qkmm_use_real),
            )
            self._qkmm.psi_A              = _qkmm_data["psi_A"].to(self.device)
            self._qkmm.norms_A            = _qkmm_data["norms_A"].to(self.device)
            self._qkmm.M                  = _qkmm_data["M"]
            self._qkmm.N                  = _qkmm_data["N"]
            self._qkmm.G                  = _qkmm_data["G"]
            self._qkmm._use_real_active   = _qkmm_data.get("use_real", False) or False
            print(f"[MFVM] Restored QKMM matrix "
                  f"(M={self._qkmm.M}, N={self._qkmm.N}, mode={self._qkmm.mode})")

        print(f"[MFVM] Ready on {self.device}  "
              f"[psi_cache={'on' if psi_cache else 'off'}, "
              f"deriv={derivative_mode}, "
              f"qkmm_dtype={'auto' if qkmm_use_real is None else ('real' if qkmm_use_real else 'complex')}]")
        self.state.summary()

    # ─────────────────────────────────────────────────────────
    # Persistence
    # ─────────────────────────────────────────────────────────

    def save(self, path: str = None):
        """
        Save current (ρ, S) state and any pre-encoded QKMM matrix to .mfvm file.
        QKMM amplitude fields are included automatically if encode_matrix()
        has been called — enabling write-once-query-many across sessions.
        """
        p = path or self.state_file
        self.state.save(p, qkmm=self._qkmm)
        qkmm_note = (f" + QKMM (M={self._qkmm.M})"
                     if self._qkmm and self._qkmm.psi_A is not None else "")
        print(f"[MFVM] State saved -> '{p}'{qkmm_note}")

    def checkpoint(self, tag: str = ""):
        """Save a timestamped checkpoint."""
        ts   = int(time.time())
        name = (f"mfvm_checkpoint_{tag}_{ts}.mfvm" if tag
                else f"mfvm_checkpoint_{ts}.mfvm")
        self.state.save(name)
        print(f"[MFVM] Checkpoint -> '{name}'")

    def reset(self, n_qubits: int = None):
        """
        Reset register to |00...0⟩ (computational zero state).
        Quantum convention: reset means return to |0⟩, not uniform superposition.
        """
        n   = n_qubits or self.state.n_qubits
        dim = 2 ** n
        self.state.n_qubits = n
        rho      = torch.zeros(dim, dtype=torch.float32, device=self.device)
        rho[0]   = 1.0
        self.state.rho = rho    # setter invalidates psi cache
        self.state.S   = torch.zeros(dim, dtype=torch.float32, device=self.device)
        print(f"[MFVM] Reset to {n}-qubit |{'0'*n}>")

    # ─────────────────────────────────────────────────────────
    # Dynamic register management
    # ─────────────────────────────────────────────────────────

    def _reattach(self):
        """Re-bind gate/query/hamiltonian subsystems after register resize."""
        self.gate        = GateLibrary(self.state)
        self.query       = QueryEngine(self.state)
        self.hamiltonian = HamiltonianEvolver(
            self.state,
            dt=self.dt,
            derivative_mode=self.derivative_mode,
        )

    def add_qubit(self, p0: float = 0.5):
        """Add a qubit in superposition (product state with existing register)."""
        self.state.add_qubit(p0)
        self._reattach()
        print(f"[MFVM] Register extended -> {self.state.n_qubits} qubits (dim={self.state.dim})")

    def remove_qubit(self, qubit_idx: int):
        """Trace out a qubit (partial trace — reduces register size)."""
        self.state.remove_qubit(qubit_idx)
        self._reattach()
        print(f"[MFVM] Qubit {qubit_idx} removed -> {self.state.n_qubits} qubits")

    # ─────────────────────────────────────────────────────────
    # VM Mode: run programs
    # ─────────────────────────────────────────────────────────

    def run_circuit(self, circuit: List[Tuple], autosave: bool = True) -> torch.Tensor:
        """
        Execute a gate circuit on the current state.

        circuit: list of (gate_name, *args) tuples
        autosave: save state to .mfvm file after execution (default True).
                  Set False during rapid notebook iteration to avoid per-circuit
                  save overhead. Call vm.save() explicitly when you want a checkpoint.
        Example:
            vm.run_circuit([
                ('hadamard', 0),
                ('cnot', 0, 1),
                ('phase', 0, np.pi/4),
            ])
            # or without autosave for fast iteration:
            vm.run_circuit([...], autosave=False)

        Returns: current ρ (measurement probabilities)
        """
        gate_map = {
            'h':          self.gate.hadamard,
            'hadamard':   self.gate.hadamard,
            'x':          self.gate.pauli_x,
            'pauli_x':    self.gate.pauli_x,
            'y':          self.gate.pauli_y,
            'pauli_y':    self.gate.pauli_y,
            'z':          self.gate.pauli_z,
            'pauli_z':    self.gate.pauli_z,
            'phase':      self.gate.phase,
            's':          self.gate.s_gate,
            't':          self.gate.t_gate,
            'rx':         self.gate.rx,
            'ry':         self.gate.ry,
            'rz':         self.gate.rz,
            'cnot':       self.gate.cnot,
            'cx':         self.gate.cnot,
            'cz':         self.gate.cz,
            'swap':       self.gate.swap,
            'bell':       self.gate.bell_state,
            'ghz':        self.gate.ghz_state,
            'qft':        self.gate.qft,
        }

        for instruction in circuit:
            name = instruction[0].lower()
            args = instruction[1:]
            if name not in gate_map:
                raise ValueError(
                    f"Unknown gate: '{name}'. Available: {list(gate_map.keys())}")
            gate_map[name](*args)

        if autosave:
            self.state.save(self.state_file, qkmm=self._qkmm)
        return self.state.rho.clone()

    def run_hamiltonian(
        self,
        V:            Union[torch.Tensor, Callable, None] = None,
        n_steps:      int = 500,
        record_every: int = 10,
    ) -> list:
        """
        Evolve under a Hamiltonian for n_steps timesteps.
        Returns energy history.

        V: potential tensor (dim,), callable, or None (free particle)

        Example:
            history = vm.run_hamiltonian(
                V=HamiltonianEvolver.double_well(depth=1.0),
                n_steps=1000
            )
        """
        return self.hamiltonian.evolve(V=V, n_steps=n_steps, record_every=record_every)

    def run_search(
        self,
        target:       int,
        n_qubits:     int = None,
        n_iterations: int = None,
    ) -> dict:
        """
        Grover search for target in {0, ..., 2^n_qubits - 1}.
        Returns result dict with found state, probability, and timing.
        """
        n = n_qubits or self.state.n_qubits
        if n != self.state.n_qubits:
            self.reset(n_qubits=n)

        t0 = time.perf_counter()
        self.gate.grover_search(target, n_iterations=n_iterations)
        t_ms = (time.perf_counter() - t0) * 1000

        probs      = self.state.rho
        found      = int(torch.argmax(probs))
        prob_found = float(probs[found])
        success    = (found == target)

        self.state.save(self.state_file, qkmm=self._qkmm)
        return {
            "target":       target,
            "found":        found,
            "success":      success,
            "probability":  prob_found,
            "time_ms":      t_ms,
            "n_qubits":     n,
            "search_space": 2 ** n,
        }

    def run_vqe(
        self,
        V:             Union[torch.Tensor, Callable],
        n_steps:       int   = 1000,
        target_energy: float = None,
    ) -> dict:
        """
        Variational Quantum Eigensolver via Madelung evolution.
        Evolves the state to minimize ⟨H⟩ = ⟨T + V + Q⟩.

        No shot noise. No gradient estimation. Direct fluid evolution.
        Uses derivative_mode configured at VM construction time.
        For best VQE accuracy, construct with derivative_mode='spectral'.
        """
        t0 = time.perf_counter()
        history = self.hamiltonian.evolve(V=V, n_steps=n_steps, record_every=10)
        t_ms = (time.perf_counter() - t0) * 1000

        final_energy = self.hamiltonian.energy(V)
        converged    = (
            target_energy is not None
            and abs(final_energy - target_energy) / (abs(target_energy) + EPS) < 0.01
        )

        return {
            "final_energy":  final_energy,
            "target_energy": target_energy,
            "converged":     converged,
            "error_%":       (abs(final_energy - target_energy)
                              / (abs(target_energy) + EPS) * 100
                              if target_energy is not None else None),
            "time_ms":       t_ms,
            "n_steps":       n_steps,
            "history":       history,
        }

    # ─────────────────────────────────────────────────────────
    # QKMM: matrix memory interface
    # ─────────────────────────────────────────────────────────

    def encode_matrix(
        self,
        A:                 torch.Tensor,
        mode:              str            = "exact",
        approx_oversample: int            = 4,
        use_real:          Optional[bool] = None,
    ) -> "MFVirtualMachine":
        """
        Encode matrix A into QKMM memory. Write once, query many.

        mode:              'exact' (machine precision) | 'approximate' (faster, lossy)
        approx_oversample: G = ceil(log2(N)) * this factor in approximate mode
        use_real:          override qkmm_use_real set at VM construction.
                           None = use VM default (qkmm_use_real).

        Returns self for chaining: vm.encode_matrix(A).matmul(B)
        """
        effective_use_real = use_real if use_real is not None else self.qkmm_use_real
        self._qkmm = QKMMMemory(
            device             = self.device,
            mode               = mode,
            approx_oversample  = approx_oversample,
            use_real           = effective_use_real,
        ).encode(A)
        return self

    def matmul(self, B: torch.Tensor) -> torch.Tensor:
        """
        Query encoded matrix against B. encode_matrix() must be called first.
        Amortized cost falls with each repeated query.
        """
        if self._qkmm is None:
            raise RuntimeError("Call encode_matrix() before matmul().")
        return self._qkmm.query(B)

    def benchmark_qkmm(self, **kwargs) -> dict:
        """Run QKMM amortized benchmark. See QKMMMemory.benchmark() for args."""
        mem = QKMMMemory(device=self.device, use_real=self.qkmm_use_real)
        return mem.benchmark(**kwargs)

    # ─────────────────────────────────────────────────────────
    # Diagnostics
    # ─────────────────────────────────────────────────────────

    def diagnostics(self, mode: str = None):
        """
        Print full state diagnostics.

        mode: 'A' (position-space) | 'B' (computational-basis) | None (auto-detect).
        Mode A reports quantum pressure and osmotic velocity — only meaningful
        when the state is a position-space wavefunction on a periodic grid.
        Mode B (default when mode=None) skips those quantities and reports
        entanglement structure instead.

        Set mode='A' explicitly if you have initialised a position-space field.
        """
        # Auto-detect: if derivative_mode is spectral the user almost certainly
        # intends position-space (mode A). Otherwise assume mode B.
        if mode is None:
            mode = 'A' if self.derivative_mode == 'spectral' else 'B'

        print(f"\n{'='*55}")
        print(f"  MFVM Diagnostics  [mode {mode}]")
        print(f"{'='*55}")
        self.state.summary()
        print(f"  Qubits:      {self.state.n_qubits}")
        print(f"  dim:         {self.state.dim}")
        print(f"  deriv mode:  {self.derivative_mode}")
        print(f"  integrator:  {self.hamiltonian.integrator}")

        if self.state.n_qubits >= 2:
            mi = self.state.mutual_information(0, 1)
            print(f"  MI(0,1):     {mi:.4f}  ({'entangled' if mi > 0.01 else 'separable'})")
            ee = self.state.entanglement_entropy(0, 1)
            print(f"  S_ent(0,1):  {ee:.4f}")

        if mode == 'A':
            # Position-space quantities — only meaningful for mode-A (periodic grid) states
            E_kin_proxy = float(torch.sum(self.state.rho * self.state.quantum_potential))
            print(f"  <Q>:         {E_kin_proxy:.4f}  (quantum pressure, mode A only)")
            print(f"  Osmotic v:   {float(self.state.osmotic_velocity.abs().mean()):.4f}  (mean |u|, mode A only)")
        else:
            # Mode B: computational-basis register — spatial quantities are meaningless
            rho = self.state.rho
            print(f"  ρ max:       {float(rho.max()):.6f}  (index {int(rho.argmax())})")
            print(f"  ρ entropy:   {float(-(rho * torch.log(rho + EPS)).sum()):.4f}  (Shannon)")

        print(f"  QKMM:        {'encoded (M={})'.format(self._qkmm.M) if self._qkmm else 'not loaded'}")
        print(f"{'='*55}\n")

    def __repr__(self) -> str:
        return (
            f"MFVirtualMachine("
            f"n_qubits={self.state.n_qubits}, "
            f"device={self.device}, "
            f"derivative_mode={self.derivative_mode}, "
            f"state_file='{self.state_file}')"
        )
