"""
MFVM — Madelung Flow Virtual Machine
Divergence Lab / Kai Cobbs

A virtual quantum computing architecture running on classical GPU hardware.
The fundamental unit: the v-qubit — a (ρ, S) Madelung fluid on {|0⟩, |1⟩}.

Two computational paradigms:

  Gate/Hamiltonian (MFVirtualMachine, MadelungState):
    Single amplitude register evolved by quantum gates or continuous PDEs.
    Non-destructive readout, no shot noise, persistent state.

  Field evaluation (BatchedAmplitudeState):
    R independent registers evolved simultaneously as a (R, dim) field.
    One oracle pass scores all R queries against all N candidates at once.
    Distributional output: full (R, N) probability matrix per pass.
    Closer to attention/kernel methods than to sequential search.

  Memory (QKMMMemory):
    Amplitude encoding of matrix rows. Write once, query many.
    Flat O(M×N) memory independent of query count Q.
"""

from .state          import MadelungState
from .query          import QueryEngine
from .gates          import GateLibrary
from .qkmm           import QKMMMemory
from .hamiltonian    import HamiltonianEvolver
from .vm             import MFVirtualMachine
from .parallel_state import BatchedAmplitudeState

__version__ = "0.1.0"
__author__  = "Kai Cobbs / Divergence Lab"

__all__ = [
    "MFVirtualMachine",
    "MadelungState",
    "QueryEngine",
    "GateLibrary",
    "QKMMMemory",
    "HamiltonianEvolver",
    "BatchedAmplitudeState",
]
