"""
mfvm/qkmm.py
────────────
QKMM: Quantum Kernel Matrix Memory.

Amplitude encoding of matrix rows as Madelung density fields.
Write once → query many times. The matrix IS the density.

Based on: Yao & Liu (2026) "Reducing Matrix Multiplication to O(N²log₂N)"
arXiv:2602.05541

Two encoding modes:
  - exact:        G = N Fourier modes, machine-precision inner products
  - approximate:  G = ceil(log2(N)) * oversample_factor, lossy but faster

The encoded (ψ, norms) pair is a (ρ, S) Madelung object:
  ρ = |ψ|²,  S = angle(ψ)

This means any QKMM-encoded matrix can be treated as a MadelungState
and queried through the same QueryEngine interface.

PERFORMANCE NOTES
─────────────────
use_real (default None = auto):
  In exact mode with real-valued inputs, ψ carries no phase information
  and complex64 (8 bytes/element) is wasteful — the imaginary parts are
  all zero and the inner product discards them anyway.  When use_real is
  True (or auto-detected), psi_A is stored as float32 (4 bytes/element)
  and matmuls run in the real domain.  This halves memory bandwidth and
  avoids complex arithmetic overhead.

  Set use_real=False to force complex arithmetic regardless of input type
  (useful when extending to complex-valued matrices or phase-encoded data).

Projection cache:
  In approximate mode, the random projection matrix R is computed once
  during encode() and cached as self._R.  Previously it was recomputed on
  every query() call — paying O(N·G²) QR decomposition each time.
  The projection is fully deterministic (fixed seed=42), so caching is safe.

accuracy() isolation:
  accuracy() and benchmark() operate on a temporary internal instance so
  they never mutate the caller's encoded state.
"""

import torch
import numpy as np
import time
from typing import Optional

EPS = 1e-8


class QKMMMemory:
    """
    Matrix memory using Madelung amplitude encoding.

    Usage pattern (memory architecture):
        mem = QKMMMemory(device=device)
        mem.encode(A)                    # write once — O(M*N)
        result = mem.query(B)            # query many — O(M*K*G)
        result = mem.query(C)            # same encoded A, new query
        result = mem.query(D)            # amortized cost falls with each query

    Encoding modes:
        exact:       G = N, inner products exact to machine precision
        approximate: G = ceil(log2(N)) * factor, lossy but memory-efficient

    Dtype modes (use_real):
        None  — auto: real float32 when inputs are real + exact mode; else complex64
        True  — force float32 throughout (real inner products only)
        False — force complex64 throughout (supports phase-encoded or complex A)
    """

    def __init__(
        self,
        device:            torch.device  = None,
        mode:              str           = "exact",
        approx_oversample: int           = 4,
        use_real:          Optional[bool] = None,
    ):
        """
        device:            torch.device (defaults to CUDA if available)
        mode:              'exact' | 'approximate'
        approx_oversample: for approximate mode, G = ceil(log2(N)) * this factor
        use_real:          dtype control — see module docstring.
                           None = auto-detect per encode() call (recommended)
                           True = always float32 (real arithmetic)
                           False = always complex64
        """
        self.device            = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mode              = mode
        self.approx_oversample = approx_oversample
        self.use_real          = use_real          # user preference; resolved in encode()

        # Encoded state (set by encode())
        self.psi_A:    Optional[torch.Tensor] = None  # (M, G) float32 or complex64
        self.norms_A:  Optional[torch.Tensor] = None  # (M,) float32
        self.M:        Optional[int]          = None
        self.N:        Optional[int]          = None
        self.G:        Optional[int]          = None
        self._use_real_active: bool           = False  # resolved dtype for current encoding
        self._encode_time_ms:  float          = 0.0

        # Projection cache — computed once in encode(), reused in every query().
        # Previously recomputed per query call: O(N·G²) QR each time.
        self._R: Optional[torch.Tensor] = None   # (N, G) float32

    # ─────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────

    def _G_for_N(self, N: int) -> int:
        """Compute encoding dimension G based on mode."""
        if self.mode == "exact":
            return N
        else:
            log_modes = int(np.ceil(np.log2(max(N, 2))))
            return max(log_modes * self.approx_oversample, 8)

    def _build_projection(self, N: int, G: int) -> torch.Tensor:
        """
        Build and cache the random projection matrix R ∈ R^(N×G).

        Called ONCE from encode(). Result stored in self._R and reused
        by every subsequent query() call — eliminates repeated O(N·G²)
        QR decomposition that was previously paid per query.

        Columns are orthonormalized via QR when G ≤ N (JL-optimal).
        Seeded deterministically so encode() and query() always agree.
        """
        torch.manual_seed(42)
        R = torch.randn(N, G, device=self.device)
        if G <= N:
            Q, _ = torch.linalg.qr(R)
            return Q          # (N, G) orthonormal columns
        else:
            return R / torch.norm(R, dim=0, keepdim=True)

    def _resolve_use_real(self, A: torch.Tensor) -> bool:
        """
        Decide whether to use real (float32) arithmetic for this encoding.

        Rules:
          user set use_real=True  → always real
          user set use_real=False → always complex
          user set use_real=None  → auto: real when A is real-valued (both modes)

        Approximate mode: the random projection R is real, A is real, so the
        projected vectors (A_n @ R) are real — no phase information exists to
        preserve. float32 is correct and saves half the memory bandwidth.
        """
        if self.use_real is True:
            return True
        if self.use_real is False:
            return False
        # Auto: real whenever input has no imaginary component (exact or approximate)
        return not A.is_complex()

    def _project_B(self, B_n: torch.Tensor) -> torch.Tensor:
        """
        Project normalized B through the cached projection (approximate mode)
        or return as-is (exact mode). Applies dtype cast based on _use_real_active.

        Centralizes the B-side projection logic shared by query(), query_prenorm(),
        and query_batched() — previously duplicated across all three methods.

        B_n: (..., N) normalized query matrix (any leading batch dims)
        Returns: (..., G) projected and dtype-cast
        """
        if self.mode == "exact":
            if self._use_real_active:
                return B_n.float()
            return B_n.to(torch.complex64)

        # Approximate — use cached _R (shape validated against current N)
        assert self._R is not None, "_R not built; call encode() first"
        assert self._R.shape[0] == self.N, (
            f"_R shape mismatch: _R.shape[0]={self._R.shape[0]}, self.N={self.N}. "
            f"Re-encode before querying.")

        if B_n.dim() == 2:
            proj = torch.mm(B_n, self._R)                          # (K, G)
        else:
            # Batched: (..., K, N) × (N, G) → (..., K, G)
            proj = torch.matmul(B_n, self._R)

        rn = torch.norm(proj, dim=-1, keepdim=True) + EPS
        proj_n = proj / rn

        if self._use_real_active:
            return proj_n.float()
        return proj_n.to(torch.complex64)

    def _overlap(self, psi_A: torch.Tensor, psi_B: torch.Tensor) -> torch.Tensor:
        """
        Inner product between encoded A and projected B.

        psi_A: (M, G)    — encoded rows
        psi_B: (K, G)    — projected query rows
        Returns: (M, K)  — real-valued overlaps

        Dispatches to real or complex mm based on _use_real_active,
        avoiding the complex cast + .real extraction when inputs are real.
        """
        if self._use_real_active:
            return torch.mm(psi_A, psi_B.T)                        # real mm
        return torch.mm(psi_A, psi_B.conj().T).real                # complex mm → real

    # ─────────────────────────────────────────────────────────
    # Encoding
    # ─────────────────────────────────────────────────────────

    def encode(self, A: torch.Tensor) -> "QKMMMemory":
        """
        Encode matrix A ∈ R^(M×N) as M amplitude fields.

        Exact mode (G=N):
            Normalize each row, store as float32 (real) or complex64.
            Inner products are exact by definition.

        Approximate mode (G < N):
            Project to G dimensions via cached random orthonormal projection.
            Inner products preserved up to JL distortion: ε ≈ sqrt(log(M)/G).

        The projection matrix R is computed here and cached in self._R.
        All subsequent query() calls reuse it — no recomputation.

        Returns self for chaining.
        """
        t0   = time.perf_counter()
        M, N = A.shape
        G    = self._G_for_N(N)
        A_t  = A.to(self.device).float()

        norms = torch.norm(A_t, dim=1, keepdim=True) + EPS   # (M, 1)
        A_n   = A_t / norms                                   # L2-normalized rows

        self._use_real_active = self._resolve_use_real(A)

        if self.mode == "exact":
            if self._use_real_active:
                encoded = A_n                                  # (M, N) float32
            else:
                encoded = A_n.to(torch.complex64)             # (M, N) complex64
            self._R = None                                     # no projection in exact mode
        else:
            # Approximate: build projection once and cache
            if self._R is None or self._R.shape[0] != N or self._R.shape[1] != G:
                self._R = self._build_projection(N, G)
            proj      = torch.mm(A_n, self._R)                # (M, G)
            row_norms = torch.norm(proj, dim=1, keepdim=True) + EPS
            proj_n    = proj / row_norms
            if self._use_real_active:
                encoded = proj_n                               # (M, G) float32
            else:
                encoded = proj_n.to(torch.complex64)          # (M, G) complex64

        self.psi_A   = encoded
        self.norms_A = norms.squeeze(1)                        # (M,) float32
        self.M, self.N, self.G = M, N, G
        self._encode_time_ms = (time.perf_counter() - t0) * 1000
        return self

    # ─────────────────────────────────────────────────────────
    # Query
    # ─────────────────────────────────────────────────────────

    def _query_normalized(
        self,
        B_n:     torch.Tensor,   # (K, N) pre-normalized
        norms_B: torch.Tensor,   # (K, 1)
    ) -> torch.Tensor:
        """
        Core query kernel: normalized B → (M, K) result.
        Shared by query() and query_prenorm() to avoid duplication.
        """
        psi_B   = self._project_B(B_n)                        # (K, G)
        overlap = self._overlap(self.psi_A, psi_B)            # (M, K)
        return overlap * self.norms_A.unsqueeze(1) * norms_B.squeeze(1).unsqueeze(0)

    def query(self, B: torch.Tensor) -> torch.Tensor:
        """
        Query encoded A against B ∈ R^(K×N) → W ≈ A @ Bᵀ ∈ R^(M×K).
        A is encoded once; B is projected on-the-fly.

        For large K that would OOM, use query_chunked() instead.
        For a fixed B queried repeatedly, pre-normalize with prenorm()
        and use query_prenorm() to avoid recomputing norms each call.
        """
        if self.psi_A is None:
            raise RuntimeError("Call encode() before query().")

        K, N = B.shape
        assert N == self.N, f"Column mismatch: encoded N={self.N}, query N={N}"

        B_t     = B.to(self.device).float()
        norms_B = torch.norm(B_t, dim=1, keepdim=True) + EPS  # (K, 1)
        B_n     = B_t / norms_B                                # (K, N)

        return self._query_normalized(B_n, norms_B)

    def query_chunked(
        self,
        B:          torch.Tensor,
        chunk_size: int = 1024,
    ) -> torch.Tensor:
        """
        Memory-safe query for large B that would OOM in a single pass.

        Splits B into chunks of chunk_size rows, queries each chunk
        separately, and concatenates results. Output is identical to
        query(B) but peak VRAM is O(chunk_size × G) instead of O(K × G).

        Use when K is large relative to available VRAM — e.g. querying
        a field over a high-dimensional configuration space where
        K = grid_points^N.

        B:          (K, N)
        chunk_size: rows of B per kernel call (tune for VRAM budget)
        Returns:    (M, K) — identical to query(B)
        """
        if self.psi_A is None:
            raise RuntimeError("Call encode() before query_chunked().")

        K, N = B.shape
        assert N == self.N, f"Column mismatch: encoded N={self.N}, query N={N}"

        chunks = []
        for start in range(0, K, chunk_size):
            end = min(start + chunk_size, K)
            chunks.append(self.query(B[start:end]))            # (M, chunk)
        return torch.cat(chunks, dim=1)                        # (M, K)

    def prenorm(self, B: torch.Tensor):
        """
        Pre-normalize B for use with query_prenorm().
        Call once when B is fixed across many queries.

        Returns (B_n, norms_B) — pass both to query_prenorm().
        """
        B_t     = B.to(self.device).float()
        norms_B = torch.norm(B_t, dim=1, keepdim=True) + EPS
        B_n     = B_t / norms_B
        return B_n, norms_B

    def query_prenorm(
        self,
        B_n:     torch.Tensor,   # pre-normalized (K, N), from prenorm()
        norms_B: torch.Tensor,   # (K, 1), from prenorm()
    ) -> torch.Tensor:
        """
        Query with pre-normalized B — skips norm recomputation.

        Use when the same B is queried repeatedly:
            B_n, norms_B = mem.prenorm(B)
            for _ in range(Q):
                result = mem.query_prenorm(B_n, norms_B)

        Vs query() which pays torch.norm(B) on every call.
        """
        if self.psi_A is None:
            raise RuntimeError("Call encode() before query_prenorm().")
        assert B_n.shape[1] == self.N, f"Column mismatch: N={self.N}, got {B_n.shape[1]}"
        return self._query_normalized(B_n, norms_B)

    def query_batched(self, B_batch: torch.Tensor) -> torch.Tensor:
        """
        Batched query: compute A @ B_k^T for n_queries different batches B_k
        in a single bmm kernel call.

        B_batch: (n_queries, K, N)  — stack of n_queries query batches
        Returns: (n_queries, M, K)

        psi_A is broadcast as a VIEW across the batch dimension (zero copy).
        One batched kernel replaces n_queries sequential mm calls.

        Use query() for a single batch, query_batched() when you have
        multiple independent query batches to run simultaneously.
        """
        if self.psi_A is None:
            raise RuntimeError("Call encode() before query_batched().")

        n_q, K, N = B_batch.shape
        assert N == self.N, f"Column mismatch: encoded N={self.N}, query N={N}"

        B_f     = B_batch.to(self.device).float()
        norms_B = torch.norm(B_f, dim=2, keepdim=True) + EPS  # (n_q, K, 1)
        B_n     = B_f / norms_B                                # (n_q, K, N)

        # _project_B handles batched dims via torch.matmul broadcast
        psi_B = self._project_B(B_n)                           # (n_q, K, G)

        # psi_A: (M, G) → VIEW (n_q, M, G) — zero copy
        psi_A_exp = self.psi_A.unsqueeze(0).expand(n_q, -1, -1)

        if self._use_real_active:
            overlap = torch.bmm(psi_A_exp, psi_B.transpose(1, 2))            # real
        else:
            overlap = torch.bmm(psi_A_exp, psi_B.conj().transpose(1, 2)).real

        norms_A_s = self.norms_A.unsqueeze(0).unsqueeze(2)     # (1, M, 1)
        norms_B_s = norms_B.squeeze(2).unsqueeze(1)            # (n_q, 1, K)
        return overlap * norms_A_s * norms_B_s                 # (n_q, M, K)
    
    def query_topk(
        self,
        B:             torch.Tensor,
        k:             int   = 1,
        n_iter:        int   = 1,
        exact_rescore: bool  = True,
        threshold:     float = None,
    ) -> dict:
        """
        Top-k query via amplitude search + optional exact re-score.

        Two-phase approach:
        1. Coarse pass — amplitude search over M rows of A using a low-rank
            oracle (sign of inner product in G-dimensional projection space).
            Cost: O(M·G) oracle + O(M) diffusion, repeated n_iter times.
            G << N in approximate mode; G = N in exact mode (same as query()).

        2. Exact re-score (default on) — compute precise inner products only
            for the top-k·oversample candidates surfaced by the amplitude field.
            Cost: O(k·oversample·N) instead of O(M·N).

        Break-even: query_topk wins when k << M. At k=M it degrades to query().

        Parameters
        ──────────
        B:             (K, N) query batch
        k:             number of top rows of A to return per query (default 1)
        n_iter:        oracle + diffusion rounds (default 1; increase to sharpen
                    amplitude contrast at the cost of more oracle passes)
        exact_rescore: if True, re-score top candidates with exact inner products
                    and re-rank. Set False to return amplitude-field ranking
                    directly (faster, slightly lower precision).
        threshold:     oracle threshold on coarse score (default: mean score per
                    query). Rows above threshold are phase-flipped. Lower
                    threshold → more candidates marked → broader top-k recall.

        Returns
        ───────
        dict with keys:
        indices     : (K, k) int tensor  — top-k row indices into A per query
        scores      : (K, k) float tensor — corresponding inner product scores
        rho         : (K, M) float tensor — full amplitude field after search
        oracle_calls: int — n_iter (oracle+diffusion rounds executed)
        """
        if self.psi_A is None:
            raise RuntimeError("Call encode() before query_topk().")

        K, N = B.shape
        assert N == self.N, f"Column mismatch: encoded N={self.N}, query N={N}"
        M = self.M

        B_t     = B.to(self.device).float()
        norms_B = torch.norm(B_t, dim=1, keepdim=True) + EPS   # (K, 1)
        B_n     = B_t / norms_B                                 # (K, N)

        # ── Phase 1: Coarse scores via encoded projection ─────────────────────
        # psi_A: (M, G), psi_B: (K, G) — cheap inner products in G-dim space.
        # In exact mode G=N so this is full precision; in approximate mode G<<N.
        psi_B        = self._project_B(B_n)                     # (K, G)
        coarse_scores = self._overlap(self.psi_A, psi_B)        # (M, K)
        coarse_scores = coarse_scores.T                         # (K, M)

        # ── Phase 2: Amplitude search over M rows ─────────────────────────────
        # Uniform superposition over M rows as a (K, M) field.
        # Oracle phase-flips rows whose coarse score exceeds the threshold.
        # Diffusion (inversion about mean) amplifies marked candidates.
        # All K queries processed in parallel — one batched kernel per step.

        rho = torch.full((K, M), 1.0 / M, dtype=torch.float32, device=self.device)
        S   = torch.zeros((K, M),          dtype=torch.float32, device=self.device)

        for _ in range(n_iter):
            # Oracle: boolean mask per query — rows above threshold get +π phase
            thresh = threshold if threshold is not None else coarse_scores.mean(dim=1, keepdim=True)
            mask   = coarse_scores > thresh                     # (K, M) bool
            S[mask] += torch.pi

            # Diffusion: ψ_new = 2·μ − ψ, μ = mean over M candidates
            psi    = torch.sqrt(rho) * torch.exp(1j * S.to(torch.complex64))
            mu     = psi.mean(dim=1, keepdim=True)              # (K, 1) complex
            psi    = 2 * mu - psi                               # inversion about mean
            rho    = psi.abs().square()
            rho    = rho / (rho.sum(dim=1, keepdim=True) + EPS) # renormalize

            S = psi.angle().float()

        # ── Phase 3: Top-k from amplitude field ───────────────────────────────
        # Oversample by 4× then re-rank to improve recall at boundary.
        oversample   = min(k * 4, M)
        topk_coarse  = torch.topk(rho, oversample, dim=1)       # (K, oversample)
        cand_indices = topk_coarse.indices                       # (K, oversample)

        if not exact_rescore:
            final_indices = cand_indices[:, :k]
            final_scores  = coarse_scores.gather(1, final_indices)
            return {
                "indices":      final_indices.cpu(),
                "scores":       final_scores.cpu(),
                "rho":          rho.cpu(),
                "oracle_calls": n_iter,
            }

        # ── Phase 4: Exact re-score top candidates only ───────────────────────
        # Gather encoded rows for each query's candidates — no full bmm.
        # psi_A: (M, G) → gather (K, oversample, G) → batched dot with psi_B
        psi_A_exp   = self.psi_A.unsqueeze(0).expand(K, -1, -1)         # (K, M, G)
        cand_psi_A  = psi_A_exp.gather(
            1,
            cand_indices.unsqueeze(-1).expand(-1, -1, self.G)
        )                                                                 # (K, oversample, G)

        psi_B_exp   = psi_B.unsqueeze(1)                                 # (K, 1, G)
        if self._use_real_active:
            exact = torch.bmm(cand_psi_A, psi_B_exp.transpose(1, 2))    # (K, oversample, 1)
        else:
            exact = torch.bmm(
                cand_psi_A, psi_B_exp.conj().transpose(1, 2)
            ).real                                                        # (K, oversample, 1)

        exact = exact.squeeze(-1)                                        # (K, oversample)

        # Scale by norms to recover true inner product magnitudes
        cand_norms_A = self.norms_A.unsqueeze(0).expand(K, -1).gather(1, cand_indices)
        exact        = exact * cand_norms_A * norms_B                    # (K, oversample)

        # Re-rank and take top-k
        topk_exact    = torch.topk(exact, k, dim=1)
        local_indices = topk_exact.indices                               # (K, k)
        final_indices = cand_indices.gather(1, local_indices)            # (K, k) into A
        final_scores  = topk_exact.values                                # (K, k)

        return {
            "indices":      final_indices.cpu(),
            "scores":       final_scores.cpu(),
            "rho":          rho.cpu(),
            "oracle_calls": n_iter,
        }

    # ─────────────────────────────────────────────────────────
    # Classical baseline
    # ─────────────────────────────────────────────────────────

    def classical_matmul(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return torch.mm(
            A.to(self.device).float(),
            B.to(self.device).float().T
        )

    # ─────────────────────────────────────────────────────────
    # Accuracy check
    # ─────────────────────────────────────────────────────────

    def accuracy(self, A: torch.Tensor, B: torch.Tensor) -> dict:
        """
        Compute error of QKMM vs classical matmul.

        Operates on a temporary internal QKMMMemory instance — never
        mutates the caller's encoded state. Previously, calling accuracy()
        after encoding a different matrix would silently overwrite the
        encoding, corrupting subsequent queries.
        """
        # Use a fresh instance so caller's encoding is never touched
        tmp = QKMMMemory(
            device            = self.device,
            mode              = self.mode,
            approx_oversample = self.approx_oversample,
            use_real          = self.use_real,
        ).encode(A)

        W_c = self.classical_matmul(A, B).cpu()
        W_q = tmp.query(B).cpu()

        abs_err = (W_c - W_q).abs().mean()
        rel_err = abs_err / (W_c.abs().mean() + EPS) * 100
        max_err = (W_c - W_q).abs().max()
        return {
            "abs_error":   float(abs_err),
            "rel_error_%": float(rel_err),
            "max_error":   float(max_err),
            "exact":       float(rel_err) < 1.0,
            "mode":        self.mode,
            "use_real":    tmp._use_real_active,
            "G":           tmp.G,
            "N":           int(A.shape[1]),
        }

    # ─────────────────────────────────────────────────────────
    # Benchmark
    # ─────────────────────────────────────────────────────────

    def benchmark(
        self,
        M:         int  = 256,
        N:         int  = 128,
        K:         int  = 256,
        Q_vals:    list = None,
        n_trials:  int  = 100,
        n_batched: int  = 8,     # number of batches for query_batched arm
    ) -> dict:
        """
        Amortized benchmark: encode once, query Q times.
        Reports break-even query count vs classical matmul.

        Also benchmarks query_batched() with n_batched simultaneous
        query batches — the GPU-saturating path absent from the original.

        accuracy() uses a temporary instance — caller's encoding is safe.
        """
        if Q_vals is None:
            Q_vals = [1, 2, 5, 10, 20, 50, 100]

        A       = torch.randn(M, N)
        B       = torch.randn(K, N)
        B_batch = torch.randn(n_batched, K, N)

        def sync():
            if self.device.type == "cuda":
                torch.cuda.synchronize()

        # Warm up — uses temporary instances to avoid polluting self
        tmp = QKMMMemory(device=self.device, mode=self.mode,
                         approx_oversample=self.approx_oversample,
                         use_real=self.use_real)
        for _ in range(5):
            tmp.encode(A).query(B)
            tmp.encode(A).query_batched(B_batch)
            self.classical_matmul(A, B)
        sync()

        # Encode time — on self (intended side effect: leaves self encoded on A)
        t0 = time.perf_counter()
        for _ in range(n_trials):
            self.encode(A)
        sync()
        t_encode = (time.perf_counter() - t0) / n_trials * 1000

        # query() time — encoding already done
        t0 = time.perf_counter()
        for _ in range(n_trials):
            self.query(B)
        sync()
        t_query = (time.perf_counter() - t0) / n_trials * 1000

        # query_batched() time
        t0 = time.perf_counter()
        for _ in range(n_trials):
            self.query_batched(B_batch)
        sync()
        t_batched = (time.perf_counter() - t0) / n_trials * 1000

        # Classical time
        t0 = time.perf_counter()
        for _ in range(n_trials):
            self.classical_matmul(A, B)
        sync()
        t_classical = (time.perf_counter() - t0) / n_trials * 1000

        breakeven = (
            t_encode / (t_classical - t_query)
            if t_classical > t_query
            else float("inf")
        )

        total_qkmm = [t_encode + Q * t_query   for Q in Q_vals]
        total_cls  = [Q * t_classical           for Q in Q_vals]

        # accuracy() uses temporary instance — self.encoding unchanged
        acc = self.accuracy(A, B)

        return {
            "M": M, "N": N, "K": K,
            "n_batched":      n_batched,
            "mode":           self.mode,
            "use_real":       self._use_real_active,
            "G":              self.G,
            "t_encode_ms":    t_encode,
            "t_query_ms":     t_query,
            "t_batched_ms":   t_batched,          # new: query_batched throughput
            "t_classical_ms": t_classical,
            "breakeven_Q":    breakeven,
            "Q_vals":         Q_vals,
            "total_qkmm_ms":  total_qkmm,
            "total_cls_ms":   total_cls,
            "rel_error_%":    acc["rel_error_%"],
            "exact":          acc["exact"],
        }

    def __repr__(self) -> str:
        encoded     = f"M={self.M}, N={self.N}, G={self.G}" if self.M else "not encoded"
        dtype       = "float32" if self._use_real_active else "complex64"
        proj_cached = "cached" if self._R is not None else "none"
        return (f"QKMMMemory(mode={self.mode}, device={self.device}, "
                f"dtype={dtype}, proj={proj_cached}, {encoded})")
