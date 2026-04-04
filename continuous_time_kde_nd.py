"""
ContinuousTimeKDE_nD
====================
N-dimensional GPU-accelerated time-varying KDE with full Madelung field
extraction, FFT-based Poisson phase recovery (approximate and exact), and
stochastic-thermodynamic flux decomposition.

Generalizes ContinuousTimeKDE to arbitrary feature dimension N.  Kernel
setup, causal-mode logic, and temporal-weight conventions are identical to the
2-D class so results are directly comparable.

Capabilities
------------
* Arbitrary N spatial / feature dimensions.
* All intermediate kernel parameters exposed: precision matrix, covariance
  matrix, Scott bandwidth factor, neff, normalised temporal weights.
* Diffusion tensor extracted from kernel geometry (get_diffusion).
* Two Poisson solvers for the phase field S:
    - Approximate  (default): ∇²S = -∂ρ/∂t / ρ          (spectral, O(G log G))
    - Exact (exact=True):     ∇·(ρ ∇S) = -∂ρ/∂t         (CG, iterative, ~10-50 its)
* Full spectral field operators exposed: gradient, Laplacian, divergence, curl proxy.
* Stochastic-thermodynamic flux decomposition: irreversible (housekeeping) vs
  reversible (osmotic / excess) components with per-component entropy production.
* Full Madelung field: (rho, S, u_osm, v_cur, Q, J, energies).

Practical grid limits (RTX 3060 Laptop, 6.4 GB VRAM)
------------------------------------------------------
N=1  : up to ~10 000 grid points
N=2  : up to 300 x 300 ≈ 90 000 grid points
N=3  : up to 60 x 60 x 60 ≈ 216 000 grid points (tight with exact solver)
N>=4 : sparse regime; prefer get_field at target points over full grid
"""

import math
import torch
import numpy as np
import pandas as pd


EPS = 1e-12


class ContinuousTimeKDE_nD:
    """
    N-dimensional time-varying kernel density estimator with Madelung field
    extraction, causal-mode support, and FFT / CG Poisson phase recovery.

    Parameters
    ----------
    df : pd.DataFrame
    feature_cols : list[str]
        The N spatial / feature column names.
    time_col : str
        Column containing datetime values.
    time_bandwidth_days : float
        Temporal Gaussian kernel bandwidth sigma_t in days.
    causal_mode : {'symmetric', 'strict', 'walk_forward'}
        symmetric    — past and future data (default)
        strict       — only past data at every query time
        walk_forward — symmetric inside training window, causal outside
    train_end_date : str or datetime, optional
        Required when causal_mode='walk_forward'.
    device : str, optional
        'cuda' or 'cpu'.  Auto-detects if None.
    """

    # ──────────────────────────────────────────────────────────────────
    # Construction
    # ──────────────────────────────────────────────────────────────────

    def __init__(
        self,
        df,
        feature_cols,
        time_col,
        time_bandwidth_days=30,
        causal_mode="symmetric",
        train_end_date=None,
        device=None,
    ):
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        print(f"ContinuousTimeKDE_nD initialised on: {self.device}")

        valid_modes = {"symmetric", "strict", "walk_forward"}
        if causal_mode not in valid_modes:
            raise ValueError(f"causal_mode must be one of {valid_modes}")
        if causal_mode == "walk_forward" and train_end_date is None:
            raise ValueError("train_end_date required for walk_forward mode")

        self.causal_mode = causal_mode
        self.feature_cols = list(feature_cols)
        self.n_dim = len(self.feature_cols)
        self.time_col = time_col
        self.sigma_t = float(time_bandwidth_days)

        cols = self.feature_cols + [time_col]
        self.data = df[cols].dropna().copy()
        self.start_date = self.data[time_col].min()
        self.data["time_numeric"] = (
            (self.data[time_col] - self.start_date).dt.total_seconds() / 86400.0
        )

        self.X = torch.tensor(
            self.data[self.feature_cols].values,
            dtype=torch.float32,
            device=self.device,
        )
        self.t = torch.tensor(
            self.data["time_numeric"].values,
            dtype=torch.float32,
            device=self.device,
        )

        self.train_end_date = None
        self.train_end_days = None
        if causal_mode == "walk_forward":
            self.train_end_date = pd.to_datetime(train_end_date)
            self.train_end_days = (self.train_end_date - self.start_date).days
            print(
                f"  Walk-forward: train <= {self.train_end_date.date()} "
                f"(symmetric), test uses causal kernel"
            )

        print(f"  N={self.n_dim} dimensions: {self.feature_cols}")
        print(f"  T={len(self.t):,} observations, sigma_t={self.sigma_t} days")

    # ──────────────────────────────────────────────────────────────────
    # Internal: temporal weights and kernel geometry
    # ──────────────────────────────────────────────────────────────────

    def _apply_causal_constraint(
        self, w: torch.Tensor, t_query: float
    ) -> torch.Tensor:
        """
        Apply causal mask according to self.causal_mode.

        All public methods route temporal weights through here, so causal
        behaviour is enforced uniformly including for the diffusion tensor
        and flux decomposition.
        """
        if self.causal_mode == "symmetric":
            return w
        if self.causal_mode == "strict":
            return w * (self.t <= t_query).float()
        # walk_forward
        if t_query > self.train_end_days:
            return w * (self.t <= t_query).float()
        return w

    def _to_days(self, query_date) -> float:
        if isinstance(query_date, str):
            query_date = pd.to_datetime(query_date)
        return (query_date - self.start_date).total_seconds() / 86400.0

    def _kernel_setup(self, t_query: float) -> dict:
        """
        Compute temporal weights and estimate the N x N kernel covariance.

        All tensors live on GPU.  Check result['valid'] before use.

        Returned keys
        -------------
        X_sub       (M, N)  active data points (after causal + threshold mask)
        t_sub       (M,)    active times
        w_raw       (M,)    raw (un-normalised) temporal weights
        w_norm      (M,)    normalised  sum(w) = 1
        neff        scalar  effective sample count  1 / sum(w^2)
        bandwidth   scalar  Scott factor  neff^{-1/(N+4)}
        cov         (N, N)  weighted spatial covariance of raw data
        cov_k       (N, N)  kernel covariance  = bandwidth^2 * cov
        precision   (N, N)  kernel precision  cov_k^{-1}  (via Cholesky)
        det         scalar  |cov_k|
        norm_const  float   (2*pi)^{-N/2} * |cov_k|^{-1/2}
        """
        w_raw = torch.exp(-0.5 * ((self.t - t_query) / self.sigma_t) ** 2)
        w_raw = self._apply_causal_constraint(w_raw, t_query)

        mask = w_raw > 1e-4
        if mask.sum() < self.n_dim + 2:
            return {"valid": False}

        X_sub = self.X[mask]
        t_sub = self.t[mask]
        w_raw_sub = w_raw[mask]
        w_norm = w_raw_sub / w_raw_sub.sum()

        neff = 1.0 / (w_norm ** 2).sum()

        mu = (w_norm.unsqueeze(1) * X_sub).sum(0)
        Xc = X_sub - mu.unsqueeze(0)
        denom = 1.0 - (w_norm ** 2).sum()
        cov = (Xc * w_norm.unsqueeze(1)).T @ Xc / denom

        bandwidth = neff ** (-1.0 / (self.n_dim + 4))
        cov_k = cov * (bandwidth ** 2)

        try:
            L = torch.linalg.cholesky(cov_k)
        except torch.linalg.LinAlgError:
            jitter = 1e-6 * torch.eye(self.n_dim, device=self.device)
            try:
                L = torch.linalg.cholesky(cov_k + jitter)
            except torch.linalg.LinAlgError:
                return {"valid": False}

        precision = torch.cholesky_inverse(L)
        det = torch.diagonal(L).prod() ** 2

        if det.item() <= 1e-30:
            return {"valid": False}

        norm_const = (2.0 * math.pi) ** (-self.n_dim / 2.0) * det.cpu().item() ** (-0.5)

        return {
            "valid": True,
            "X_sub": X_sub,
            "t_sub": t_sub,
            "w_raw": w_raw_sub,
            "w_norm": w_norm,
            "neff": neff,
            "bandwidth": bandwidth,
            "cov": cov,
            "cov_k": cov_k,
            "precision": precision,
            "det": det,
            "norm_const": norm_const,
        }

    # ──────────────────────────────────────────────────────────────────
    # Diffusion tensor
    # ──────────────────────────────────────────────────────────────────

    def get_diffusion(self, query_date) -> dict:
        """
        Extract the N x N diffusion tensor at a query time.

        Definition
        ----------
        D = Sigma_k / (2 * sigma_t)

        where Sigma_k is the kernel covariance and sigma_t is the temporal
        bandwidth in days.  This follows from identifying the Gaussian KDE
        kernel as the Green's function of the diffusion operator: in 1-D,
        bandwidth h at temporal scale sigma_t gives D = h^2 / (2 * sigma_t).
        The anisotropic generalisation replaces h^2 with the full kernel
        covariance matrix.

        The causal_mode is honoured: 'strict' and 'walk_forward' kernels
        use only past data when estimating Sigma_k.

        Returns
        -------
        dict
            D           (N, N) ndarray   full diffusion tensor
            D_diag      (N,)   ndarray   diagonal (per-dimension diffusivities)
            D_scalar    float            isotropic approximation  tr(D) / N
            cov_k       (N, N) ndarray   kernel covariance Sigma_k
            cov_data    (N, N) ndarray   weighted data covariance (before bandwidth scaling)
            precision   (N, N) ndarray   kernel precision
            neff        float            effective sample size
            bandwidth   float            Scott factor
            causal_mode str              mode applied
            valid       bool
        """
        t_query = self._to_days(query_date)
        k = self._kernel_setup(t_query)
        n = self.n_dim

        if not k["valid"]:
            return {
                "D": np.zeros((n, n)), "D_diag": np.zeros(n),
                "D_scalar": 0.0, "valid": False,
                "causal_mode": self.causal_mode,
            }

        cov_k_np = k["cov_k"].cpu().numpy()
        D = cov_k_np / (2.0 * self.sigma_t)

        return {
            "D": D,
            "D_diag": np.diag(D),
            "D_scalar": float(np.trace(D) / n),
            "cov_k": cov_k_np,
            "cov_data": k["cov"].cpu().numpy(),
            "precision": k["precision"].cpu().numpy(),
            "neff": k["neff"].cpu().item(),
            "bandwidth": k["bandwidth"].cpu().item(),
            "causal_mode": self.causal_mode,
            "valid": True,
        }

    # ──────────────────────────────────────────────────────────────────
    # Primary field evaluation at arbitrary points
    # ──────────────────────────────────────────────────────────────────

    def get_field(self, query_date, query_points, batch_size: int = 500) -> dict:
        """
        Evaluate the KDE density and its derivatives at arbitrary points in R^N.

        Parameters
        ----------
        query_date : str or datetime
        query_points : array-like, shape (G, N)
        batch_size : int
            Number of query points per GPU batch.

        Returns
        -------
        dict
            rho           (G,)    density rho, properly normalised
            grad_rho      (G, N)  gradient of rho
            osmotic_vel   (G, N)  u = grad(rho) / (2*rho)  [Nelson osmotic velocity]
            score         (G, N)  grad(log rho) = grad(rho) / rho  [Stein score]
            p_dot         (G,)    d(rho)/dt
            precision     (N, N)  kernel precision matrix
            covariance    (N, N)  kernel covariance matrix
            neff          float   effective sample size
            bandwidth     float   Scott factor
            w_norm        (M,)    normalised temporal weights over active subset
            valid         bool
        """
        t_query = self._to_days(query_date)
        k = self._kernel_setup(t_query)

        Q_arr = np.asarray(query_points)
        G = Q_arr.shape[0]
        zeros_G = np.zeros(G)
        zeros_GN = np.zeros((G, self.n_dim))

        if not k["valid"]:
            return {
                "rho": zeros_G, "grad_rho": zeros_GN,
                "osmotic_vel": zeros_GN, "score": zeros_GN,
                "p_dot": zeros_G, "valid": False,
            }

        X_sub = k["X_sub"]
        t_sub = k["t_sub"]
        w_norm = k["w_norm"]
        w_raw = k["w_raw"]
        precision = k["precision"]
        norm_const = k["norm_const"]

        Q = torch.tensor(Q_arr, dtype=torch.float32, device=self.device)

        rho_out = torch.zeros(G, device=self.device)
        grad_out = torch.zeros(G, self.n_dim, device=self.device)
        pdot_out = torch.zeros(G, device=self.device)

        # Time-derivative weight factor: d(log w_i)/dt = -(t_q - t_i) / sigma_t^2
        grad_w_factor = -(t_query - t_sub) / (self.sigma_t ** 2)
        norm_drift = (w_raw * grad_w_factor).sum() / w_raw.sum()

        for s in range(0, G, batch_size):
            e = min(s + batch_size, G)
            q = Q[s:e]                          # (B, N)

            # Build (B, M) distance without ever allocating (B, M, N)
            dist_sq = torch.zeros(e-s, len(w_norm), device=self.device)
            prec_dx_dims = []                   # store (B, M) slices for grad

            for _d in range(self.n_dim):
                dx_d = q[:, _d].unsqueeze(1) - X_sub[:, _d].unsqueeze(0)  # (B, M)
                # prec_dx[..., _d] = sum_j dx_j * precision[j, _d]
                pd_d = sum(
                    (q[:, _j].unsqueeze(1) - X_sub[:, _j].unsqueeze(0))
                    * precision[_j, _d]
                    for _j in range(self.n_dim)
                )
                dist_sq += dx_d * pd_d
                prec_dx_dims.append(pd_d)       # (B, M) — cheap
                del dx_d

            kern = torch.exp(-0.5 * dist_sq)
            del dist_sq
            wk = kern * w_norm.unsqueeze(0)
            del kern

            rho_chunk = wk.sum(1)
            rho_out[s:e] = rho_chunk

            for _d in range(self.n_dim):
                grad_out[s:e, _d] = -(wk * prec_dx_dims[_d]).sum(1)
            del prec_dx_dims

            term1 = (wk * grad_w_factor.unsqueeze(0)).sum(1)
            pdot_out[s:e] = term1 - rho_chunk * norm_drift
            del wk, rho_chunk, term1
            torch.cuda.empty_cache()

        rho_np = rho_out.cpu().numpy() * norm_const
        grad_np = grad_out.cpu().numpy() * norm_const
        pdot_np = pdot_out.cpu().numpy() * norm_const

        osmotic_np = grad_np / (2.0 * (rho_np[:, None] + EPS))
        score_np = grad_np / (rho_np[:, None] + EPS)

        low = rho_np < 1e-10
        grad_np[low] = 0.0
        osmotic_np[low] = 0.0
        score_np[low] = 0.0
        pdot_np[low] = 0.0

        return {
            "rho": rho_np,
            "grad_rho": grad_np,
            "osmotic_vel": osmotic_np,
            "score": score_np,
            "p_dot": pdot_np,
            "precision": k["precision"].cpu().numpy(),
            "covariance": k["cov_k"].cpu().numpy(),
            "neff": k["neff"].cpu().item(),
            "bandwidth": k["bandwidth"].cpu().item(),
            "w_norm": k["w_norm"].cpu().numpy(),
            "valid": True,
        }

    # ──────────────────────────────────────────────────────────────────
    # Spectral field operators (GPU, N-dimensional)
    # ──────────────────────────────────────────────────────────────────

    def _build_k_vecs(self, grid_shapes: list, dx: list):
        """
        Build broadcast-ready frequency vectors for each spatial dimension.

        Returns
        -------
        k_vecs : list of N tensors, k_vecs[i] has shape (1,...,G_i,...,1)
        k_sq   : (G1,...,GN) tensor  |k|^2 = sum_i k_i^2
        """
        N = self.n_dim
        k_vecs = []
        k_sq = torch.zeros(grid_shapes, device=self.device)
        for i, (n_pts, d) in enumerate(zip(grid_shapes, dx)):
            k_1d = torch.fft.fftfreq(n_pts, d=d, device=self.device) * 2.0 * math.pi
            shape = [1] * N
            shape[i] = n_pts
            k_1d_v = k_1d.view(shape)
            k_vecs.append(k_1d_v)
            k_sq = k_sq + k_1d_v ** 2
        return k_vecs, k_sq

    def spectral_gradient(
        self, f_t: torch.Tensor, k_vecs: list
    ) -> torch.Tensor:
        """
        N-D spectral gradient of a scalar field.

        Parameters
        ----------
        f_t    : (G1,...,GN) GPU tensor
        k_vecs : from _build_k_vecs

        Returns
        -------
        (N, G1,...,GN) GPU tensor — component i is df/dx_i
        """
        f_hat = torch.fft.fftn(f_t)
        return torch.stack(
            [torch.fft.ifftn(1j * k_v * f_hat).real for k_v in k_vecs], dim=0
        )

    def spectral_laplacian(
        self, f_t: torch.Tensor, k_sq: torch.Tensor
    ) -> torch.Tensor:
        """
        N-D spectral Laplacian  nabla^2 f.

        Parameters
        ----------
        f_t  : (G1,...,GN) GPU tensor
        k_sq : (G1,...,GN) GPU tensor  |k|^2

        Returns
        -------
        (G1,...,GN) GPU tensor
        """
        return torch.fft.ifftn(-k_sq * torch.fft.fftn(f_t)).real

    def spectral_divergence(
        self, F_t: torch.Tensor, k_vecs: list
    ) -> torch.Tensor:
        """
        N-D spectral divergence of a vector field F.

        Parameters
        ----------
        F_t    : (N, G1,...,GN) GPU tensor
        k_vecs : from _build_k_vecs

        Returns
        -------
        (G1,...,GN) GPU tensor   nabla . F = sum_i dF_i/dx_i
        """
        div = torch.zeros(F_t.shape[1:], device=self.device)
        for i, k_v in enumerate(k_vecs):
            div = div + torch.fft.ifftn(1j * k_v * torch.fft.fftn(F_t[i])).real
        return div

    def spectral_curl_proxy(
        self, F_t: torch.Tensor, k_vecs: list
    ) -> torch.Tensor:
        """
        N-D anti-symmetric part of the Jacobian — proxy for rotation /
        irrotational vs solenoidal decomposition.

        For N=2 returns the scalar vorticity:
            omega = dF_y/dx - dF_x/dy                  shape: (G1, G2)

        For N>2 returns the full skew-symmetric Jacobian tensor Omega_{ij}:
            Omega_{ij} = (dF_j/dx_i - dF_i/dx_j) / 2  shape: (N, N, G1,...,GN)

        Parameters
        ----------
        F_t    : (N, G1,...,GN) GPU tensor
        k_vecs : from _build_k_vecs
        """
        N = self.n_dim
        G_shape = F_t.shape[1:]

        # jac[i, j] = dF_j / dx_i
        jac = torch.zeros((N, N) + G_shape, device=self.device)
        for j in range(N):
            F_j_hat = torch.fft.fftn(F_t[j])
            for i, k_v in enumerate(k_vecs):
                jac[i, j] = torch.fft.ifftn(1j * k_v * F_j_hat).real

        omega = (jac - jac.transpose(0, 1)) * 0.5

        if N == 2:
            return omega[0, 1]   # scalar curl
        return omega

    # ──────────────────────────────────────────────────────────────────
    # Poisson solvers for S
    # ──────────────────────────────────────────────────────────────────

    def _poisson_matvec(
        self,
        S_flat: torch.Tensor,
        rho_t: torch.Tensor,
        k_vecs: list,
        G_shape: tuple,
    ) -> torch.Tensor:
        """
        Matrix-free matvec for the exact Poisson operator  A(S) = -nabla.(rho nabla S).
        Used inside CG.  All tensors on GPU.
        """
        S_t = S_flat.reshape(G_shape)
        grad_S = self.spectral_gradient(S_t, k_vecs)       # (N, *G_shape)
        rho_grad_S = rho_t.unsqueeze(0) * grad_S           # (N, *G_shape)
        return (-self.spectral_divergence(rho_grad_S, k_vecs)).reshape(-1)

    def _solve_poisson_cg(
        self,
        rho_t: torch.Tensor,
        pdot_t: torch.Tensor,
        k_vecs: list,
        k_sq: torch.Tensor,
        G_shape: tuple,
        max_iter: int = 100,
        tol: float = 1e-5,
    ) -> tuple:
        """
        Preconditioned conjugate-gradient solve of the exact equation:

            -nabla . (rho nabla S) = d(rho)/dt

        Preconditioner: spectral inverse of the constant-coefficient Laplacian
            P^{-1}(r) = F^{-1}[ F[r] / |k|^2 ]   (DC mode zeroed)

        This is equivalent to solving the approximate equation as a warm start
        and then correcting for the density weighting via CG.  Convergence is
        typically 10-50 iterations when rho is smooth.

        Returns
        -------
        S_t  : (G1,...,GN) GPU tensor  (zero-mean gauge)
        info : dict with 'n_iter', 'rel_residual', 'converged'
        """
        G_total = int(np.prod(G_shape))

        b = pdot_t.reshape(-1).to(torch.float64)

        k_sq_f64 = k_sq.to(torch.float64)
        k_sq_safe = k_sq_f64.clone()
        k_sq_safe[k_sq == 0.0] = 1.0

        def precond(r):
            r_nd = r.to(torch.float32).reshape(G_shape)
            z_hat = torch.fft.fftn(r_nd) / k_sq_safe.to(torch.float32)
            z_hat.reshape(-1)[0] = 0.0
            return torch.fft.ifftn(z_hat).real.reshape(-1).to(torch.float64)

        def matvec(s):
            return self._poisson_matvec(
                s.to(torch.float32), rho_t, k_vecs, G_shape
            ).to(torch.float64)

        x = torch.zeros(G_total, dtype=torch.float64, device=self.device)
        r = b - matvec(x)
        z = precond(r)
        p = z.clone()
        rz = (r * z).sum()
        b_norm = b.norm()

        info = {"n_iter": 0, "rel_residual": float("inf"), "converged": False}

        for it in range(max_iter):
            Ap = matvec(p)
            alpha = rz / ((p * Ap).sum() + EPS)
            x = x + alpha * p
            r = r - alpha * Ap

            rel_res = r.norm() / (b_norm + EPS)
            if rel_res.item() < tol:
                info = {"n_iter": it + 1, "rel_residual": rel_res.item(), "converged": True}
                break

            z = precond(r)
            rz_new = (r * z).sum()
            beta = rz_new / (rz + EPS)
            p = z + beta * p
            rz = rz_new

        if not info["converged"]:
            info = {
                "n_iter": max_iter,
                "rel_residual": (r.norm() / (b_norm + EPS)).item(),
                "converged": False,
            }

        S = x.to(torch.float32).reshape(G_shape)
        S = S - S.mean()   # gauge: mean S = 0
        return S, info

    def solve_phase_poisson(
        self,
        query_date,
        grid_linspaces: list,
        exact: bool = False,
        cg_max_iter: int = 100,
        cg_tol: float = 1e-5,
        batch_size: int = 500,
    ) -> dict:
        """
        Recover the phase field S on a regular N-D grid via Poisson solve.

        Two solver options
        ------------------
        exact=False  (default, spectral O(G log G))
            Approximate equation:

                nabla^2 S = -d(rho)/dt / rho

            Obtained by expanding nabla.(rho nabla S) and dropping the
            nabla(log rho) . nabla(S) coupling.  Valid inside the bulk of
            the density support where rho is locally smooth.

        exact=True   (preconditioned CG, iterative)
            Exact variable-coefficient equation:

                nabla . (rho nabla S) = -d(rho)/dt

            Use this for quantum experiments where accuracy in S matters.
            The preconditioner is the spectral solve, so warm-start
            convergence is typically 10-50 iterations.

        Parameters
        ----------
        query_date : str or datetime
        grid_linspaces : list[np.ndarray]
            N uniform linspace arrays, one per dimension.
            E.g. for 2-D: [np.linspace(xmin, xmax, 100), np.linspace(ymin, ymax, 100)]
        exact : bool
        cg_max_iter : int
        cg_tol : float
        batch_size : int

        Returns
        -------
        dict
            S           (*G_shape)      phase field (zero-mean gauge)
            rho         (*G_shape)      density
            osmotic_vel (N, *G_shape)   u = grad(rho) / (2*rho)
            p_dot       (*G_shape)      d(rho)/dt
            rhs         (*G_shape)      Poisson RHS: -p_dot / rho
            grid_coords list[ndarray]   meshgrid output (indexing='ij')
            grid_linspaces              echo of input
            solver      str             'spectral' or 'cg'
            cg_info     dict            CG diagnostics (exact=True only)
            valid       bool
        """
        N = self.n_dim
        grid_shapes = [len(g) for g in grid_linspaces]
        dx = [float(g[1] - g[0]) for g in grid_linspaces]

        grids = np.meshgrid(*grid_linspaces, indexing="ij")
        flat_coords = np.stack([g.ravel() for g in grids], axis=1)

        field = self.get_field(query_date, flat_coords, batch_size=batch_size)
        if not field["valid"]:
            return {"valid": False}

        G_shape = tuple(grid_shapes)
        rho = field["rho"].reshape(G_shape)
        pdot = field["p_dot"].reshape(G_shape)

        rho_t = torch.tensor(rho, dtype=torch.float32, device=self.device)
        pdot_t = torch.tensor(pdot, dtype=torch.float32, device=self.device)

        k_vecs, k_sq = self._build_k_vecs(grid_shapes, dx)

        cg_info = {}

        if exact:
            S_t, cg_info = self._solve_poisson_cg(
                rho_t, pdot_t, k_vecs, k_sq, G_shape,
                max_iter=cg_max_iter, tol=cg_tol,
            )
            solver_label = "cg"
        else:
            rhs_np = -pdot / (rho + EPS)
            rhs_t = torch.tensor(rhs_np, dtype=torch.float32, device=self.device)
            k_sq_safe = k_sq.clone()
            k_sq_safe[k_sq == 0.0] = 1.0
            S_hat = -torch.fft.fftn(rhs_t) / k_sq_safe
            S_hat.reshape(-1)[0] = 0.0
            S_t = torch.fft.ifftn(S_hat).real
            solver_label = "spectral"

        S = S_t.cpu().numpy()
        rhs_np = -pdot / (rho + EPS)

        osm = field["osmotic_vel"].reshape(G_shape + (N,))
        osmotic_reshaped = np.moveaxis(osm, -1, 0)   # (N, *G_shape)

        return {
            "S": S,
            "rho": rho,
            "osmotic_vel": osmotic_reshaped,
            "p_dot": pdot,
            "rhs": rhs_np,
            "grid_coords": grids,
            "grid_linspaces": grid_linspaces,
            "solver": solver_label,
            "cg_info": cg_info,
            "valid": True,
        }

    # ──────────────────────────────────────────────────────────────────
    # Full Madelung decomposition
    # ──────────────────────────────────────────────────────────────────

    def get_madelung_field(
        self,
        query_date,
        grid_linspaces: list,
        exact: bool = False,
        cg_max_iter: int = 100,
        cg_tol: float = 1e-5,
        batch_size: int = 500,
    ) -> dict:
        """
        Full (rho, S) Madelung decomposition plus all derived quantities on a
        regular N-D grid.

        Extends solve_phase_poisson with:

            current_vel     (N, *G_shape)  v = nabla S  (phase-driven flow)
            quantum_pot     (*G_shape)     Q = -nabla^2(sqrt(rho)) / (2*sqrt(rho))
            flux            (N, *G_shape)  J = rho * v
            divergence_flux (*G_shape)     nabla . J  (continuity check: should ~ -d(rho)/dt)
            osmotic_energy  float          integral rho |u|^2 dV
            current_energy  float          integral rho |v|^2 dV
            quantum_energy  float          integral rho Q dV  (Fisher / Bohm information)

        All parameters are passed through to solve_phase_poisson.
        """
        result = self.solve_phase_poisson(
            query_date, grid_linspaces, exact=exact,
            cg_max_iter=cg_max_iter, cg_tol=cg_tol, batch_size=batch_size,
        )
        if not result["valid"]:
            return result

        S = result["S"]
        rho = result["rho"]
        lspaces = result["grid_linspaces"]
        N = self.n_dim
        G_shape = rho.shape
        dx_list = [float(g[1] - g[0]) for g in lspaces]
        grid_shapes = [len(g) for g in lspaces]

        S_t = torch.tensor(S, dtype=torch.float32, device=self.device)
        rho_t = torch.tensor(rho, dtype=torch.float32, device=self.device)

        k_vecs, k_sq = self._build_k_vecs(grid_shapes, dx_list)

        # Current velocity v = nabla S
        current_vel_t = self.spectral_gradient(S_t, k_vecs)   # (N, *G_shape)
        current_vel_np = current_vel_t.cpu().numpy()

        # Quantum potential Q = -nabla^2(sqrt(rho)) / (2*sqrt(rho))
        sqrt_rho = torch.sqrt(rho_t + EPS)
        lap_sqrt = self.spectral_laplacian(sqrt_rho, k_sq)
        quantum_pot = (-lap_sqrt / (2.0 * (sqrt_rho + EPS))).cpu().numpy()

        # Probability flux J = rho * v
        flux = np.stack([rho * current_vel_np[i] for i in range(N)], axis=0)

        # Divergence of J (continuity check)
        flux_t = torch.tensor(flux, dtype=torch.float32, device=self.device)
        div_flux = self.spectral_divergence(flux_t, k_vecs).cpu().numpy()

        # Integrated energies  integral rho |f|^2 dV
        dV = float(np.prod(dx_list))
        osm = result["osmotic_vel"]   # (N, *G_shape)
        osm_sq = sum(osm[i] ** 2 for i in range(N))
        cur_sq = sum(current_vel_np[i] ** 2 for i in range(N))

        osmotic_energy = float(np.sum(rho * osm_sq) * dV)
        current_energy = float(np.sum(rho * cur_sq) * dV)
        quantum_energy = float(np.sum(rho * quantum_pot) * dV)

        result.update({
            "current_vel": current_vel_np,
            "quantum_pot": quantum_pot,
            "flux": flux,
            "divergence_flux": div_flux,
            "osmotic_energy": osmotic_energy,
            "current_energy": current_energy,
            "quantum_energy": quantum_energy,
        })
        return result

    # ──────────────────────────────────────────────────────────────────
    # Stochastic-thermodynamic flux decomposition
    # ──────────────────────────────────────────────────────────────────

    def get_flux_decomposition(
        self,
        query_date,
        grid_linspaces: list,
        exact: bool = False,
        cg_max_iter: int = 100,
        cg_tol: float = 1e-5,
        batch_size: int = 500,
        verbose: bool = True,
    ) -> dict:
        """
        Decompose the probability current into irreversible (housekeeping) and
        reversible (osmotic / excess) components.

        Follows Ito's housekeeping-excess entropy production split:
        Ito, S. (2018). Stochastic thermodynamic interpretation of information
        geometry. PRL, 121(3), 030605.

        Definitions
        -----------
        Probability current J recovered from the continuity equation:

            d(rho)/dt + nabla . J = 0

        via Poisson solve on d(rho)/dt.  The current velocity is:

            v = J / rho

        Decomposition:

            v_osm   = D . nabla(log rho)     [osmotic / reversible]
            v_irrev = v - v_osm              [housekeeping / irreversible]

        where D is the N x N diffusion tensor from get_diffusion().

        Entropy production rates:

            sigma_irrev = integral rho * v_irrev^T D^{-1} v_irrev dV
            sigma_rev   = integral rho * v_osm^T D^{-1} v_osm dV

        The causal_mode is respected for both the KDE field and the diffusion
        tensor.  A strict or walk_forward kernel only uses past observations.

        Parameters
        ----------
        exact : bool
            Pass True for the exact CG Poisson solve (recommended for quantum
            systems where J accuracy matters).
        verbose : bool
            Print diffusion tensor and CG diagnostics.

        Returns
        -------
        dict
            rho         (*G_shape)    density
            p_dot       (*G_shape)    d(rho)/dt
            J           (N, *G_shape) probability current
            v_total     (N, *G_shape) total current velocity J / rho
            v_osm       (N, *G_shape) osmotic (reversible) component
            v_irrev     (N, *G_shape) irreversible (housekeeping) component
            sigma_irrev float         housekeeping entropy production rate
            sigma_rev   float         excess (osmotic) entropy production rate
            sigma_total float         total entropy production
            irrev_ratio float         sigma_irrev / sigma_total
            D_info      dict          from get_diffusion()
            S           (*G_shape)    phase field
            grid_coords list[ndarray]
            solver      str
            cg_info     dict
            valid       bool
        """
        N = self.n_dim
        grid_shapes = [len(g) for g in grid_linspaces]
        dx_list = [float(g[1] - g[0]) for g in grid_linspaces]

        # Diffusion tensor — causal_mode respected via _kernel_setup
        D_info = self.get_diffusion(query_date)
        if not D_info["valid"]:
            return {"valid": False}

        D_np = D_info["D"]   # (N, N)

        if verbose:
            print(f"Diffusion tensor D [tr/N = {D_info['D_scalar']:.4g}], "
                  f"causal_mode='{self.causal_mode}':")
            print(np.array2string(D_np, precision=4))
            print(f"  neff={D_info['neff']:.1f}, bandwidth={D_info['bandwidth']:.4g}")

        # Full Madelung field
        result = self.get_madelung_field(
            query_date, grid_linspaces, exact=exact,
            cg_max_iter=cg_max_iter, cg_tol=cg_tol, batch_size=batch_size,
        )
        if not result["valid"]:
            return {"valid": False}

        if verbose and exact:
            ci = result.get("cg_info", {})
            print(f"  CG: {ci.get('n_iter')} iters, "
                  f"rel_res={ci.get('rel_residual', float('nan')):.2e}, "
                  f"converged={ci.get('converged')}")

        G_shape = tuple(grid_shapes)
        rho = result["rho"]
        flux = result["flux"]          # (N, *G_shape)  J = rho * v_current

        # Score = nabla log rho  (osmotic_vel = score / 2)
        score = result["osmotic_vel"] * 2.0   # (N, *G_shape)

        # Osmotic velocity: v_osm_i = sum_j D_{ij} score_j
        D_t = torch.tensor(D_np, dtype=torch.float32, device=self.device)
        score_t = torch.tensor(score, dtype=torch.float32, device=self.device)
        score_flat = score_t.reshape(N, -1)        # (N, G_total)
        v_osm_flat = D_t @ score_flat              # (N, G_total)
        v_osm = v_osm_flat.cpu().numpy().reshape((N,) + G_shape)

        # Total current velocity
        mask = rho > 1e-10
        v_total = np.zeros_like(flux)
        for i in range(N):
            v_total[i][mask] = flux[i][mask] / rho[mask]

        # Irreversible component
        v_irrev = v_total - v_osm
        for i in range(N):
            v_irrev[i][~mask] = 0.0
            v_osm[i][~mask] = 0.0

        # Entropy production: sigma = integral rho v^T D^{-1} v dV
        L_D = torch.linalg.cholesky(D_t + 1e-12 * torch.eye(N, device=self.device))
        D_inv = torch.cholesky_inverse(L_D).cpu().numpy()   # (N, N)
        dV = float(np.prod(dx_list))

        def ep_rate(v_field, rho_field):
            """integral rho * v^T D^{-1} v dV"""
            vf = v_field.reshape(N, -1)           # (N, G_total)
            Dv = D_inv @ vf                        # (N, G_total)
            vDv = (vf * Dv).sum(axis=0)            # (G_total,)
            return float(np.sum(rho_field.reshape(-1) * vDv) * dV)

        sigma_irrev = ep_rate(v_irrev, rho)
        sigma_rev = ep_rate(v_osm, rho)
        sigma_total = sigma_irrev + sigma_rev

        return {
            "rho": rho,
            "p_dot": result["p_dot"],
            "J": flux,
            "v_total": v_total,
            "v_osm": v_osm,
            "v_irrev": v_irrev,
            "sigma_irrev": sigma_irrev,
            "sigma_rev": sigma_rev,
            "sigma_total": sigma_total,
            "irrev_ratio": sigma_irrev / (sigma_total + EPS),
            "D_info": D_info,
            "S": result["S"],
            "grid_coords": result["grid_coords"],
            "grid_linspaces": grid_linspaces,
            "solver": result["solver"],
            "cg_info": result.get("cg_info", {}),
            "valid": True,
        }

    # ──────────────────────────────────────────────────────────────────
    # Convenience wrappers
    # ──────────────────────────────────────────────────────────────────

    def grid_field_2d(
        self, query_date, linspace_0, linspace_1, batch_size=500
    ) -> dict:
        """
        Convenience wrapper for n_dim=2 that returns 2-D shaped arrays
        matching the legacy ContinuousTimeKDE interface
        (grid_x, grid_y, Z, U, V, P_dot).
        """
        if self.n_dim != 2:
            raise ValueError("grid_field_2d is only valid for n_dim=2")

        g0, g1 = np.meshgrid(linspace_0, linspace_1, indexing="ij")
        pts = np.stack([g0.ravel(), g1.ravel()], axis=1)

        f = self.get_field(query_date, pts, batch_size=batch_size)
        if not f["valid"]:
            zeros = np.zeros_like(g0)
            return {
                "grid_x": g0, "grid_y": g1,
                "Z": zeros, "U": zeros, "V": zeros, "P_dot": zeros,
                "valid": False,
            }

        sh = g0.shape
        Z = f["rho"].reshape(sh)
        U = f["osmotic_vel"][:, 0].reshape(sh)
        V = f["osmotic_vel"][:, 1].reshape(sh)
        P_dot = f["p_dot"].reshape(sh)

        return {
            "grid_x": g0, "grid_y": g1,
            "Z": Z, "U": U, "V": V, "P_dot": P_dot,
            "precision": f["precision"],
            "covariance": f["covariance"],
            "neff": f["neff"],
            "bandwidth": f["bandwidth"],
            "valid": True,
        }

    # ──────────────────────────────────────────────────────────────────
    # Tensor refresh
    # ──────────────────────────────────────────────────────────────────

    def _update_tensors(self):
        """Rebuild GPU tensors from self.data — call after filtering self.data."""
        self.X = torch.tensor(
            self.data[self.feature_cols].values,
            dtype=torch.float32,
            device=self.device,
        )
        self.t = torch.tensor(
            self.data["time_numeric"].values,
            dtype=torch.float32,
            device=self.device,
        )
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
