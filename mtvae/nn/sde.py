"""Latent neural SDE with a path-wise KL term.
Euler-Maruyama accumulates ``0.5 * int ||(f_post - f_prior) / g||^2 dt``; the prior drift is 0."""
from __future__ import annotations

from typing import Optional, Tuple

import math
import torch
from torch import nn


# ---------------------------------------------------------------------------
# Time embedding
# ---------------------------------------------------------------------------


class TimeFourier(nn.Module):
    """Small learned Fourier embedding of scalar time ``t in [0, 1]``.

    Computes ``proj(sin(W t + b))``. Given a scalar ``t``, an ``(N,)`` or an
    ``(N, 1)`` tensor, returns an ``(N, out_dim)`` embedding.

    Parameters
    ----------
    out_dim : int, default 4
        Embedding dimension.
    hidden_dim : int, default 10
        Number of Fourier features.
    """

    def __init__(self, out_dim: int = 4, hidden_dim: int = 10) -> None:
        super().__init__()
        self.out_dim = out_dim
        # Learnable Fourier features: y = sin(W t + b)
        self.W = nn.Parameter(torch.randn(hidden_dim) * 2.0 * math.pi)
        self.b = nn.Parameter(torch.zeros(hidden_dim))
        self.proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Embed ``t`` of shape ``(N,)`` or ``(N, 1)`` to ``(N, out_dim)``."""
        if t.dim() == 0:
            t = t.unsqueeze(0)
        if t.dim() == 1:
            t = t.unsqueeze(-1)  # (N, 1)
        # (N, hidden_dim)
        phase = t * self.W[None, :] + self.b[None, :]
        features = torch.sin(phase)
        return self.proj(features)


def _broadcast_time(t: float | torch.Tensor, batch: int, device: torch.device,
                    dtype: torch.dtype) -> torch.Tensor:
    """Turn a Python float or 0/1-D tensor ``t`` into a ``(batch, 1)`` tensor."""
    if isinstance(t, torch.Tensor):
        if t.dim() == 0:
            t = t.unsqueeze(0)
        if t.shape[0] == 1 and batch > 1:
            t = t.expand(batch)
        return t.to(device=device, dtype=dtype).reshape(-1, 1)
    return torch.full((batch, 1), float(t), device=device, dtype=dtype)


# ---------------------------------------------------------------------------
# Drift networks
# ---------------------------------------------------------------------------


class Drift(nn.Module):
    """Learned drift network ``f(z, t)``.

    An MLP on ``[z, time_embedding(t)]``. The output layer is
    zero-initialised, so the drift is exactly zero at initialisation and
    training starts from no transport.

    Parameters
    ----------
    latent_dim : int
        State dimension ``D``.
    hidden : tuple of int, default (128, 128)
        Hidden widths.
    time_embed_dim : int, default 4
        Output dimension of the :class:`TimeFourier` embedding.
    time_hidden : int, default 10
        Number of Fourier features.
    """

    def __init__(self, latent_dim: int, hidden: Tuple[int, ...] = (128, 128),
                 time_embed_dim: int = 4, time_hidden: int = 10) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.time_embed = TimeFourier(out_dim=time_embed_dim, hidden_dim=time_hidden)
        layers: list[nn.Module] = []
        prev = latent_dim + time_embed_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.SiLU())
            prev = h
        out = nn.Linear(prev, latent_dim)
        # Zero-init output so f = 0 at step 0.
        nn.init.zeros_(out.weight)
        nn.init.zeros_(out.bias)
        layers.append(out)
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor, t: float | torch.Tensor) -> torch.Tensor:
        """Drift at states ``z`` of shape ``(N, D)`` and time ``t``."""
        t_vec = _broadcast_time(t, z.shape[0], z.device, z.dtype)
        t_embed = self.time_embed(t_vec)          # (N, time_embed_dim)
        h = torch.cat([z, t_embed], dim=-1)       # (N, D + T)
        return self.net(h)                        # (N, D)


class ZeroDrift(nn.Module):
    """Constant-zero drift; the prior drift for ``prior_mode="zero"``."""

    def forward(self, z: torch.Tensor, t: float | torch.Tensor) -> torch.Tensor:
        """Return zeros shaped like ``z``."""
        return torch.zeros_like(z)


class DriftTimeOnly(nn.Module):
    """State-independent drift ``f(t)`` — only time is an input.

    The prior drift for ``prior_mode="learned_t_only"``. It is a curve in
    ``R^D`` parameterised by ``t`` (a population-mean flow at each time), so
    it cannot track the state-dependent ``f_post(z, t)`` and the path KL stays
    informative. The output layer is zero-initialised.

    Parameters
    ----------
    latent_dim : int
        State dimension ``D``.
    hidden : tuple of int, default (64,)
        Hidden widths.
    time_embed_dim : int, default 4
        Output dimension of the :class:`TimeFourier` embedding.
    time_hidden : int, default 10
        Number of Fourier features.
    """

    def __init__(self, latent_dim: int, hidden: Tuple[int, ...] = (64,),
                 time_embed_dim: int = 4, time_hidden: int = 10) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.time_embed = TimeFourier(out_dim=time_embed_dim, hidden_dim=time_hidden)
        layers: list[nn.Module] = []
        prev = time_embed_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.SiLU())
            prev = h
        out = nn.Linear(prev, latent_dim)
        nn.init.zeros_(out.weight)
        nn.init.zeros_(out.bias)
        layers.append(out)
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor, t: float | torch.Tensor) -> torch.Tensor:
        """Drift at time ``t``, shape ``(N, D)``; ``z`` sets only the batch size."""
        t_vec = _broadcast_time(t, z.shape[0], z.device, z.dtype)
        t_embed = self.time_embed(t_vec)            # (N, time_embed_dim)
        # State-independent — every cell at the same t gets the same drift.
        return self.net(t_embed)                    # (N, D)


# ---------------------------------------------------------------------------
# Diffusion networks
# ---------------------------------------------------------------------------


class Diffusion(nn.Module):
    """Diffusion network ``g(z, t) > 0``; depends on time only.

    Modes:

    - ``"scalar"``: one time-varying ``sigma(t)``, broadcast across all
      dimensions (the default);
    - ``"per_dim"``: a time-varying ``sigma(t)`` per latent dimension;
    - ``"fixed"``: a constant, non-trainable ``init_sigma``.

    The learned modes pass an MLP of the time embedding through softplus.
    The output weights are zero-initialised and the output bias is set to
    ``softplus^{-1}(init_sigma)``, so ``g = init_sigma`` at initialisation.

    Parameters
    ----------
    latent_dim : int
        State dimension ``D``.
    mode : {"scalar", "per_dim", "fixed"}, default "scalar"
        See above.
    time_embed_dim : int, default 4
        Output dimension of the :class:`TimeFourier` embedding.
    time_hidden : int, default 10
        Number of Fourier features.
    hidden : tuple of int, default (32,)
        Hidden widths.
    init_sigma : float, default 0.02
        Diffusion at initialisation (throughout for ``"fixed"``).
    trainable : bool, default True
        If False, the parameters of a learned mode are frozen.

    Raises
    ------
    ValueError
        If ``mode`` is unknown.
    """

    def __init__(self, latent_dim: int, mode: str = "scalar",
                 time_embed_dim: int = 4, time_hidden: int = 10,
                 hidden: Tuple[int, ...] = (32,), init_sigma: float = 0.02,
                 trainable: bool = True) -> None:
        super().__init__()
        if mode not in ("scalar", "per_dim", "fixed"):
            raise ValueError(f"Unknown diffusion mode: {mode!r}")
        self.latent_dim = latent_dim
        self.mode = mode
        self.trainable = trainable

        if mode == "fixed":
            # Non-trainable constant sigma.
            self.register_buffer("sigma_const", torch.tensor(float(init_sigma)))
            return

        self.time_embed = TimeFourier(out_dim=time_embed_dim, hidden_dim=time_hidden)
        out_dim = 1 if mode == "scalar" else latent_dim
        layers: list[nn.Module] = []
        prev = time_embed_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.SiLU())
            prev = h
        out = nn.Linear(prev, out_dim)
        # With zero output weights, softplus(b) = init_sigma for
        # b = log(exp(init_sigma) - 1).
        target_bias = math.log(math.expm1(init_sigma))
        nn.init.zeros_(out.weight)
        nn.init.constant_(out.bias, target_bias)
        layers.append(out)
        self.net = nn.Sequential(*layers)
        if not trainable:
            for p in self.parameters():
                p.requires_grad = False

    def forward(self, z: torch.Tensor, t: float | torch.Tensor) -> torch.Tensor:
        """Diffusion at time ``t``, shape ``(N, D)``; ``z`` sets the batch size."""
        if self.mode == "fixed":
            return self.sigma_const.expand(z.shape[0], self.latent_dim)
        t_vec = _broadcast_time(t, z.shape[0], z.device, z.dtype)
        t_embed = self.time_embed(t_vec)          # (N, T)
        raw = self.net(t_embed)                   # (N, 1) or (N, D)
        sigma = torch.nn.functional.softplus(raw)
        if self.mode == "scalar":
            sigma = sigma.expand(-1, self.latent_dim)  # broadcast to (N, D)
        return sigma


# ---------------------------------------------------------------------------
# Euler-Maruyama integrator with KL augmentation
# ---------------------------------------------------------------------------


def euler_maruyama_kl(f_prior: Optional[nn.Module], f_post: nn.Module,
                      g_net: nn.Module, z0: torch.Tensor, ts: torch.Tensor,
                      dt_step: float = 0.01,
                      generator: Optional[torch.Generator] = None,
                      ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Integrate ``dz = f_post dt + g dW`` and accumulate the path KL.

    Parameters
    ----------
    f_prior : torch.nn.Module or None
        Prior drift network with signature ``f(z, t) -> (N, D)``. ``None``
        means zero prior drift.
    f_post : torch.nn.Module
        Posterior drift network.
    g_net : torch.nn.Module
        Diffusion network returning a positive ``sigma`` of shape ``(N, D)``.
    z0 : torch.Tensor
        Initial state, shape ``(N, D)``.
    ts : torch.Tensor
        1-D increasing tensor of stage times (normalised to ``[0, 1]``). The
        trajectory has one snapshot at each ``ts[i]``. A non-positive
        interval is skipped (the state is carried over unchanged).
    dt_step : float, default 0.01
        Target Euler step size. Each interval takes
        ``max(1, round((ts[i+1] - ts[i]) / dt_step))`` equal steps.
    generator : torch.Generator, optional
        Generator for the Brownian increments; the global torch RNG is used
        when omitted.

    Returns
    -------
    traj : torch.Tensor
        Shape ``(T, N, D)`` where ``T = len(ts)``; ``traj[0]`` is ``z0``.
    kl : torch.Tensor
        Shape ``(T, N)``: the path KL accumulated up to each stage time;
        ``kl[0]`` is zero.

    Raises
    ------
    ValueError
        If ``ts`` is not 1-D.
    """
    if ts.dim() != 1:
        raise ValueError(f"`ts` must be 1-D, got shape {tuple(ts.shape)}")

    z = z0
    kl = torch.zeros(z0.shape[0], device=z0.device, dtype=z0.dtype)
    z_stages = [z]
    kl_stages = [kl]

    for i in range(len(ts) - 1):
        t0 = float(ts[i].item())
        t1 = float(ts[i + 1].item())
        span = t1 - t0
        if span <= 0:
            # Degenerate or reversed interval — no integration.
            z_stages.append(z)
            kl_stages.append(kl)
            continue
        n_steps = max(1, int(round(span / dt_step)))
        dt = span / n_steps
        sqrt_dt = dt ** 0.5
        t = t0
        for _ in range(n_steps):
            fq = f_post(z, t)                       # (N, D)
            if f_prior is None:
                fp = torch.zeros_like(fq)
            else:
                fp = f_prior(z, t)                  # (N, D)
            gv = g_net(z, t)                        # (N, D)
            if generator is not None:
                eps = torch.randn(z.shape, generator=generator,
                                  device=z.device, dtype=z.dtype)
            else:
                eps = torch.randn_like(z)
            z = z + fq * dt + gv * eps * sqrt_dt
            # KL integrand at the pre-step state (Ito convention), with the
            # same f_post / f_prior / g as the SDE step.
            diff = (fq - fp) / (gv + 1e-8)
            kl = kl + 0.5 * (diff * diff).sum(dim=-1) * dt
            t = t + dt
        z_stages.append(z)
        kl_stages.append(kl)

    traj = torch.stack(z_stages, dim=0)            # (T, N, D)
    kl_out = torch.stack(kl_stages, dim=0)         # (T, N)
    return traj, kl_out


# ---------------------------------------------------------------------------
# Sampler
# ---------------------------------------------------------------------------


class SDESampler(nn.Module):
    """Latent neural SDE with a path-KL accumulator.

    Two entry points:

    - ``forward_chain(z, ts)`` integrates through a sequence of stage times
      and returns ``(trajectory, kl)``; Stage-B training uses it.
    - ``forward(z, t, t0)`` integrates from ``t0`` to ``t`` and returns only
      the end state.

    Parameters
    ----------
    latent_dim : int, default 48
        Dimension of ``z``; callers pass the model's latent dimension.
    prior_mode : {"zero", "learned", "learned_t_only"}, default "zero"
        Prior drift: :class:`ZeroDrift`, a :class:`Drift` with its own
        weights, or a state-independent :class:`DriftTimeOnly`.
    diffusion_mode : {"scalar", "per_dim", "fixed"}, default "scalar"
        See :class:`Diffusion`.
    drift_hidden : tuple of int, default (128, 128)
        Hidden widths of the posterior drift (and of a ``"learned"`` prior).
    diffusion_hidden : tuple of int, default (32,)
        Hidden widths of the diffusion network.
    time_embed_dim, time_hidden : int, default 4, 10
        Time-embedding sizes shared by all networks.
    diffusion_init : float, default 0.02
        Diffusion at initialisation.
    dt_step : float, default 0.01
        Euler step size.
    prior_hidden : tuple of int, default (64,)
        Hidden widths of a ``"learned_t_only"`` prior.

    Raises
    ------
    ValueError
        If ``prior_mode`` (or ``diffusion_mode``) is unknown.
    """

    def __init__(self, latent_dim: int = 48, prior_mode: str = "zero",
                 diffusion_mode: str = "scalar",
                 drift_hidden: Tuple[int, ...] = (128, 128),
                 diffusion_hidden: Tuple[int, ...] = (32,),
                 time_embed_dim: int = 4, time_hidden: int = 10,
                 diffusion_init: float = 0.02, dt_step: float = 0.01,
                 prior_hidden: Tuple[int, ...] = (64,)) -> None:
        super().__init__()
        if prior_mode not in ("zero", "learned", "learned_t_only"):
            raise ValueError(f"Unknown prior_mode: {prior_mode!r}")
        self.latent_dim = latent_dim
        self.prior_mode = prior_mode
        self.diffusion_mode = diffusion_mode
        self.dt_step = dt_step

        self.f_post = Drift(latent_dim, hidden=drift_hidden,
                            time_embed_dim=time_embed_dim, time_hidden=time_hidden)
        if prior_mode == "learned":
            self.f_prior: nn.Module = Drift(
                latent_dim, hidden=drift_hidden,
                time_embed_dim=time_embed_dim, time_hidden=time_hidden,
            )
        elif prior_mode == "learned_t_only":
            # State-independent prior f_prior(t): it cannot track the
            # state-dependent f_post(z, t), so the KL stays informative.
            self.f_prior = DriftTimeOnly(
                latent_dim, hidden=prior_hidden,
                time_embed_dim=time_embed_dim, time_hidden=time_hidden,
            )
        else:
            self.f_prior = ZeroDrift()
        self.g_net = Diffusion(latent_dim, mode=diffusion_mode,
                               time_embed_dim=time_embed_dim,
                               time_hidden=time_hidden,
                               hidden=diffusion_hidden,
                               init_sigma=diffusion_init)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _prior_module(self) -> Optional[nn.Module]:
        """Return the prior drift module."""
        return self.f_prior if self.prior_mode == "learned" else self.f_prior

    def forward_chain(self, z: torch.Tensor, ts: torch.Tensor,
                      generator: Optional[torch.Generator] = None,
                      ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Integrate through ``ts`` and return ``(traj, kl)``.

        Parameters
        ----------
        z : torch.Tensor, shape (N, D)
            Initial state at ``ts[0]``.
        ts : torch.Tensor
            1-D increasing stage times.
        generator : torch.Generator, optional
            Generator for the Brownian increments.

        Returns
        -------
        traj : torch.Tensor, shape (len(ts), N, D)
        kl : torch.Tensor, shape (len(ts), N)
            See :func:`euler_maruyama_kl`.
        """
        return euler_maruyama_kl(
            f_prior=self.f_prior, f_post=self.f_post, g_net=self.g_net,
            z0=z, ts=ts, dt_step=self.dt_step, generator=generator,
        )

    def forward(self, z: torch.Tensor,
                t: torch.Tensor, t0: torch.Tensor,
                generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """Integrate from ``t0`` to ``t`` and return the end state only.

        ``t`` and ``t0`` may be scalars or tensors (e.g. ``(batch, 1)``) whose
        entries all hold the same time; the first entry is used. The path KL
        is discarded.

        Parameters
        ----------
        z : torch.Tensor, shape (N, D)
            State at ``t0``.
        t, t0 : torch.Tensor
            End and start times.
        generator : torch.Generator, optional
            Generator for the Brownian increments.

        Returns
        -------
        torch.Tensor, shape (N, D)
            State at ``t``.
        """
        if t.dim() == 0 or t0.dim() == 0:
            t_scalar = float(t.item() if t.dim() == 0 else t.reshape(-1)[0].item())
            t0_scalar = float(t0.item() if t0.dim() == 0 else t0.reshape(-1)[0].item())
        else:
            t_scalar = float(t.reshape(-1)[0].item())
            t0_scalar = float(t0.reshape(-1)[0].item())
        ts = torch.tensor([t0_scalar, t_scalar], device=z.device, dtype=z.dtype)
        traj, _ = self.forward_chain(z, ts, generator=generator)
        return traj[-1]
