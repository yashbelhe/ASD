import matplotlib.pyplot as plt
import torch
from dataclasses import dataclass

from .segments import points_on_grid, get_segments_on_grid, snap_segment_to_discontinuity
from .kde import KDEKeOpsKNN

def _infer_integrand_device(integrand):
    for param in integrand.parameters():
        return param.device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _to_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    return torch.as_tensor(value, device=device)


def _select_segments(grid_size, dim, custom_segments, device):
    if custom_segments is not None:
        assert grid_size is None
        segments = _to_device(custom_segments, device)
    else:
        segments = get_segments_on_grid(grid_size, dim=dim, device=device)
    return segments


def _sample_edge_points(dim, grid_size, num_subdivision, custom_segments, custom_x, integrand):
    device = _infer_integrand_device(integrand)
    if custom_x is not None:
        x = _to_device(custom_x, device)
    else:
        segments = _select_segments(grid_size, dim, custom_segments, device)
        integrand_pw_constant = lambda pts: integrand(pts, ret_const=True)
        segments = snap_segment_to_discontinuity(segments, integrand_pw_constant, num_subdivision)
        x = segments.mean(axis=1)
    mask = torch.all(x > 0, dim=1) & torch.all(x < 1, dim=1)
    x = x[mask]
    if x.shape[0] > 10**7:
        idx = torch.randperm(x.shape[0], device=x.device)[:10**7]
        x = x[idx]
    x.requires_grad = True
    return x


def _plot_segments_if_needed(integrand, x, f_min_idx, dim, plot_resolution, should_plot):
    if not should_plot:
        return
    points_ = points_on_grid(plot_resolution, dim=dim)
    values = integrand(points_).detach()
    if len(values.shape) > 1 and values.shape[1] == 3:
        values = values.reshape(plot_resolution, plot_resolution, 3)
    else:
        values = values.reshape(plot_resolution, plot_resolution)
    if dim != 2:
        print('TODO: Plotting 3D integrand')
        return
    mask = f_min_idx >= 0
    plt.figure(figsize=(8, 8))
    plt.imshow(values.cpu(), origin='lower', extent=[0, 1, 0, 1])
    plt.colorbar()
    plt.scatter(
        x[mask, 0].detach().cpu(),
        x[mask, 1].detach().cpu(),
        c=f_min_idx[mask].detach().cpu(),
        cmap='tab10',
        s=0.5,
    )
    plt.show()


def _compute_kde_weights(x, f_min_idx, kde_k):
    if x.numel() == 0:
        return torch.zeros(0, device=x.device), f_min_idx
    shifted = ((f_min_idx >= 0).type(f_min_idx.dtype) * (f_min_idx + 1))
    kde = KDEKeOpsKNN(do_KDE=True)
    weights = kde(x, shifted, K=kde_k)
    return weights, shifted - 1


def _compute_df_dx(integrand, x, p, df_dx_mode, f):
    if df_dx_mode == 'forward':
        dim = x.shape[1]
        grads = []
        for axis in range(dim):
            dx = torch.zeros_like(x)
            dx[:, axis] = 1.0
            grads.append(integrand.forward_grad(x, dx, torch.zeros_like(p), ret_impl=True)[1])
        return torch.stack(grads, dim=1)
    return torch.autograd.grad(f.sum(), x, create_graph=False, retain_graph=True)[0].detach()


def _compute_delta_outputs(integrand, x, f_min_idx, mode, mode_aux_data, mask_fn):
    with torch.no_grad():
        out_true = integrand(x, impl_idx=f_min_idx, force_sign=0)
        out_false = integrand(x, impl_idx=f_min_idx, force_sign=1)
        delta_out = (out_true - out_false).detach()
        if mode in ('L2_img', 'L1_img'):
            if mode_aux_data is None:
                raise ValueError('mode_aux_data must be provided for image supervision modes.')
            err_size = mode_aux_data.shape[0]
            x_indices = (x[:, 1] * err_size).long().clamp(0, err_size - 1)
            y_indices = (x[:, 0] * err_size).long().clamp(0, err_size - 1)
            gt_vals = mode_aux_data[x_indices, y_indices]
            if mode == 'L1_img':
                delta_out = ((out_true - gt_vals).abs() - (out_false - gt_vals).abs()).detach()
            else:
                delta_out = ((out_true - gt_vals) ** 2 - (out_false - gt_vals) ** 2).detach()
        elif mode in ('L2_test_fn', 'L1_test_fn'):
            if mode_aux_data is None:
                raise ValueError('mode_aux_data must be provided for test-function supervision modes.')
            gt_vals = mode_aux_data(x)
            if mode == 'L2_test_fn':
                delta_out = ((out_true - gt_vals) ** 2 - (out_false - gt_vals) ** 2).detach()
            else:
                delta_out = ((out_true - gt_vals).abs() - (out_false - gt_vals).abs()).detach()
        if len(delta_out.shape) == 1:
            delta_out = delta_out.unsqueeze(-1)
        if mask_fn is not None:
            delta_out *= mask_fn(x).unsqueeze(-1)
    return delta_out

@dataclass
class BoundaryLossConfig:
    dim: int = 2
    grid_size: int = 2000
    plot_segments: bool = False
    fwd_grad: tuple = (False, -1)
    num_subdivision: int = 20
    div_eps: float = 1e-15
    plot_resolution: int = 1000
    kde_k: int = 9
    mode: str = "direct"
    mode_aux_data: any = None
    df_dx_mode: str = "forward"
    mask_fn: any = None
    custom_segments: any = None
    custom_x: any = None


def _pop(kwargs, name, default):
    name_lower = name.lower()
    if name in kwargs:
        return kwargs.pop(name)
    if name_lower in kwargs:
        return kwargs.pop(name_lower)
    return default


def boundary_loss(integrand, cfg=None, **kwargs):
    """
    Compute the boundary loss. Accepts either a BoundaryLossConfig or the legacy keyword arguments.
    """
    if isinstance(cfg, BoundaryLossConfig):
        if kwargs:
            raise TypeError(f"Unexpected keyword arguments: {list(kwargs.keys())}")
        config = cfg
    else:
        dim = cfg if cfg is not None else _pop(kwargs, "dim", 2)
        config = BoundaryLossConfig(
            dim=dim,
            grid_size=_pop(kwargs, "GRID_SIZE", 2000),
            plot_segments=_pop(kwargs, "plot_segments", False),
            fwd_grad=_pop(kwargs, "fwd_grad", (False, -1)),
            num_subdivision=_pop(kwargs, "NUM_SUBDIVISION", 20),
            div_eps=_pop(kwargs, "DIV_EPS", 1e-15),
            plot_resolution=_pop(kwargs, "PLOT_RESOLUTION", 1000),
            kde_k=_pop(kwargs, "KDE_K", 9),
            mode=_pop(kwargs, "mode", "direct"),
            mode_aux_data=_pop(kwargs, "mode_aux_data", None),
            df_dx_mode=_pop(kwargs, "df_dx_mode", "forward"),
            mask_fn=_pop(kwargs, "mask_fn", None),
            custom_segments=_pop(kwargs, "custom_segments", None),
            custom_x=_pop(kwargs, "custom_x", None),
        )
        if kwargs:
            raise TypeError(f"Unexpected keyword arguments: {list(kwargs.keys())}")
    cfg = config

    assert cfg.mode in ['direct', 'L2_img', 'L2_test_fn', 'L1_img', 'L1_test_fn']

    x = _sample_edge_points(
        cfg.dim,
        cfg.grid_size,
        cfg.num_subdivision,
        cfg.custom_segments,
        cfg.custom_x,
        integrand,
    )

    f, f_min_idx = integrand(x, ret_impl=True)
    _plot_segments_if_needed(integrand, x, f_min_idx, cfg.dim, cfg.plot_resolution, cfg.plot_segments)

    p, f_min_idx = _compute_kde_weights(x, f_min_idx, cfg.kde_k)
    df_dx = _compute_df_dx(integrand, x, p, cfg.df_dx_mode, f)
    df_dx_norm = torch.norm(df_dx, p=2, dim=1).detach().clamp(min=1e-1, max=1e4)
    delta_out = _compute_delta_outputs(integrand, x, f_min_idx, cfg.mode, cfg.mode_aux_data, cfg.mask_fn)

    p = p.unsqueeze(-1)
    f = f.unsqueeze(-1)
    df_dx_norm = df_dx_norm.unsqueeze(-1)

    if cfg.fwd_grad[0]:
        dp = torch.zeros_like(p)
        if isinstance(cfg.fwd_grad[1], int):
            dp[cfg.fwd_grad[1]] = 1.0
        else:
            dp = cfg.fwd_grad[1]
        dx = torch.zeros_like(x)
        df_dp0 = integrand.forward_grad(x, dx, dp, ret_impl=True)[1].unsqueeze(-1)
        return (p * delta_out * df_dp0 / (df_dx_norm + cfg.div_eps)), x

    for tensor, name in ((p, 'p'), (delta_out, 'delta_out'), (f, 'f'), (df_dx_norm, 'df_dx_norm')):
        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
            print(f"Warning: NaN or Inf detected in {name}")

    return (p * delta_out * f / (df_dx_norm + cfg.div_eps)).sum()
