import torch


def points_on_grid(grid_size, jitter=False, dim=2, device=None):
    """Sample regularly spaced points on an N-D unit grid."""
    coords = [torch.linspace(0, 1, grid_size + 1, device=device)[:-1] for _ in range(dim)]
    meshgrids = torch.meshgrid(*coords, indexing="xy")
    if jitter:
        meshgrids = [grid + torch.rand_like(grid) / grid_size for grid in meshgrids]
    return torch.stack([grid.flatten() for grid in meshgrids], dim=1)


def get_segments_on_grid(grid_size, dim=2):
    """Sample random segments anchored on a grid and random directions."""
    grid = points_on_grid(grid_size, jitter=True, dim=dim)

    if dim == 2:
        theta = torch.rand_like(grid[:, 0]) * torch.pi * 2
        random_dir = torch.stack([torch.cos(theta), torch.sin(theta)], dim=1) / grid_size
        random_dir *= torch.rand_like(grid[:, 0]).unsqueeze(-1)
    elif dim == 3:
        theta = torch.rand_like(grid[:, 0]) * torch.pi * 2
        phi = torch.acos(2 * torch.rand_like(grid[:, 0]) - 1)
        x = torch.sin(phi) * torch.cos(theta)
        y = torch.sin(phi) * torch.sin(theta)
        z = torch.cos(phi)
        random_dir = torch.stack([x, y, z], dim=1) / grid_size
    else:
        random_dir = torch.randn(grid.shape, device=grid.device)
        random_dir = random_dir / torch.norm(random_dir, dim=1, keepdim=True) / grid_size

    return torch.stack([grid, grid + random_dir], dim=1)


def target_diff_segment(segments, target_fn):
    return target_fn(segments[:, 0]) - target_fn(segments[:, 1])


def diff_segment(segments):
    return segments[:, 0] - segments[:, 1]


def segment_grad_norm(segments, target_fn):
    num = target_diff_segment(segments, target_fn)
    den = diff_segment(segments)
    return num.abs() / torch.norm(den, p=2, dim=1)


def snap_segment_to_discontinuity(segments, target_fn, num_subdivision, max_segments=10**7):
    """Recursively bisect segments to drive them toward discontinuities."""
    for i in range(num_subdivision):
        seg_grad_norms = segment_grad_norm(segments, target_fn)
        segments = segments[seg_grad_norms > 1e-6]
        midpoints = segments.mean(axis=1)
        if i == num_subdivision - 1:
            break
        segments_first = torch.stack([segments[:, 0], midpoints], dim=1)
        segments_second = torch.stack([midpoints, segments[:, 1]], dim=1)
        segments = torch.cat([segments_first, segments_second], dim=0)
        if segments.shape[0] > max_segments:
            indices = torch.randperm(segments.shape[0], device=segments.device)[:max_segments]
            segments = segments[indices]
            print("WARNING: Reduced number of segments due to limit")
    return segments
