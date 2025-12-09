import math
import warnings

import torch

try:  # Optional dependency
    from pykeops.torch import LazyTensor
except ImportError as e:  # pragma: no cover
    LazyTensor = None
    _PYKEOPS_IMPORT_ERROR = e
else:
    _PYKEOPS_IMPORT_ERROR = None


def ranges_slices(batch):
    Ns = batch.bincount()
    indices = Ns.cumsum(0)
    ranges = torch.cat((0 * indices[:1], indices))
    ranges = torch.stack((ranges[:-1], ranges[1:])).t().int().contiguous().to(batch.device)
    slices = (1 + torch.arange(len(Ns))).int().to(batch.device)
    return ranges, slices


def diagonal_ranges(batch_x=None, batch_y=None):
    if batch_x is None and batch_y is None:
        return None
    ranges_x, slices_x = ranges_slices(batch_x)
    ranges_y, slices_y = ranges_slices(batch_y)
    return ranges_x, slices_x, ranges_y, ranges_y, slices_y, ranges_x


def _require_pykeops():
    if LazyTensor is None:
        raise ImportError(
            "pykeops is required for KDEKeOpsKNN. Install pykeops or switch to KDETorchKNN."
        ) from _PYKEOPS_IMPORT_ERROR


def _ball_volume(dim, radius):
    """Return the volume/length of an (dim)-D ball with radius `radius`.

    Derivation mirrors
    https://math.stackexchange.com/questions/4028633/which-is-the-algorithm-for-knn-density-estimator
    """

    if dim == 1:
        return 2.0 * radius
    if dim == 2:
        return math.pi * radius.pow(2)
    if dim == 3:
        return (4.0 / 3.0) * math.pi * radius.pow(3)
    raise ValueError(f"Only 1D-3D KDE volumes are supported, got dimension {dim}.")


class KDEKeOpsKNN(torch.nn.Module):
    def __init__(self, do_KDE=True):
        super().__init__()
        self.do_KDE = do_KDE

    def forward(self, x, min_t_idx, K, sz=5):
        if not self.do_KDE:
            raise ValueError("do_KDE=False is unsupported for KDEKeOpsKNN.")
        if K <= 1:
            raise ValueError("K must be greater than 1 for KDE.")
        _require_pykeops()

        min_t_idx = min_t_idx.to(torch.int64)
        with torch.no_grad():
            _, NI = x.shape
            if NI not in (2, 3):
                raise ValueError(f"KDEKeOpsKNN only supports 2D or 3D inputs, got {NI}D.")
            p = torch.zeros(x.shape[0], device=x.device)

            mask = min_t_idx > 0
            x_ = x[mask]
            min_t_idx_m = min_t_idx[mask]
            if x_.numel() == 0:
                return p

            if NI == 2:
                y = (x_ * sz).long()
                y_f = y[:, 0] * sz + y[:, 1]
                y_f += min_t_idx_m * sz ** 2
            else:  # NI == 3
                y = (x_ * sz).long()
                y_f = y[:, 0] * sz ** 2 + y[:, 1] * sz + y[:, 2]
                y_f += min_t_idx_m * sz ** 3

            sort_idx = torch.argsort(y_f)
            inv_sort_idx = torch.zeros_like(sort_idx)
            inv_sort_idx[sort_idx] = torch.arange(sort_idx.shape[0], device=sort_idx.device)
            x_, y_f = x_[sort_idx], y_f[sort_idx]

            x_i = LazyTensor(x_[:, None, :])
            y_j = LazyTensor(x_[None, :, :])
            D_ij = ((x_i - y_j) ** 2).sum(-1)
            D_ij.ranges = diagonal_ranges(y_f, y_f)

            idx_i = D_ij.argKmin(K, dim=1)
            knn = x_[idx_i[:, -1]]
            knn_dist = (x_ - knn).square().sum(axis=-1).sqrt()
            volume = _ball_volume(NI - 1, knn_dist)
            p_m = volume / (K - 1)
            p[mask] = p_m[inv_sort_idx]
            return p


class KDETorchKNN(torch.nn.Module):
    def __init__(self, uniform_KDE=False):
        super().__init__()
        self.uniform_KDE = uniform_KDE
        if uniform_KDE:
            warnings.warn(
                "Uniform KDE selected; boundary integral probabilities may be less accurate.",
                RuntimeWarning,
            )

    def forward(self, x, min_t_idx, K, sz=None):  # noqa: ARG002 (sz kept for API parity)
        _, NI = x.shape
        if K <= 1:
            raise ValueError("K must be greater than 1 for KDE.")

        with torch.no_grad():
            elems, counts = min_t_idx.unique(return_counts=True)
            p = torch.zeros(x.shape[0], device=x.device)
            for e, c in zip(elems, counts):
                mask = min_t_idx == e
                N = torch.sum(mask)
                if N < K or NI - 1 == 0:
                    p[mask] = 1 / c
                    continue
                if not self.uniform_KDE:
                    dist = (x[mask].unsqueeze(1) - x[mask].unsqueeze(0)).square().sum(axis=2).sqrt()
                    knn_dist, _ = torch.kthvalue(dist, k=K)
                    dim = NI - 1
                    p[mask] = _ball_volume(dim, knn_dist) / (K - 1)
                else:
                    p[mask] = 1 / c

        return p
