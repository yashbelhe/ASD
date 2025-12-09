"""Compatibility shim that re-exports utilities from python.utils.*"""

from .utils.device import DEVICE
from .utils.segments import (
    points_on_grid,
    get_segments_on_grid,
    snap_segment_to_discontinuity,
)
from .utils.boundary import (
    BoundaryLossConfig,
    boundary_loss,
    edge_loss_slang,
)
from .utils.finite_diff import (
    compute_and_plot_fwd_grad,
    efficient_finite_diff_grad,
    plot_fwd_grad_ours_and_fd,
)
from .utils.slang_runtime import (
    launch_1d,
    SlangShaderForwardGrad,
    SlangShaderForwardGradRGB,
    SlangShader,
    SlangShaderRGB,
)
from .utils.camera import Camera
from .utils.io import imwrite, vidwrite

__all__ = [
    'DEVICE',
    'points_on_grid',
    'get_segments_on_grid',
    'snap_segment_to_discontinuity',
    'BoundaryLossConfig',
    'boundary_loss',
    'edge_loss_slang',
    'compute_and_plot_fwd_grad',
    'efficient_finite_diff_grad',
    'plot_fwd_grad_ours_and_fd',
    'launch_1d',
    'SlangShaderForwardGrad',
    'SlangShaderForwardGradRGB',
    'SlangShader',
    'SlangShaderRGB',
    'Camera',
    'imwrite',
    'vidwrite',
]
