import math
import torch
import torch.nn as nn
import slangtorch
from python.helpers import (
    SlangShader,
    SlangShaderForwardGrad,
    SlangShaderRGB,
    SlangShaderForwardGradRGB,
    launch_1d,
)


class BaseIntegrandSlang(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, impl_idx=None, force_sign=-1, ret_const=False, ret_impl=False):
        r, impl_fn, impl_idx, out_idx = SlangShader.apply(
            x, self.p, impl_idx, force_sign, self.shader, ret_const, ret_impl
        )
        if ret_const:
            return out_idx
        if ret_impl:
            return impl_fn, impl_idx
        return r

    def forward_grad(self, x, d_x, d_p, impl_idx=None, force_sign=-1, ret_const=False, ret_impl=False):
        d_r, d_impl_fn = SlangShaderForwardGrad(
            x, d_x, self.p, d_p, impl_idx, force_sign, self.shader, ret_const, ret_impl
        )
        return d_r, d_impl_fn


class BaseIntegrandSlangRGB(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, impl_idx=None, force_sign=-1, ret_const=False, ret_impl=False):
        r, impl_fn, impl_idx, out_idx = SlangShaderRGB.apply(
            x, self.p, impl_idx, force_sign, self.shader, ret_const, ret_impl
        )
        if ret_const:
            return out_idx
        if ret_impl:
            return impl_fn, impl_idx
        return r

    def forward_grad(self, x, d_x, d_p, impl_idx=None, force_sign=-1, ret_const=False, ret_impl=False):
        d_r, d_impl_fn = SlangShaderForwardGradRGB(
            x, d_x, self.p, d_p, impl_idx, force_sign, self.shader, ret_const, ret_impl
        )
        return d_r, d_impl_fn


class CirclesIntegrandSlang(BaseIntegrandSlang):
    def __init__(self):
        super().__init__()
        self.shader = slangtorch.loadModule("slang/__gen__circle_single.slang")
        self.p = nn.Parameter(torch.tensor([
            0.5, 0.5, 0.4, 1.0,
        ]))
        self.out_dim = 1


class MultiCirclesIntegrandSlang(BaseIntegrandSlang):
    def __init__(self, circles):
        """circles: iterable of (cx, cy, radius, opacity)."""
        super().__init__()
        self.shader = slangtorch.loadModule("slang/__gen__circles.slang")
        params = []
        for (cx, cy, r, op) in circles:
            params.extend([cx, cy, r, op])
        self.p = nn.Parameter(torch.tensor(params, dtype=torch.float32))
        self.out_dim = 1


class BinaryThresholdIntegrandSlang(BaseIntegrandSlang):
    def __init__(self, grid_size=2, seed=40, init_scale=1 / 30):
        super().__init__()
        self.shader = slangtorch.loadModule("slang/__gen__binary_threshold.slang")
        if seed is not None:
            torch.manual_seed(seed)
        params = torch.zeros(1 + grid_size * grid_size, dtype=torch.float32)
        params[0] = float(grid_size)
        params[1:] = (torch.rand(grid_size * grid_size) - 0.5) * init_scale
        self.p = nn.Parameter(params)
        self.out_dim = 1


class CelShadingIntegrandSlang(BaseIntegrandSlangRGB):
    def __init__(self, time=4.5):
        super().__init__()
        self.shader = slangtorch.loadModule("slang/__gen__cel_shading.slang")
        params = torch.zeros(10, dtype=torch.float32)
        i_time = torch.tensor(time, dtype=torch.float32)
        params[0] = i_time
        params[1] = torch.sin(i_time * 2.0) * 6.0
        params[2] = 4.0
        params[3] = torch.sin(i_time * 1.25) * 5.0
        params[4] = 0.25
        params[5] = 0.7
        params[6] = 0.9
        params[7] = 0.1
        self.p = nn.Parameter(params)
        self.out_dim = 3


class TrilinearThresholdIntegrandSlang(BaseIntegrandSlang):
    def __init__(self, grid_size=2, seed=40, init_scale=1 / 30):
        super().__init__()
        self.shader = slangtorch.loadModule("slang/__gen__trilinear_thresholding.slang")
        self.grid_size = grid_size
        torch.manual_seed(seed if seed is not None else 0)
        num_grid_vals = grid_size ** 3
        params = torch.zeros(1 + num_grid_vals + 2, dtype=torch.float32)
        params[0] = float(grid_size)
        params[1 : 1 + num_grid_vals] = (torch.rand(num_grid_vals) - 0.5) * init_scale
        params[-2:] = torch.tensor([0.0, 1.0])
        self.p = nn.Parameter(params)
        self.p.register_hook(lambda grad: grad.index_fill(0, torch.tensor([0], device=grad.device), 0))
        self.out_dim = 1


class ImplicitRaymarchingIntegrandSlang(BaseIntegrandSlangRGB):
    def __init__(self, grid_size=64, threshold=0.0, grid_values=None, num_view=8, sphere_radius=0.3, seed=0):
        super().__init__()
        self.shader = slangtorch.loadModule("slang/__gen__implicit_raymarching.slang")
        self.grid_size = grid_size
        self.threshold = threshold
        torch.manual_seed(seed)
        if grid_values is None:
            grid_values = ((sphere_radius - torch.linalg.norm(
                torch.stack(torch.meshgrid(
                    torch.linspace(0, 1, grid_size),
                    torch.linspace(0, 1, grid_size),
                    torch.linspace(0, 1, grid_size),
                    indexing="ij",
                ), dim=0) - 0.5, dim=0))).unsqueeze(0)
        self.grid_values = nn.Parameter(grid_values.reshape(grid_size, grid_size, grid_size))
        self.num_view = num_view
        self.active_view = 0

        phi = torch.linspace(0, 2 * math.pi, int(math.sqrt(num_view)) + 1)
        theta = torch.linspace(0, math.pi, int(math.sqrt(num_view)) + 1)
        phi, theta = torch.meshgrid(phi, theta, indexing="ij")
        camera_pos = torch.stack([
            torch.sin(theta).flatten() * torch.cos(phi).flatten(),
            torch.sin(theta).flatten() * torch.sin(phi).flatten(),
            torch.cos(theta).flatten(),
        ], dim=1)
        indices = torch.randperm(camera_pos.size(0))
        camera_pos = camera_pos[indices][:num_view]
        camera_pos = 1.5 * camera_pos / torch.linalg.norm(camera_pos, dim=1, keepdim=True)
        self.register_buffer("camera_pos", camera_pos)
        self.register_buffer("look_at", torch.tensor([0.0, 0.0, 0.0]))
        self.register_buffer("fov", torch.tensor([0.6]))
        self.register_buffer("step_size", torch.tensor([2.0 / 500.0]))
        self.register_buffer("light_dir", torch.tensor([0.5, 0.5, -1.5]))
        self.register_buffer("light_color", torch.tensor([1.0, 1.0, 1.0]))
        self.register_buffer("diffuse_color", torch.tensor([0.8, 0.3, 0.2]))
        self.out_dim = 3

    def set_active_view(self, idx: int):
        self.active_view = idx % max(1, self.camera_pos.shape[0])

    @property
    def p(self):
        device = self.grid_values.device
        grid_flat = self.grid_values.reshape(-1)
        cam = torch.cat([
            self.camera_pos[self.active_view],
            self.look_at,
            self.fov,
            self.step_size,
        ])
        light = torch.cat([self.light_dir, self.light_color])
        material = self.diffuse_color
        header = torch.tensor([float(self.grid_size), float(self.threshold)], device=device)
        return torch.cat([header, grid_flat, cam, light, material])

class VoronoiSimpleIntegrandSlang(BaseIntegrandSlangRGB):
    def __init__(self, grid_size=10, jitter_scale=0.8, seed=12, warp=False):
        super().__init__()
        self.use_warp = warp
        shader_name = "__gen__voronoi_simple_warp.slang" if warp else "__gen__voronoi_simple.slang"
        self.shader = slangtorch.loadModule(f"slang/{shader_name}")
        self.grid_size = grid_size

        # Precompute per-cell points/colors; store as (x, y, r, g, b) per site.
        rand_device = torch.device("cpu")
        rng = torch.Generator(device=rand_device)
        if seed is not None:
            rng.manual_seed(seed)

        params = []
        for i in range(grid_size):
            for j in range(grid_size):
                cx = (i + 0.5) / grid_size
                cy = (j + 0.5) / grid_size
                if seed is None:
                    jitter = torch.rand(2, device=rand_device)
                    color = torch.rand(3, device=rand_device)
                else:
                    jitter = torch.rand(2, generator=rng, device=rand_device)
                    color = torch.rand(3, generator=rng, device=rand_device)
                x = cx + (jitter[0].item() - 0.5) * jitter_scale / grid_size
                y = cy + (jitter[1].item() - 0.5) * jitter_scale / grid_size
                x = max(i / grid_size, min((i + 1) / grid_size - 1e-6, x))
                y = max(j / grid_size, min((j + 1) / grid_size - 1e-6, y))
                params.extend([x, y, color[0].item(), color[1].item(), color[2].item()])

        self.p = nn.Parameter(torch.tensor(params, dtype=torch.float32))
        self.out_dim = 3


def _create_line_constant_block(x0, y0, x1, y1, width, color, opacity):
    """Return a length-20 parameter block for a single line primitive."""
    block = [0.0] * 20
    block[0] = 4.0  # line primitive type
    block[1] = x0
    block[2] = y0
    block[3] = x1
    block[4] = y1
    block[5] = width
    block[6] = 0.0  # padding for control points
    block[7] = width  # stroke width entry for acceleration structure
    block[8] = 0.0  # constant fill type
    block[9] = color[0]
    block[10] = color[1]
    block[11] = color[2]
    block[12] = opacity
    # Remaining entries stay zero
    return block


def _create_bezier_constant_block(x0, y0, x1, y1, x2, y2, width, color, opacity):
    """Return a length-20 parameter block for a quadratic Bezier primitive."""
    block = [0.0] * 20
    block[0] = 0.0  # bezier primitive type
    block[1] = x0
    block[2] = y0
    block[3] = x1
    block[4] = y1
    block[5] = x2
    block[6] = y2
    block[7] = width
    block[8] = 0.0  # constant fill type
    block[9] = color[0]
    block[10] = color[1]
    block[11] = color[2]
    block[12] = opacity
    return block


_accel_setup_kernel = slangtorch.loadModule("slang/__gen__accel_structure_setup.slang")


def _build_acceleration_structure(primitive_types, control_points, stroke_widths, grid_size, max_elements_per_cell, device):
    """GPU implementation of the acceleration structure builder for all supported primitives."""
    num_primitives = primitive_types.shape[0]
    grid_tensor = torch.zeros(
        grid_size * grid_size * (max_elements_per_cell + 1),
        dtype=torch.int32,
        device=device,
    )
    primitive_indices = torch.arange(num_primitives, device=device, dtype=torch.int32)
    control = control_points.view(-1, 6)
    launch_1d(
        _accel_setup_kernel.setup_unified_acceleration_structure(
            primitive_types=primitive_types.int(),
            control_points=control.float(),
            stroke_widths=stroke_widths.float().unsqueeze(-1),
            primitive_indices=primitive_indices,
            grid_tensor=grid_tensor,
            grid_size=grid_size,
            max_elements_per_cell=max_elements_per_cell,
        ),
        LEN=max(1, num_primitives),
    )
    return grid_tensor.view(grid_size, grid_size, max_elements_per_cell + 1).to(torch.float32)


class VectorGraphicsRGBPaddedAccelIntegrandSlang(BaseIntegrandSlangRGB):
    """Minimal vector-graphics integrand with a grid acceleration structure."""

    def __init__(
        self,
        n=30,
        grid_size=64,
        max_elements_per_cell=128,
        background_color=(0.0, 0.0, 0.0),
        primitive_type="line",
        seed=42,
    ):
        super().__init__()
        primitive_type = primitive_type.lower()
        if primitive_type not in {"line", "bezier"}:
            raise ValueError("primitive_type must be either 'line' or 'bezier'.")
        self.shader = slangtorch.loadModule("slang/__gen__vector_graphics_rgb_padded_accel.slang")
        self.grid_size = grid_size
        self.max_elements_per_cell = max_elements_per_cell
        self.background_color = torch.tensor(background_color, dtype=torch.float32)
        self.primitive_type = primitive_type

        torch.manual_seed(seed if seed is not None else 42)
        params = []
        cell_width = 1.0 / n
        for i in range(n):
            for j in range(n):
                color = torch.rand(3).tolist()
                opacity = 1.0
                if self.primitive_type == "line":
                    cx = (i + 0.5) / n + (torch.rand(1).item() - 0.5) * 0.2 * cell_width
                    cy = (j + 0.5) / n + (torch.rand(1).item() - 0.5) * 0.2 * cell_width
                    length = 0.3 * cell_width
                    angle = torch.rand(1).item() * math.tau
                    dx = math.cos(angle) * length
                    dy = math.sin(angle) * length
                    x0 = min(max(cx - dx, 0.0), 1.0)
                    y0 = min(max(cy - dy, 0.0), 1.0)
                    x1 = min(max(cx + dx, 0.0), 1.0)
                    y1 = min(max(cy + dy, 0.0), 1.0)
                    width = 0.6 / n
                    block = _create_line_constant_block(x0, y0, x1, y1, width, color, opacity)
                else:
                    # Stratify control points within the grid cell to encourage coverage.
                    span = 1.0 / n
                    x0 = (i + torch.rand(1).item() * 0.3) / n
                    y0 = (j + torch.rand(1).item()) / n
                    x1 = (i + 0.35 + torch.rand(1).item() * 0.3) / n
                    y1 = (j + torch.rand(1).item()) / n
                    x2 = (i + 0.7 + torch.rand(1).item() * 0.3) / n
                    y2 = (j + torch.rand(1).item()) / n
                    x0 = min(max(x0, 0.0), 1.0)
                    y0 = min(max(y0, 0.0), 1.0)
                    x1 = min(max(x1, 0.0), 1.0)
                    y1 = min(max(y1, 0.0), 1.0)
                    x2 = min(max(x2, 0.0), 1.0)
                    y2 = min(max(y2, 0.0), 1.0)
                    width = (0.2 + torch.rand(1).item() * 0.3) * span
                    block = _create_bezier_constant_block(x0, y0, x1, y1, x2, y2, width, color, opacity)
                params.extend(block)

        params_tensor = torch.tensor(params, dtype=torch.float32)
        self.primitive_types = nn.Parameter(params_tensor[0::20])
        self.control_points = nn.Parameter(torch.cat([params_tensor[i : i + 6] for i in range(1, len(params_tensor), 20)]))
        self.stroke_widths = nn.Parameter(params_tensor[7::20])
        self.fill_types = nn.Parameter(params_tensor[8::20])
        self.fill_colors = nn.Parameter(torch.cat([params_tensor[i : i + 3] for i in range(9, len(params_tensor), 20)]))
        self.opacities = nn.Parameter(params_tensor[12::20])
        self.other_fill_params = nn.Parameter(torch.cat([params_tensor[i : i + 6] for i in range(13, len(params_tensor), 20)]))

        self.total_primitives = len(self.primitive_types)
        self.num_primitives = self.total_primitives
        self.out_dim = 3

    @property
    def p(self):
        device = self.primitive_types.device
        active_indices = torch.arange(self.total_primitives, device=device)
        idx = torch.arange(self.num_primitives, device=device)

        result = torch.zeros(self.total_primitives * 20, device=device)
        result[idx * 20] = self.primitive_types[active_indices]

        control_idx = idx.unsqueeze(1) * 20 + torch.arange(1, 7, device=device)
        active_control_points = self.control_points.view(-1, 6)[active_indices]
        result.index_put_((control_idx.view(-1),), active_control_points.view(-1))

        result[idx * 20 + 7] = self.stroke_widths[active_indices]
        result[idx * 20 + 8] = self.fill_types[active_indices]

        color_idx = idx.unsqueeze(1) * 20 + torch.arange(9, 12, device=device)
        active_colors = self.fill_colors.view(-1, 3)[active_indices]
        result.index_put_((color_idx.view(-1),), active_colors.view(-1))

        result[idx * 20 + 12] = self.opacities[active_indices]

        other_idx = idx.unsqueeze(1) * 20 + torch.arange(13, 19, device=device)
        active_other = self.other_fill_params.view(-1, 6)[active_indices]
        result.index_put_((other_idx.view(-1),), active_other.view(-1))

        grid = _build_acceleration_structure(
            self.primitive_types[active_indices],
            self.control_points.view(-1, 6)[active_indices],
            self.stroke_widths[active_indices],
            self.grid_size,
            self.max_elements_per_cell,
            device,
        ).reshape(-1)

        header = torch.tensor(
            [float(self.num_primitives), float(self.grid_size), float(self.max_elements_per_cell)],
            device=device,
        )
        return torch.cat([header, self.background_color.to(device), result, grid])
