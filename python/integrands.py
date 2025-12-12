import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import slangtorch
from largesteps.geometry import compute_matrix
from largesteps.parameterize import from_differential, to_differential
import plyfile
from python.utils.slang_runtime import (
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


class HalfPlaneIntegrandSlang(BaseIntegrandSlang):
    def __init__(self, threshold=0.9):
        super().__init__()
        self.shader = slangtorch.loadModule("slang/__gen__half_plane.slang")
        self.p = nn.Parameter(torch.tensor([float(threshold)]))
        self.out_dim = 1


class TriangleIntegrandSlang(BaseIntegrandSlang):
    def __init__(self):
        super().__init__()
        self.shader = slangtorch.loadModule("slang/__gen__triangle.slang")
        self.p = nn.Parameter(torch.tensor([
            0.2, 0.2,
            0.8, 0.3,
            0.3, 0.8,
        ]))
        self.out_dim = 1


class QuadraticBezierIntegrandSlang(BaseIntegrandSlang):
    def __init__(self):
        super().__init__()
        self.shader = slangtorch.loadModule("slang/__gen__bezier_curve.slang")
        self.p = nn.Parameter(torch.tensor([
            0.2, 0.2,
            0.5, 0.8,
            0.8, 0.2,
            0.02,
        ]))
        self.out_dim = 1


class VoronoiGridIntegrandSlang(BaseIntegrandSlangRGB):
    def __init__(self, n=3, jitter=0.08):
        super().__init__()
        self.shader = slangtorch.loadModule("slang/__gen__voronoi_simple.slang")
        params = []
        torch.manual_seed(42)
        for i in range(n):
            for j in range(n):
                base_x = (i + 0.5) / n
                base_y = (j + 0.5) / n
                offset_x = (torch.rand(1).item() - 0.5) * jitter / n
                offset_y = (torch.rand(1).item() - 0.5) * jitter / n
                r, g, b = torch.rand(3).tolist()
                params.extend([base_x + offset_x, base_y + offset_y, r, g, b])
        self.p = nn.Parameter(torch.tensor(params, dtype=torch.float32))
        self.out_dim = 3


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


class SweptBrushIntegrandSlang(BaseIntegrandSlang):
    def __init__(self, bristle_params, n_bristles=None, start_step=0, n_steps=95, draw_path=False):
        super().__init__()
        defines = {"DRAW_PATH": 1} if draw_path else {}
        self.shader = slangtorch.loadModule("slang/__gen__swept_brush.slang", defines=defines)

        bristle_params = torch.as_tensor(bristle_params, dtype=torch.float32)
        if bristle_params.ndim != 2 or bristle_params.shape[1] != 3:
            raise ValueError("bristle_params must be (N, 3) for (center offset, radius offset, thickness).")
        if n_bristles is None:
            n_bristles = bristle_params.shape[0]
        if bristle_params.shape[0] != n_bristles:
            raise ValueError("n_bristles must match bristle_params rows.")

        params = torch.zeros(3 + 3 * n_bristles, dtype=torch.float32)
        params[0] = float(n_bristles)
        params[1] = float(start_step)
        params[2] = float(n_steps)
        params[3:] = bristle_params.reshape(-1)

        self.p = nn.Parameter(params)
        self.out_dim = 1


class SweptBilinearIntegrandSlang(BaseIntegrandSlang):
    def __init__(self, grid_size=512, start_step=0, n_steps=95, grid_params=None, seed=40):
        super().__init__()
        defines = {"SINGLE_STEP": 1} if n_steps == 1 else {"N_STEPS": int(n_steps)}
        self.shader = slangtorch.loadModule("slang/__gen__swept_bilinear.slang", defines=defines)

        if seed is not None:
            torch.manual_seed(seed)
        params = torch.zeros(3 + grid_size * grid_size, dtype=torch.float32)
        params[0] = float(grid_size)
        params[1] = float(start_step)
        params[2] = float(n_steps)

        if grid_params is not None:
            grid_tensor = torch.as_tensor(grid_params, dtype=torch.float32)
            if grid_tensor.numel() != grid_size * grid_size:
                raise ValueError("grid_params must have grid_size * grid_size entries.")
            params[3:] = grid_tensor.reshape(-1)
        else:
            params[3:] = (torch.rand_like(params[3:]) - 0.5) / 30.0

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


def _build_acceleration_structure(
    primitive_types,
    control_points,
    stroke_widths,
    grid_size,
    max_elements_per_cell,
    device,
    accel_setup_kernel,
):
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
        accel_setup_kernel.setup_unified_acceleration_structure(
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
        self._accel_setup_kernel = slangtorch.loadModule("slang/accel_structure_setup.slang")

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
            self._accel_setup_kernel,
        ).reshape(-1)

        header = torch.tensor(
            [float(self.num_primitives), float(self.grid_size), float(self.max_elements_per_cell)],
            device=device,
        )
        return torch.cat([header, self.background_color.to(device), result, grid])


def _projection_matrix(x=0.5, n=1.5, f=100.0):
    return np.array(
        [
            [n / x, 0.0, 0.0, 0.0],
            [0.0, n / -x, 0.0, 0.0],
            [0.0, 0.0, -(f + n) / (f - n), -(2.0 * f * n) / (f - n)],
            [0.0, 0.0, -1.0, 0.0],
        ],
        dtype=np.float32,
    )


def _translation_matrix(x, y, z):
    mat = np.eye(4, dtype=np.float32)
    mat[0, 3] = x
    mat[1, 3] = y
    mat[2, 3] = z
    return mat


def _random_rotation_matrix(rng):
    u1, u2, u3 = rng.random(3)
    qx = math.sqrt(1.0 - u1) * math.sin(math.tau * u2)
    qy = math.sqrt(1.0 - u1) * math.cos(math.tau * u2)
    qz = math.sqrt(u1) * math.sin(math.tau * u3)
    qw = math.sqrt(u1) * math.cos(math.tau * u3)
    rot = np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=np.float32,
    )
    mat = np.eye(4, dtype=np.float32)
    mat[:3, :3] = rot
    return mat


def _create_view_buffers(num_views, seed):
    rng = np.random.default_rng(seed if seed is not None else 0)
    proj = torch.from_numpy(_projection_matrix())
    view_mats = []
    light_dirs = []
    for _ in range(num_views):
        view = _translation_matrix(0.0, 0.0, -4.0) @ _random_rotation_matrix(rng)
        view_mats.append(view)
        campos = np.linalg.inv(view)[:3, 3]
        light = -campos / (np.linalg.norm(campos) + 1e-8)
        light_dirs.append(light.astype(np.float32))
    view_tensor = torch.from_numpy(np.stack(view_mats).astype(np.float32))
    mvps = proj @ view_tensor
    light_tensor = torch.from_numpy(np.stack(light_dirs).astype(np.float32))
    return mvps, light_tensor


def _load_ply_mesh(ply_path):
    plydata = plyfile.PlyData.read(str(ply_path))
    vertex_data = plydata["vertex"]
    vertices = np.vstack([vertex_data["x"], vertex_data["y"], vertex_data["z"]]).T.astype(np.float32)
    faces = np.stack([face[0] for face in plydata["face"]], axis=0).astype(np.int64)
    return torch.from_numpy(vertices), torch.from_numpy(faces)


def _normalize_vertices(vertices, scale):
    mean = vertices.mean(dim=0, keepdim=True)
    centered = vertices - mean
    span = (centered.max() - centered.min()).clamp(min=1e-6)
    return centered * (scale / span)


def _rotation_from_quat(quat):
    quat = quat.clone()
    norm = quat.norm(dim=1, keepdim=True).clamp(min=1e-12)
    q = quat / norm
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = torch.zeros(quat.shape[0], 3, 3, dtype=quat.dtype, device=quat.device)
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - z * w)
    R[:, 0, 2] = 2 * (x * z + y * w)
    R[:, 1, 0] = 2 * (x * y + z * w)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - x * w)
    R[:, 2, 0] = 2 * (x * z - y * w)
    R[:, 2, 1] = 2 * (y * z + x * w)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def _build_scaling_rotation(scales, quat):
    R = _rotation_from_quat(quat)
    L = torch.zeros(scales.shape[0], 3, 3, dtype=scales.dtype, device=scales.device)
    L[:, 0, 0] = scales[:, 0]
    L[:, 1, 1] = scales[:, 1]
    L[:, 2, 2] = scales[:, 2]
    return torch.matmul(R, L)


_ROTATED_ELLIPSE_PRIM_TYPE = 5.0


class TriangleRasterizerIntegrandSlang(BaseIntegrandSlangRGB):
    """Triangle-mesh rasterizer backed by the vector-graphics shader."""

    def __init__(
        self,
        ply_file,
        grid_size=256,
        max_elements_per_cell=64,
        background_color=(0.0, 0.0, 0.0),
        scale=0.5,
        res=512,
        num_view=16,
        seed=42,
    ):
        super().__init__()
        self.shader = slangtorch.loadModule("slang/__gen__vector_graphics_rgb_padded_accel.slang")
        self.grid_size = grid_size
        self.max_elements_per_cell = max_elements_per_cell
        self.background_color = torch.tensor(background_color, dtype=torch.float32)
        self.scale = scale
        self.res = res
        self.num_views = max(1, num_view)
        self.lambda_ = 19.0
        self.out_dim = 3
        self.active_view = 0
        self._accel_setup_kernel = slangtorch.loadModule("slang/accel_structure_setup.slang")

        vertices, faces = _load_ply_mesh(ply_file)
        vertices = _normalize_vertices(vertices.float(), self.scale)
        faces = faces.long()
        self._assign_buffer("faces", faces, dtype=torch.long)
        self._update_mesh(vertices)

        mvps, light = _create_view_buffers(self.num_views, seed)
        self._assign_buffer("mvps", mvps)
        self._assign_buffer("lightdir", light)

    def _assign_buffer(self, name, tensor, dtype=None):
        if not torch.is_tensor(tensor):
            tensor = torch.as_tensor(tensor)
        target_dtype = dtype or tensor.dtype
        tensor = tensor.to(dtype=target_dtype, copy=True)
        if name in self._buffers:
            setattr(self, name, tensor)
        else:
            self.register_buffer(name, tensor)

    def _update_mesh(self, vertices):
        device = vertices.device
        faces = self.faces.to(device)
        self.num_primitives = faces.shape[0]
        base_colors = torch.ones(self.num_primitives, 3, dtype=torch.float32, device=device)
        self._assign_buffer("base_colors_tri", base_colors)
        M = compute_matrix(vertices, faces, self.lambda_)
        if isinstance(M, torch.Tensor):
            M = M.to(device)
        self._assign_buffer("mesh_matrix", M)
        self.u = nn.Parameter(to_differential(self.mesh_matrix, vertices))

    @property
    def vertices(self):
        return from_differential(self.mesh_matrix, self.u, "Cholesky")

    def reset_mesh(self, vertices, faces):
        device = self.u.device if hasattr(self, "u") else vertices.device
        face_tensor = faces.to(device=device, dtype=torch.long)
        vert_tensor = vertices.to(device=device, dtype=torch.float32)
        self._assign_buffer("faces", face_tensor, dtype=torch.long)
        self._update_mesh(vert_tensor)

    def set_active_view(self, idx):
        self.active_view = int(idx) % self.num_views

    @property
    def p(self):
        device = self.u.device
        if self.num_primitives == 0:
            grid = torch.zeros(
                self.grid_size * self.grid_size * (self.max_elements_per_cell + 1),
                device=device,
                dtype=torch.float32,
            )
            header = torch.tensor(
                [0.0, float(self.grid_size), float(self.max_elements_per_cell)],
                device=device,
            )
            return torch.cat([header, self.background_color.to(device), grid])

        if not (0 <= self.active_view < self.num_views):
            raise ValueError(f"Active view {self.active_view} out of range (0-{self.num_views - 1}).")

        vertices = self.vertices
        v_hom = F.pad(vertices, (0, 1), value=1.0)
        view = self.mvps[self.active_view].transpose(0, 1)
        v_ndc = torch.matmul(v_hom, view)
        triangles = v_ndc[self.faces]
        world_triangles = vertices[self.faces]
        normals = torch.cross(world_triangles[:, 0] - world_triangles[:, 1], world_triangles[:, 2] - world_triangles[:, 1])
        normals = normals / (normals.norm(dim=-1, keepdim=True) + 1e-10)
        z_depth = triangles.mean(dim=1)[:, 2]
        order = torch.argsort(z_depth)
        sorted_triangles = triangles[order]
        sorted_world_normals = normals[order]
        sorted_distances = z_depth[order]
        control_points = (sorted_triangles[..., :2].reshape(-1, 6) + 1.0) * 0.5

        light = self.lightdir[self.active_view]
        shading = (-light.view(1, 3) * sorted_world_normals).sum(dim=-1, keepdim=True).clamp(min=0.0)
        base_colors = self.base_colors_tri[order]
        active_colors = base_colors * shading

        result = torch.zeros(self.num_primitives * 20, device=device)
        idx = torch.arange(self.num_primitives, device=device)
        result[idx * 20] = 1.0
        control_idx = idx.unsqueeze(1) * 20 + torch.arange(1, 7, device=device)
        result.index_put_((control_idx.view(-1),), control_points.view(-1))
        result[idx * 20 + 7] = 0.0
        result[idx * 20 + 8] = 0.0
        color_idx = idx.unsqueeze(1) * 20 + torch.arange(9, 12, device=device)
        result.index_put_((color_idx.view(-1),), active_colors.view(-1))
        result[idx * 20 + 12] = 1.0

        grid = _build_acceleration_structure(
            torch.ones(self.num_primitives, device=device),
            control_points,
            torch.zeros(self.num_primitives, device=device),
            self.grid_size,
            self.max_elements_per_cell,
            device,
            self._accel_setup_kernel,
        )
        grid_int = grid.to(torch.int64)
        counts = grid_int[..., 0].clamp_(min=0, max=self.max_elements_per_cell)
        elements = grid_int[..., 1:]
        max_cell = elements.shape[-1]
        flat_elements = elements.view(-1, max_cell)
        flat_counts = counts.view(-1)
        idx_range = torch.arange(max_cell, device=device).unsqueeze(0)
        valid_mask = idx_range < flat_counts.unsqueeze(1)
        flat_elements[~valid_mask] = -1
        depths = torch.full_like(elements, float("inf"), dtype=torch.float32)
        flat_depths = depths.view(-1, max_cell)
        if valid_mask.any():
            valid_elements = flat_elements[valid_mask].long()
            flat_depths[valid_mask] = -sorted_distances[valid_elements]
        sorted_idx = torch.argsort(depths, dim=-1)
        sorted_elements = torch.gather(elements, -1, sorted_idx)
        grid_int[..., 1:] = sorted_elements
        grid_int[..., 0] = counts
        grid_flat = grid_int.to(torch.float32).reshape(-1)

        header = torch.tensor(
            [float(self.num_primitives), float(self.grid_size), float(self.max_elements_per_cell)],
            device=device,
        )
        return torch.cat([header, self.background_color.to(device), result, grid_flat])


class EllipsoidRasterizerIntegrandSlang(BaseIntegrandSlangRGB):
    """Projects 3D ellipsoids to screen-space ellipses and renders them via the vector shader."""

    def __init__(
        self,
        num_primitives=10000,
        grid_size=256,
        max_elements_per_cell=64,
        background_color=(0.0, 0.0, 0.0),
        center_scale=0.5,
        res=512,
        num_view=16,
        ellipsoid_radius=1e-3,
        seed=42,
    ):
        super().__init__()
        self.shader = slangtorch.loadModule("slang/__gen__vector_graphics_rgb_padded_accel.slang")
        self.grid_size = grid_size
        self.max_elements_per_cell = max_elements_per_cell
        self.background_color = torch.tensor(background_color, dtype=torch.float32)
        self.resolution = res
        self.num_views = max(1, num_view)
        self.out_dim = 3
        self.active_view = 0
        self._accel_setup_kernel = slangtorch.loadModule("slang/accel_structure_setup.slang")

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        init_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        centers = (torch.rand(num_primitives, 3, device=init_device) - 0.5) / center_scale
        scales = ellipsoid_radius * torch.rand(num_primitives, 3, device=init_device)
        rotations = torch.ones(num_primitives, 4, device=init_device)
        rotations[:, 0] = 1.0
        colors = torch.ones(num_primitives, 3, device=init_device)
        opacities = torch.ones(num_primitives, device=init_device) / 3.0

        self.ellipsoid_centers = nn.Parameter(centers)
        self.ellipsoid_scales = nn.Parameter(scales)
        self.ellipsoid_rotations = nn.Parameter(rotations)
        self.fill_colors = nn.Parameter(colors)
        self.opacities = nn.Parameter(opacities)

        mvps, light = _create_view_buffers(self.num_views, seed)
        self.register_buffer("mvps", mvps)
        self.register_buffer("lightdir", light)
        e = 1.5 / 0.5
        focal = (self.resolution / 2.0) / (1.0 / e)
        self.register_buffer("focal_length", torch.tensor(focal, dtype=torch.float32))

    @property
    def num_primitives(self):
        return self.ellipsoid_centers.shape[0]

    def set_active_view(self, idx):
        self.active_view = int(idx) % self.num_views

    def _empty_param_block(self, device):
        grid = torch.zeros(
            self.grid_size * self.grid_size * (self.max_elements_per_cell + 1),
            device=device,
            dtype=torch.float32,
        )
        header = torch.tensor(
            [0.0, float(self.grid_size), float(self.max_elements_per_cell)],
            device=device,
        )
        return torch.cat([header, self.background_color.to(device), grid])

    @property
    def p(self):
        device = self.ellipsoid_centers.device
        total = self.num_primitives
        if total == 0:
            return self._empty_param_block(device)
        if not (0 <= self.active_view < self.num_views):
            raise ValueError(f"Active view {self.active_view} out of range (0-{self.num_views - 1}).")

        ones = torch.ones(total, 1, device=device, dtype=self.ellipsoid_centers.dtype)
        centers_h = torch.cat([self.ellipsoid_centers, ones], dim=1)
        view = self.mvps[self.active_view].transpose(0, 1)
        centers_ndc = torch.matmul(centers_h, view)
        valid_mask = centers_ndc[:, 2] > 0.0
        if not valid_mask.any():
            return self._empty_param_block(device)

        centers_ndc = centers_ndc[valid_mask]
        colors = self.fill_colors[valid_mask]
        opacities = self.opacities[valid_mask].clamp(0.0, 1.0)
        scales = self.ellipsoid_scales[valid_mask].clamp(min=1e-9)
        rotations = self.ellipsoid_rotations[valid_mask]
        centers_screen = (centers_ndc[:, :2] + 1.0) * 0.5

        L = _build_scaling_rotation(scales, rotations)
        cov = torch.matmul(L, L.transpose(1, 2))
        view_matrix = self.mvps[self.active_view][:3, :3]
        camera_cov = torch.matmul(view_matrix, torch.matmul(cov, view_matrix.transpose(0, 1)))
        screen_cov = camera_cov[:, :2, :2]
        z = centers_ndc[:, 2].clamp(min=1e-6)
        scale = (self.focal_length.to(device) / z).view(-1, 1, 1)
        screen_cov = screen_cov * (scale * scale)

        order = torch.argsort(z)
        centers_sorted = centers_screen[order]
        cov_sorted = screen_cov[order]
        colors_sorted = colors[order]
        opacities_sorted = opacities[order]
        z_sorted = z[order]
        num_visible = centers_sorted.shape[0]
        if num_visible == 0:
            return self._empty_param_block(device)

        zero_pad = torch.zeros(num_visible, 1, device=device, dtype=centers_sorted.dtype)
        control_points = torch.cat(
            [
                centers_sorted,
                cov_sorted[:, 0, 0].unsqueeze(1),
                cov_sorted[:, 0, 1].unsqueeze(1),
                cov_sorted[:, 1, 1].unsqueeze(1),
                zero_pad,
            ],
            dim=1,
        )

        result = torch.zeros(num_visible * 20, device=device)
        idx = torch.arange(num_visible, device=device)
        result[idx * 20] = _ROTATED_ELLIPSE_PRIM_TYPE
        control_idx = idx.unsqueeze(1) * 20 + torch.arange(1, 7, device=device)
        result.index_put_((control_idx.view(-1),), control_points.view(-1))
        color_idx = idx.unsqueeze(1) * 20 + torch.arange(9, 12, device=device)
        result.index_put_((color_idx.view(-1),), colors_sorted.view(-1))
        result[idx * 20 + 12] = opacities_sorted

        grid = _build_acceleration_structure(
            torch.full((num_visible,), _ROTATED_ELLIPSE_PRIM_TYPE, device=device),
            control_points,
            torch.zeros(num_visible, device=device),
            self.grid_size,
            self.max_elements_per_cell,
            device,
            self._accel_setup_kernel,
        ).reshape(-1)

        header = torch.tensor(
            [float(num_visible), float(self.grid_size), float(self.max_elements_per_cell)],
            device=device,
        )
        return torch.cat([header, self.background_color.to(device), result, grid])
