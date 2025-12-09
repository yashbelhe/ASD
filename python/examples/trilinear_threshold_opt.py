"""
Port of the legacy `trilinear_threshold_opt.py` experiment.

Optimizes a trilinear grid (thresholded to binary output) to match various 3D
targets (analytic SDFs or meshes) using the Slang shader + boundary loss.
"""

import argparse
from pathlib import Path
import sys
from typing import Callable, Tuple

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import torch
import torch.optim as optim

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))

from compiler.compile_shader import compile_if_needed  # noqa: E402
from python.libigl_inside_outside import MeshInsideOutsideTest  # noqa: E402
from python.helpers import (  # noqa: E402
    BoundaryLossConfig,
    boundary_loss_slang,
    points_on_grid,
)
from python.integrands import TrilinearThresholdIntegrandSlang  # noqa: E402


def create_3d_slice_samples(resolution, aa_factor, plane_value, constant_axis=2, device=None):
    base = points_on_grid(resolution * aa_factor, jitter=True).to(device)
    if constant_axis == 0:
        samples = torch.cat([torch.full_like(base[:, :1], plane_value), base], dim=1)
        axis_name = "x"
    elif constant_axis == 1:
        samples = torch.cat([base[:, :1], torch.full_like(base[:, :1], plane_value), base[:, 1:]], dim=1)
        axis_name = "y"
    else:
        samples = torch.cat([base, torch.full_like(base[:, :1], plane_value)], dim=1)
        axis_name = "z"
    return samples, axis_name


def random_sphere(points, center, radius):
    return ((points - torch.tensor(center, device=points.device))**2).sum(dim=1).le(radius**2).float()


def csg_random_spheres(points, num=5, seed=42, min_radius=0.02, max_radius=0.05):
    torch.manual_seed(seed)
    result = torch.zeros(points.shape[0], device=points.device)
    for _ in range(num):
        center = torch.rand(3, device=points.device)
        radius = min_radius + (max_radius - min_radius) * torch.rand(1, device=points.device)
        sphere = random_sphere(points, center, radius)
        if torch.rand(1).item() > 0.3:
            result = torch.maximum(result, sphere)
        else:
            result = torch.clamp(result - sphere, 0, 1)
    return result


def torus_sdf(points, center=(0.5, 0.5, 0.5), ring_radius=0.3, tube_radius=0.1):
    center_t = torch.tensor(center, device=points.device)
    p = points - center_t
    xz = torch.sqrt(p[:, 0] ** 2 + p[:, 2] ** 2) - ring_radius
    q = torch.stack([xz, p[:, 1]], dim=1)
    dist = torch.linalg.norm(q, dim=1) - tube_radius
    return (dist <= 0).float()


def swept_torus(points, num_steps=50, spiral_radius=0.2, height=0.8, revolutions=3.0, ring_radius=0.15, tube_radius=0.05):
    result = torch.zeros(points.shape[0], device=points.device)
    t_vals = torch.linspace(0.0, 1.0, num_steps, device=points.device)
    path_y = 0.1 + height * t_vals
    angles = 2 * torch.pi * revolutions * t_vals
    path_x = 0.5 + spiral_radius * torch.cos(angles)
    path_z = 0.5 + spiral_radius * torch.sin(angles)
    for px, py, pz in zip(path_x, path_y, path_z):
        center = (px.item(), py.item(), pz.item())
        result = torch.maximum(result, torus_sdf(points, center=center, ring_radius=ring_radius, tube_radius=tube_radius))
    return result


def plot_slice_comparison(target, current, plane_value, axis_name, out_path):
    diff = (current - target).abs()
    plt.figure(figsize=(12, 4))
    for idx, (title, data) in enumerate(
        [
            (f"Target ({axis_name}={plane_value:.3f})", target),
            ("Current", current),
            ("|diff|", diff),
        ]
    ):
        plt.subplot(1, 3, idx + 1)
        plt.imshow(data.detach().cpu().numpy(), origin="lower", cmap="gray")
        plt.title(title)
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_checkpoint(path, integrand, optimizer, loss_history, edge_mode):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_params": integrand.p.detach().cpu(),
            "optimizer_state": optimizer.state_dict(),
            "loss_history": loss_history,
            "edge_loss_mode": edge_mode,
        },
        path,
    )


def make_gif(frames_dir, duration):
    frames = sorted(frames_dir.glob("iter_*.png"))
    if not frames:
        return
    images = [imageio.imread(frame) for frame in frames]
    imageio.mimsave(frames_dir / "slices.gif", images, duration=duration)


def build_target_fn(name: str, device: torch.device, mesh_path: Path = None) -> Tuple[Callable[[torch.Tensor], torch.Tensor], str]:
    lower = name.lower()
    if lower == "circle":
        return lambda x: random_sphere(x, center=(0.5, 0.5, 0.5), radius=0.4), "random_circle"
    if lower == "csg":
        return lambda x: csg_random_spheres(x, num=30, seed=123, min_radius=0.05, max_radius=0.15), "csg_random_spheres"
    if lower == "torus":
        return lambda x: torus_sdf(x, center=(0.5, 0.5, 0.5), ring_radius=0.3, tube_radius=0.1), "torus"
    if lower == "swept_torus":
        return lambda x: swept_torus(x, num_steps=200, spiral_radius=0.2, height=0.8, revolutions=3.0, ring_radius=0.15, tube_radius=0.03), "swept_torus"
    if lower == "mesh":
        if mesh_path is None:
            raise ValueError("--mesh-path must be provided when --target mesh is selected.")
        tester = MeshInsideOutsideTest(str(mesh_path))
        return lambda x: tester(x), mesh_path.stem
    raise ValueError(f"Unknown target '{name}'.")


def parse_args():
    parser = argparse.ArgumentParser(description="Optimize a trilinear grid to match 3D targets.")
    parser.add_argument("--target", choices=["circle", "csg", "torus", "swept_torus", "mesh"], default="mesh")
    parser.add_argument("--mesh-path", type=str, default="data/genus6.ply")
    parser.add_argument("--grid-n", type=int, default=200)
    parser.add_argument("--gt-resolution", type=int, default=512)
    parser.add_argument("--aa-train", type=int, default=1)
    parser.add_argument("--aa-slice", type=int, default=1)
    parser.add_argument("--num-iter", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--sample-res", type=int, default=128, help="Resolution of jittered volume samples per iteration.")
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--edge-sample-res", type=int, default=270)
    parser.add_argument("--edge-num-subdivision", type=int, default=20)
    parser.add_argument("--edge-kde-k", type=int, default=14)
    parser.add_argument("--edge-div-eps", type=float, default=1e-15)
    parser.add_argument("--edge-plot-resolution", type=int, default=1000)
    parser.add_argument("--edge-mode", choices=["L2_test_fn", "L2_img", "L1_img"], default="L2_test_fn")
    parser.add_argument("--df-dx-mode", choices=["forward", "backward"], default="backward")
    parser.add_argument("--pixel-weight", type=float, default=0.0)
    parser.add_argument("--slice-axis", type=int, default=2, choices=[0, 1, 2])
    parser.add_argument("--num-slices", type=int, default=5)
    parser.add_argument("--gif-duration", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=40)
    parser.add_argument("--results-name", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    src = repo_root / "slang" / "trilinear_thresholding.slang"
    dst = repo_root / "slang" / "__gen__trilinear_thresholding.slang"
    compile_if_needed(src, dst)

    mesh_path = Path(args.mesh_path)
    if args.target == "mesh":
        if not mesh_path.is_absolute():
            candidates = [mesh_path, repo_root / mesh_path, repo_root.parent / mesh_path]
            for cand in candidates:
                if cand.exists():
                    mesh_path = cand
                    break
            else:
                raise FileNotFoundError(f"Mesh file '{args.mesh_path}' not found.")

    integrand = TrilinearThresholdIntegrandSlang(grid_size=args.grid_n, seed=args.seed).to(device)
    target_fn, target_tag = build_target_fn(args.target, device, mesh_path if args.target == "mesh" else None)

    run_name = args.results_name or f"{target_tag}_n{args.grid_n}_iter{args.num_iter}"
    results_dir = repo_root / "results" / "trilinear_threshold" / run_name
    results_dir.mkdir(parents=True, exist_ok=True)

    boundary_cfg = BoundaryLossConfig(
        dim=3,
        grid_size=args.edge_sample_res,
        num_subdivision=args.edge_num_subdivision,
        lipschitz_bounds=1e-6,
        div_eps=args.edge_div_eps,
        plot_resolution=args.edge_plot_resolution,
        kde_k=args.edge_kde_k,
        mode=args.edge_mode,
        mode_aux_data=target_fn if args.edge_mode == "L2_test_fn" else None,
        df_dx_mode=args.df_dx_mode,
    )

    loss_history = []
    optimizer = optim.Adam(integrand.parameters(), lr=args.lr)

    def render_slice(plane):
        samples, axis = create_3d_slice_samples(args.gt_resolution, args.aa_slice, plane, args.slice_axis, device=device)
        preds = integrand(samples).reshape(args.gt_resolution, args.aa_slice, args.gt_resolution, args.aa_slice).mean(dim=(1, 3))
        target = target_fn(samples).reshape(args.gt_resolution, args.aa_slice, args.gt_resolution, args.aa_slice).mean(dim=(1, 3))
        return target, preds, axis

    initial_target, initial_pred, axis_name = render_slice(0.5)
    plt.imsave(results_dir / "initial_slice.png", initial_pred.detach().cpu().numpy(), origin="lower", cmap="gray")
    plot_slice_comparison(initial_target.detach(), initial_pred.detach(), 0.5, axis_name, results_dir / "initial_comparison.png")

    for step in range(args.num_iter):
        optimizer.zero_grad()
        pts = points_on_grid(args.sample_res, jitter=True, dim=3).to(device)
        preds = integrand(pts)
        target_vals = target_fn(pts)
        pixel_loss = (preds - target_vals).square().mean()
        boundary_loss = boundary_loss_slang(integrand, boundary_cfg)
        total_loss = args.pixel_weight * pixel_loss + boundary_loss
        total_loss.backward()
        optimizer.step()

        loss_history.append(pixel_loss.item())
        if step % args.log_every == 0:
            print(f"Iter {step:04d} | pixel={pixel_loss.item():.6f} | boundary={boundary_loss.item():.6f}")

        if step % args.save_every == 0 or step == args.num_iter - 1:
            plane = 0.5
            target_slice, pred_slice, axis_name = render_slice(plane)
            save_path = results_dir / f"iter_{step:04d}.png"
            plt.imsave(save_path, pred_slice.detach().cpu().numpy(), origin="lower", cmap="gray")
            plot_slice_comparison(target_slice.detach(), pred_slice.detach(), plane, axis_name, results_dir / f"comparison_{step:04d}.png")
            save_checkpoint(results_dir / f"checkpoint_{step:04d}.pt", integrand, optimizer, loss_history, args.edge_mode)

    torch.save(integrand.p.detach().cpu(), results_dir / "final_params.pt")

    planes = torch.linspace(0.0, 1.0, args.num_slices)
    for plane in planes:
        target_slice, pred_slice, axis_name = render_slice(plane.item())
        plot_slice_comparison(target_slice.detach(), pred_slice.detach(), plane.item(), axis_name, results_dir / f"slice_{axis_name}_{plane:.3f}.png")

    make_gif(results_dir, args.gif_duration)


if __name__ == "__main__":
    main()
