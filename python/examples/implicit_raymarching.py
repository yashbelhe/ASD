"""
Implicit grid raymarching optimization example.

Matches renders of a volumetric implicit function (by default a sphere) by
optimizing a trilinear grid using boundary + area losses across multiple
camera views.
"""

import argparse
from pathlib import Path
import sys
from typing import Optional

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import torch
import torch.optim as optim

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))

from compiler.compile_shader import compile_if_needed  # noqa: E402
from python.helpers import BoundaryLossConfig, boundary_loss_slang, points_on_grid  # noqa: E402
from python.integrands import ImplicitRaymarchingIntegrandSlang  # noqa: E402


def create_sphere_grid(n, center=(0.5, 0.5, 0.5), radius=0.3, device="cpu"):
    lin = torch.linspace(0, 1, n, device=device)
    grid_x, grid_y, grid_z = torch.meshgrid(lin, lin, lin, indexing="ij")
    dist = torch.sqrt((grid_x - center[0]) ** 2 + (grid_y - center[1]) ** 2 + (grid_z - center[2]) ** 2)
    return (radius - dist).unsqueeze(-1).squeeze(-1)


def render_image(integrand, resolution, aa, jitter, device):
    samples = points_on_grid(resolution * aa, jitter=jitter).to(device)
    preds = integrand(samples).reshape(resolution, aa, resolution, aa, integrand.out_dim)
    return preds.mean(dim=(1, 3))


def save_rgb(tensor, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = tensor.detach().cpu().numpy().clip(0.0, 1.0)
    plt.imsave(path, arr, origin="lower")


def plot_comparison(target, image, out_path):
    diff = (image - target).abs()
    plt.figure(figsize=(12, 4))
    for idx, (title, data) in enumerate(
        [("Target", target), ("Current", image), ("|Diff|", diff)]
    ):
        plt.subplot(1, 3, idx + 1)
        plt.imshow(data.detach().cpu().numpy().clip(0.0, 1.0), origin="lower")
        plt.title(title)
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def create_gif(results_dir, duration):
    frames = sorted(results_dir.glob("iter_*.png"))
    if not frames:
        return
    images = [imageio.imread(frame) for frame in frames]
    imageio.mimsave(results_dir / "optimization.gif", images, duration=duration)


def load_target_grid(path: Optional[Path], grid_size: int, device: torch.device):
    if path is None:
        return create_sphere_grid(grid_size, device=device)
    grid_path = path
    if not grid_path.is_absolute():
        candidates = [grid_path, repo_root / grid_path, repo_root.parent / grid_path]
        for cand in candidates:
            if cand.exists():
                grid_path = cand
                break
        else:
            raise FileNotFoundError(f"Target grid file '{path}' not found.")
    data = torch.load(grid_path)
    if data.ndim == 1:
        total = data.shape[0]
        voxels = grid_size ** 3
        if total == voxels + 3:
            grid = -data[1 : 1 + voxels].view(grid_size, grid_size, grid_size)
            return grid.to(device)
        if total == voxels:
            grid = data.view(grid_size, grid_size, grid_size)
            return grid.to(device)
    if data.ndim == 3 and data.shape[0] == grid_size:
        return data.to(device)
    raise ValueError(f"Unexpected target grid shape {data.shape}; expected flat vector of length {grid_size**3} or {grid_size**3 + 3}.")


def parse_args():
    parser = argparse.ArgumentParser(description="Implicit raymarching grid optimization.")
    parser.add_argument("--grid-size", type=int, default=64)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--num-view", type=int, default=8)
    parser.add_argument("--target-grid", type=str, default=None, help="Optional .pt file with target grid values.")
    parser.add_argument("--init-sphere-radius", type=float, default=0.3)
    parser.add_argument("--gt-resolution", type=int, default=256)
    parser.add_argument("--aa-target", type=int, default=4)
    parser.add_argument("--aa-train", type=int, default=1)
    parser.add_argument("--aa-eval", type=int, default=1)
    parser.add_argument("--num-iter", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--gif-duration", type=float, default=0.2)
    parser.add_argument("--edge-sample-res", type=int, default=1000)
    parser.add_argument("--edge-num-subdivision", type=int, default=20)
    parser.add_argument("--edge-kde-k", type=int, default=14)
    parser.add_argument("--edge-div-eps", type=float, default=1e-15)
    parser.add_argument("--edge-plot-resolution", type=int, default=1000)
    parser.add_argument("--df-dx-mode", choices=["forward", "backward"], default="backward")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--results-name", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    src = repo_root / "slang" / "implicit_raymarching.slang"
    dst = repo_root / "slang" / "__gen__implicit_raymarching.slang"
    compile_if_needed(src, dst)

    target_grid = load_target_grid(Path(args.target_grid) if args.target_grid else None, args.grid_size, device)
    threshold = args.threshold
    integrand_target = ImplicitRaymarchingIntegrandSlang(
        grid_size=args.grid_size,
        threshold=threshold,
        grid_values=target_grid,
        num_view=args.num_view,
    ).to(device)

    init_grid = create_sphere_grid(args.grid_size, radius=args.init_sphere_radius, device=device)
    integrand = ImplicitRaymarchingIntegrandSlang(
        grid_size=args.grid_size,
        threshold=threshold,
        grid_values=init_grid,
        num_view=args.num_view,
    ).to(device)

    run_name = args.results_name or f"implicit_grid{args.grid_size}_iter{args.num_iter}"
    results_dir = repo_root / "results" / "implicit_raymarching" / run_name
    results_dir.mkdir(parents=True, exist_ok=True)

    target_views = []
    for view_idx in range(args.num_view):
        integrand_target.set_active_view(view_idx)
        target_img = render_image(integrand_target, args.gt_resolution, args.aa_target, jitter=False, device=device)
        target_views.append(target_img.detach())
        if view_idx == 0:
            save_rgb(target_img, results_dir / "target.png")

    integrand.set_active_view(0)
    initial_img = render_image(integrand, args.gt_resolution, args.aa_eval, jitter=False, device=device)
    save_rgb(initial_img, results_dir / "initial.png")
    save_rgb(initial_img, results_dir / "iter_0000.png")
    plot_comparison(target_views[0], initial_img, results_dir / "initial_comparison.png")

    boundary_cfg = BoundaryLossConfig(
        grid_size=args.edge_sample_res,
        num_subdivision=args.edge_num_subdivision,
        div_eps=args.edge_div_eps,
        plot_resolution=args.edge_plot_resolution,
        kde_k=args.edge_kde_k,
        mode="L2_img",
        mode_aux_data=None,
        df_dx_mode=args.df_dx_mode,
    )

    optimizer = optim.Adam([integrand.grid_values], lr=args.lr)
    loss_history = []

    for step in range(args.num_iter):
        view_idx = step % args.num_view
        integrand.set_active_view(view_idx)

        preds = render_image(integrand, args.gt_resolution, args.aa_train, jitter=True, device=device)
        target_img = target_views[view_idx]
        area_loss = (preds - target_img).square().mean()

        boundary_cfg.mode_aux_data = target_img.detach()
        boundary_loss = boundary_loss_slang(integrand, boundary_cfg)
        total_loss = area_loss + boundary_loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        loss_history.append(area_loss.item())
        if step % args.log_every == 0:
            print(f"Iter {step:04d} | loss={total_loss.item():.6f}")

        if step % args.save_every == 0 or step == args.num_iter - 1:
            integrand.set_active_view(0)
            eval_img = render_image(integrand, args.gt_resolution, args.aa_eval, jitter=False, device=device)
            save_rgb(eval_img, results_dir / f"iter_{step:04d}.png")
            plot_comparison(target_views[0], eval_img, results_dir / f"comparison_{step:04d}.png")

    torch.save(integrand.grid_values.detach().cpu(), results_dir / "final_grid.pt")

    plt.figure()
    plt.title("Area Loss History")
    plt.semilogy(loss_history)
    plt.xlabel("Iteration")
    plt.ylabel("Area Loss")
    plt.grid(True, which="both")
    plt.savefig(results_dir / "loss_history.png", dpi=300, bbox_inches="tight")
    plt.close()

    create_gif(results_dir, args.gif_duration)


if __name__ == "__main__":
    main()
