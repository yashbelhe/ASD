"""Multi-view ellipsoid fitting example ported from the legacy repository."""

import argparse
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))

from compiler.compile_shader import compile_if_needed  # noqa: E402
from python.helpers import (  # noqa: E402
    BoundaryLossConfig,
    boundary_loss_slang,
    points_on_grid,
)
from python.integrands import (  # noqa: E402
    EllipsoidRasterizerIntegrandSlang,
    TriangleRasterizerIntegrandSlang,
)


def plot_comparison(target, current, initial=None, save_path=None):
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(target.detach().cpu().numpy(), extent=[0, 1, 0, 1], origin="lower")
    plt.title("Target")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(current.detach().cpu().numpy(), extent=[0, 1, 0, 1], origin="lower")
    plt.title("Current")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    if initial is None:
        diff = (current - target).abs().detach().cpu().numpy()
        plt.imshow(diff, extent=[0, 1, 0, 1], origin="lower")
        plt.title("Difference")
    else:
        plt.imshow(initial.detach().cpu().numpy(), extent=[0, 1, 0, 1], origin="lower")
        plt.title("Initial")
    plt.axis("off")
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def render_image(integrand, resolution, aa_factor, jitter, device):
    samples = points_on_grid(resolution * aa_factor, jitter=jitter).to(device)
    preds = integrand(samples)
    return preds.reshape(
        resolution,
        aa_factor,
        resolution,
        aa_factor,
        integrand.out_dim,
    ).mean(dim=(1, 3))


def save_tensor_image(path, tensor):
    plt.imsave(path, tensor.detach().cpu().numpy().clip(0, 1), origin="lower", dpi=300)


def resolve_mesh_path(identifier: str) -> Path:
    presets = {
        "bunny": repo_root.parent / "data" / "scenes" / "bunny" / "meshes" / "target.ply",
        "dragon": repo_root.parent / "data" / "scenes" / "dragon" / "meshes" / "target.ply",
        "suzanne": repo_root.parent / "data" / "scenes" / "suzanne" / "meshes" / "target.ply",
    }
    candidate = Path(identifier)
    if candidate.exists():
        return candidate
    key = identifier.lower()
    if key in presets:
        return presets[key]
    raise FileNotFoundError(f"Could not find mesh '{identifier}'.")


def build_optimizer(model, args):
    beta1 = args.beta1
    beta2 = 1 - (1 - beta1) ** 2
    groups = [
        {"params": model.ellipsoid_centers, "lr": args.center_lr, "name": "center"},
        {"params": model.ellipsoid_scales, "lr": args.scale_lr, "name": "scale"},
        {"params": model.ellipsoid_rotations, "lr": args.rotation_lr, "name": "rotation"},
        {"params": model.opacities, "lr": args.opacity_lr, "name": "opacity"},
    ]
    if args.color_lr > 0:
        groups.append({"params": model.fill_colors, "lr": args.color_lr, "name": "color"})
    return optim.Adam(groups, betas=(beta1, beta2))


def split_ellipsoids(model, args):
    grad = model.ellipsoid_centers.grad
    if grad is None:
        return 0
    mask = grad.abs().max(dim=1).values > args.split_grad_threshold
    count = int(mask.sum().item())
    if count == 0:
        return 0
    with torch.no_grad():
        centers = model.ellipsoid_centers.detach()
        scales = model.ellipsoid_scales.detach()
        rotations = model.ellipsoid_rotations.detach()
        colors = model.fill_colors.detach()
        opacities = model.opacities.detach()

        centers_clone = centers.clone()
        scales_clone = scales.clone()
        scales_clone[mask] /= args.split_factor

        jitter = torch.randn_like(centers[mask]) * args.split_center_jitter
        new_centers = centers[mask] + jitter
        new_scales = scales[mask] / args.split_factor
        new_rotations = rotations[mask]
        new_colors = colors[mask]
        new_opacities = opacities[mask]

        model.ellipsoid_centers = nn.Parameter(torch.cat([centers_clone, new_centers], dim=0))
        model.ellipsoid_scales = nn.Parameter(torch.cat([scales_clone, new_scales], dim=0))
        model.ellipsoid_rotations = nn.Parameter(torch.cat([rotations, new_rotations], dim=0))
        model.fill_colors = nn.Parameter(torch.cat([colors, new_colors], dim=0))
        model.opacities = nn.Parameter(torch.cat([opacities, new_opacities], dim=0))
    return count


def prune_ellipsoids(model, args):
    with torch.no_grad():
        opacities = model.opacities.detach()
        scales = model.ellipsoid_scales.detach()
        remove_mask = (opacities < args.opacity_threshold) | (scales < args.scale_threshold).any(dim=-1)
        keep_mask = ~remove_mask
        keep_count = int(keep_mask.sum().item())
        if keep_count == model.num_primitives or keep_count == 0:
            return 0
        model.ellipsoid_centers = nn.Parameter(model.ellipsoid_centers.detach()[keep_mask])
        model.ellipsoid_scales = nn.Parameter(model.ellipsoid_scales.detach()[keep_mask])
        model.ellipsoid_rotations = nn.Parameter(model.ellipsoid_rotations.detach()[keep_mask])
        model.fill_colors = nn.Parameter(model.fill_colors.detach()[keep_mask])
        model.opacities = nn.Parameter(model.opacities.detach()[keep_mask])
        return int(remove_mask.sum().item())


def maybe_create_gif(run_dir: Path, output: Path, pattern: str = "iter_*.png"):
    frames = sorted(run_dir.glob(pattern))
    if not frames:
        return
    try:
        import imageio.v2 as imageio
    except ImportError:
        print("imageio not available; skipping GIF generation")
        return
    images = [imageio.imread(frame) for frame in frames]
    imageio.mimsave(output, images, duration=0.25)
    print(f"Saved GIF to {output}")


def parse_args():
    parser = argparse.ArgumentParser(description="Ellipsoid rasterization optimization.")
    parser.add_argument("--mesh", type=str, default="bunny", help="Preset name or path to target mesh.")
    parser.add_argument("--num-primitives", type=int, default=100000)
    parser.add_argument("--ellipsoid-radius", type=float, default=1e-6)
    parser.add_argument("--center-scale", type=float, default=2.0)
    parser.add_argument("--mesh-scale", type=float, default=0.5, help="Normalization scale for the target mesh.")
    parser.add_argument("--grid-size", type=int, default=512)
    parser.add_argument("--max-elements", type=int, default=200)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--num-views", type=int, default=100)
    parser.add_argument("--num-iter", type=int, default=3000)
    parser.add_argument("--aa-target", type=int, default=16)
    parser.add_argument("--aa-train", type=int, default=1)
    parser.add_argument("--aa-eval", type=int, default=16)
    parser.add_argument("--area-loss", choices=["L1", "L2"], default="L2")
    parser.add_argument("--edge-loss-mode", type=str, default="L1_img")
    parser.add_argument("--edge-grid", type=int, default=5000)
    parser.add_argument("--num-subdivision", type=int, default=20)
    parser.add_argument("--kde-k", type=int, default=14)
    parser.add_argument("--div-eps", type=float, default=1e-15)
    parser.add_argument("--plot-resolution", type=int, default=1000)
    parser.add_argument("--df-dx-mode", choices=["forward", "backward"], default="backward")
    parser.add_argument("--center-lr", type=float, default=5e-3)
    parser.add_argument("--center-lr-final", type=float, default=5e-5)
    parser.add_argument("--scale-lr", type=float, default=1e-7)
    parser.add_argument("--rotation-lr", type=float, default=1e-1)
    parser.add_argument("--color-lr", type=float, default=0.0)
    parser.add_argument("--opacity-lr", type=float, default=1e-2)
    parser.add_argument("--beta1", type=float, default=0.99)
    parser.add_argument("--split-interval", type=int, default=100)
    parser.add_argument("--split-stop", type=int, default=2000)
    parser.add_argument("--split-factor", type=float, default=2.0)
    parser.add_argument("--split-center-jitter", type=float, default=0.0)
    parser.add_argument("--split-grad-threshold", type=float, default=1e-9)
    parser.add_argument("--prune-interval", type=int, default=100)
    parser.add_argument("--prune-start", type=int, default=100)
    parser.add_argument("--opacity-threshold", type=float, default=0.01)
    parser.add_argument("--scale-threshold", type=float, default=1e-9)
    parser.add_argument("--save-interval", type=int, default=50)
    parser.add_argument("--preview-interval", type=int, default=10, help="Write comparison plots every N steps.")
    parser.add_argument("--state-interval", type=int, default=100)
    parser.add_argument("--results-dir", type=Path, default=repo_root / "results" / "ellipsoid")
    parser.add_argument("--background-color", type=float, nargs=3, default=(0.0, 0.0, 0.0))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--create-gif", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    mesh_path = resolve_mesh_path(args.mesh)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    shader = repo_root / "slang" / "vector_graphics_rgb_padded_accel.slang"
    generated = repo_root / "slang" / "__gen__vector_graphics_rgb_padded_accel.slang"
    try:
        compile_if_needed(shader, generated)
    except (ModuleNotFoundError, AssertionError) as exc:
        print(f"Warning: shader transformer unavailable ({exc}); using existing generated shader.")

    target = TriangleRasterizerIntegrandSlang(
        mesh_path,
        grid_size=args.grid_size,
        max_elements_per_cell=args.max_elements,
        scale=args.mesh_scale,
        res=args.resolution,
        num_view=args.num_views,
        seed=args.seed,
    ).to(device)
    target.eval()

    ellipsoids = EllipsoidRasterizerIntegrandSlang(
        num_primitives=args.num_primitives,
        grid_size=args.grid_size,
        max_elements_per_cell=args.max_elements,
        background_color=tuple(args.background_color),
        center_scale=args.center_scale,
        res=args.resolution,
        num_view=args.num_views,
        ellipsoid_radius=args.ellipsoid_radius,
        seed=args.seed,
    ).to(device)
    ellipsoids.train()

    samples = points_on_grid(args.resolution * args.aa_target, jitter=True).to(device)
    target_imgs = []
    with torch.no_grad():
        for view_idx in range(args.num_views):
            target.set_active_view(view_idx)
            img = target(samples)
            target_imgs.append(
                img.reshape(
                    args.resolution,
                    args.aa_target,
                    args.resolution,
                    args.aa_target,
                    target.out_dim,
                ).mean(dim=(1, 3)).detach()
            )

    ellipsoids.set_active_view(0)
    out_init = render_image(ellipsoids, args.resolution, args.aa_eval, True, device)

    run_name = f"{mesh_path.stem}_n{args.num_primitives}_v{args.num_views}"
    run_dir = args.results_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    save_tensor_image(run_dir / "initial.png", out_init)

    optimizer = build_optimizer(ellipsoids, args)
    cfg = BoundaryLossConfig(
        grid_size=args.edge_grid,
        num_subdivision=args.num_subdivision,
        div_eps=args.div_eps,
        plot_resolution=args.plot_resolution,
        kde_k=args.kde_k,
        mode=args.edge_loss_mode,
        df_dx_mode=args.df_dx_mode,
    )

    loss_history = []
    for step in range(args.num_iter):
        view_idx = step % args.num_views
        ellipsoids.set_active_view(view_idx)
        target_img = target_imgs[view_idx]
        pts = points_on_grid(args.resolution * args.aa_train, jitter=True).to(device)
        preds = ellipsoids(pts).reshape(
            args.resolution,
            args.aa_train,
            args.resolution,
            args.aa_train,
            ellipsoids.out_dim,
        ).mean(dim=(1, 3))

        if args.area_loss == "L1":
            area_loss = (preds - target_img).abs().mean()
        else:
            area_loss = (preds - target_img).square().mean()

        cfg.mode_aux_data = target_img.detach()
        edge_loss = boundary_loss_slang(ellipsoids, cfg)
        total = area_loss + edge_loss
        total.backward()

        split_count = 0
        prune_count = 0
        if (
            args.split_interval > 0
            and 0 < step < args.split_stop
            and step % args.split_interval == 0
        ):
            split_count = split_ellipsoids(ellipsoids, args)
            if split_count:
                optimizer = build_optimizer(ellipsoids, args)
        if args.prune_interval > 0 and step >= args.prune_start and step % args.prune_interval == 0:
            prune_count = prune_ellipsoids(ellipsoids, args)
            if prune_count:
                optimizer = build_optimizer(ellipsoids, args)

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        progress = (step + 1) / max(1, args.num_iter)
        new_center_lr = args.center_lr * (args.center_lr_final / args.center_lr) ** progress
        for group in optimizer.param_groups:
            if group.get("name") == "center":
                group["lr"] = new_center_lr

        loss_history.append(area_loss.item())
        print(
            f"Iter {step:04d} | area={area_loss.item():.6f} "
            f"edge={edge_loss.item():.6f} | split={split_count} prune={prune_count}"
        )

        preview_due = args.preview_interval > 0 and (step + 1) % args.preview_interval == 0
        save_due = args.save_interval > 0 and (step + 1) % args.save_interval == 0
        if preview_due or save_due:
            preview_view = torch.randint(0, args.num_views, (1,), device=device).item()
            ellipsoids.set_active_view(preview_view)
            frame = render_image(ellipsoids, args.resolution, args.aa_eval, True, device)
            if preview_due:
                plot_comparison(
                    target_imgs[preview_view],
                    frame,
                    initial=None,
                    save_path=run_dir / f"preview_{step + 1:04d}.png",
                )
            if save_due:
                save_tensor_image(run_dir / f"iter_{step + 1:04d}.png", frame)
            ellipsoids.set_active_view(view_idx)

        if args.state_interval > 0 and (step + 1) % args.state_interval == 0:
            torch.save(ellipsoids.state_dict(), run_dir / f"state_{step + 1:04d}.pt")

    torch.save(ellipsoids.state_dict(), run_dir / "final.pt")
    ellipsoids.set_active_view(0)
    final = render_image(ellipsoids, args.resolution, args.aa_eval, True, device)
    plot_comparison(target_imgs[0], final, initial=out_init, save_path=run_dir / "comparison.pdf")
    save_tensor_image(run_dir / "final.png", final)

    plt.figure()
    plt.title("Area loss")
    plt.semilogy(loss_history)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.savefig(run_dir / "loss.pdf", dpi=300, bbox_inches="tight")
    plt.close()

    if args.create_gif:
        maybe_create_gif(run_dir, run_dir / "optimization.gif")


if __name__ == "__main__":
    main()
