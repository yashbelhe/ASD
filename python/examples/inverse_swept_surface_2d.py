"""
Inverse swept-surface fitting: optimize the swept_bilinear shader so its render
matches a ground-truth swept-brush target (mirrors the original notebook).
"""

import argparse
from pathlib import Path
import sys

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import torch
import torch.optim as optim

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))

from compiler.compile_shader import compile_if_needed  # noqa: E402
from python.helpers import BoundaryLossConfig, boundary_loss_slang, points_on_grid  # noqa: E402
from python.integrands import (  # noqa: E402
    SweptBilinearIntegrandSlang,
    SweptBrushIntegrandSlang,
)


def default_bristle_params():
    return torch.tensor([
        [2e-2,      -9e-2,      1e-3],
        [1.5e-2,    -8e-2,      1e-3],
        [0.0,       -7e-2,      2e-3],
        [-1e-2,     -6e-2,      1e-3],
        [0.0,       -5.5e-2,    2e-3],
        [1.2e-2,    -5e-2,      2e-3],
        [1.5e-2,    -4e-2,      4e-3],
        [-2e-3,     -3e-2,      4e-3],
        [0.0,       -2e-2,      5e-3],
        [1e-2,      0.0,        5e-3],
        [-1e-2,     1e-2,       7e-3],
        [-5e-3,     1e-2,       7e-3],
        [1e-3,      2e-2,       3e-3],
        [2e-3,      3e-2,       2e-3],
        [2e-3,      3.4e-2,     1e-3],
    ], dtype=torch.float32)


def reshape_and_average(values, resolution, aa_factor):
    return values.reshape(resolution, aa_factor, resolution, aa_factor).mean(dim=(1, 3))


def render_integrand_image(integrand, resolution, aa_factor, jitter, device, max_chunk_points=4_000_000):
    grid = resolution * aa_factor
    samples_cpu = points_on_grid(grid, jitter=jitter, device=torch.device("cpu"))
    preds = []
    chunk = max_chunk_points if max_chunk_points is not None else samples_cpu.shape[0]
    for batch in torch.split(samples_cpu, chunk):
        with torch.no_grad():
            preds.append(integrand(batch.to(device)).detach().cpu())
    preds = torch.cat(preds, dim=0)
    return reshape_and_average(preds, resolution, aa_factor)


def render_brush_image(integrand, resolution, aa_factor, jitter, device, brush_start_step=-1, max_chunk_points=4_000_000):
    with torch.no_grad():
        original = integrand.p[1].item()
        integrand.p.data[1] = float(brush_start_step)
        img = render_integrand_image(integrand, resolution, aa_factor, jitter, device, max_chunk_points)
        integrand.p.data[1] = original
    return img


def save_gray_image(img, path, invert=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    array = img.detach().cpu().numpy()
    if invert:
        array = 1.0 - array
    plt.imsave(path, array, cmap="gray", origin="lower")


def plot_comparison(target_img, candidate_img, target_brush, candidate_brush, title_mid="Candidate", save_path=None):
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 3, 1)
    plt.imshow(target_img.detach().cpu().numpy(), extent=[0, 1, 0, 1], origin="lower", cmap="gray")
    plt.title("Target Image")
    plt.axis("off")

    plt.subplot(2, 3, 2)
    plt.imshow(candidate_img.detach().cpu().numpy(), extent=[0, 1, 0, 1], origin="lower", cmap="gray")
    plt.title(f"{title_mid} Image")
    plt.axis("off")

    plt.subplot(2, 3, 3)
    diff = (candidate_img - target_img).abs().detach().cpu().numpy()
    plt.imshow(diff, extent=[0, 1, 0, 1], origin="lower", cmap="gray")
    plt.title("Image Difference")
    plt.axis("off")

    plt.subplot(2, 3, 4)
    plt.imshow(target_brush.detach().cpu().numpy(), extent=[0, 1, 0, 1], origin="lower", cmap="gray")
    plt.title("Target Brush")
    plt.axis("off")

    plt.subplot(2, 3, 5)
    plt.imshow(candidate_brush.detach().cpu().numpy(), extent=[0, 1, 0, 1], origin="lower", cmap="gray")
    plt.title(f"{title_mid} Brush")
    plt.axis("off")

    plt.subplot(2, 3, 6)
    diff_brush = (candidate_brush - target_brush).abs().detach().cpu().numpy()
    plt.imshow(diff_brush, extent=[0, 1, 0, 1], origin="lower", cmap="gray")
    plt.title("Brush Difference")
    plt.axis("off")

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_overview_figure(target_img, initial_img, final_img, target_brush, initial_brush, final_brush, save_path, num_iter):
    diff_img = (final_img - target_img).abs().detach().cpu().numpy()
    diff_brush = (final_brush - target_brush).abs().detach().cpu().numpy()

    plt.figure(figsize=(15, 10))
    axes = [
        (target_img, "Target Image"),
        (initial_img, "Initial Image"),
        (final_img, f"Final Image ({num_iter} iters)"),
        (diff_img, "Image Difference"),
        (target_brush, "Target Brush"),
        (initial_brush, "Initial Brush"),
        (final_brush, f"Final Brush ({num_iter} iters)"),
        (diff_brush, "Brush Difference"),
    ]
    for idx, (img, title) in enumerate(axes, 1):
        plt.subplot(2, 4, idx)
        plt.imshow(img.detach().cpu().numpy() if torch.is_tensor(img) else img, extent=[0, 1, 0, 1], origin="lower", cmap="gray")
        plt.title(title)
        plt.axis("off")
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def create_gif(image_dir, pattern, output_name, duration=0.2):
    image_dir = Path(image_dir)
    files = sorted(image_dir.glob(pattern))
    if not files:
        print(f"No files found matching {pattern} in {image_dir}")
        return
    images = [imageio.imread(str(f)) for f in files]
    gif_path = image_dir / output_name
    image_dir.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(gif_path, images, duration=duration)
    print(f"GIF saved to {gif_path}")


def train_integrand(integrand, gt_fn, args, device, boundary_cfg, run_dir):
    optimizer = optim.Adam(integrand.parameters(), lr=args.lr)
    if args.final_lr_mult is not None and args.num_iter > 0:
        gamma = args.final_lr_mult ** (1.0 / max(1, args.num_iter))
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    else:
        scheduler = None

    history = []
    images_dir = run_dir / "images"
    brushes_dir = run_dir / "brushes"
    snapshots = {}
    brush_snapshots = {}

    for step in range(args.num_iter):
        optimizer.zero_grad()
        samples = points_on_grid(args.train_resolution * args.aa_train, jitter=True).to(device)
        preds = integrand(samples)
        target = gt_fn(samples)

        pixel_loss = (preds - target).square().mean()
        boundary_loss = boundary_loss_slang(integrand, boundary_cfg)
        total_loss = boundary_loss
        total_loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        history.append((pixel_loss.item(), boundary_loss.item()))
        if args.log_every and step % args.log_every == 0:
            print(
                f"Iter {step:04d} | pixel={pixel_loss.item():.6f} "
                f"| boundary={boundary_loss.item():.6f}"
            )

        should_save = args.save_every > 0 and (step % args.save_every == 0 or step == args.num_iter - 1)
        if should_save:
            img = reshape_and_average(preds, args.train_resolution, args.aa_train).detach().cpu()
            save_gray_image(img, images_dir / f"image_{step:04d}.png", invert=True)
            brush_img = render_brush_image(
                integrand,
                args.gt_resolution,
                args.aa_brush,
                jitter=True,
                device=device,
                brush_start_step=args.brush_start_step,
            )
            save_gray_image(brush_img, brushes_dir / f"brush_{step:04d}.png", invert=True)

        if step in args.snapshot_iters:
            img = reshape_and_average(preds, args.train_resolution, args.aa_train).detach().cpu()
            snapshots[step] = img
            brush_img = render_brush_image(
                integrand,
                args.gt_resolution,
                args.aa_brush,
                jitter=True,
                device=device,
                brush_start_step=args.brush_start_step,
            )
            brush_snapshots[step] = brush_img.detach().cpu()

    return history, snapshots, brush_snapshots


def parse_args():
    parser = argparse.ArgumentParser(description="Inverse swept-surface fitting.")
    parser.add_argument("--gt-resolution", type=int, default=512)
    parser.add_argument("--train-resolution", type=int, default=512)
    parser.add_argument("--aa-train", type=int, default=16)
    parser.add_argument("--aa-target", type=int, default=16)
    parser.add_argument("--aa-final", type=int, default=16)
    parser.add_argument("--aa-brush", type=int, default=16)
    parser.add_argument("--grid-n", type=int, default=512)
    parser.add_argument("--num-steps", type=int, default=95)
    parser.add_argument("--start-step", type=int, default=0)
    parser.add_argument("--num-iter", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--final-lr-mult", type=float, default=0.01)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--results-name", type=str, default=None)
    parser.add_argument("--gif-duration", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=40)
    parser.add_argument("--boundary-grid-size", type=int, default=5000)
    parser.add_argument("--boundary-num-subdivision", type=int, default=20)
    parser.add_argument("--boundary-kde-k", type=int, default=14)
    parser.add_argument("--boundary-div-eps", type=float, default=1e-15)
    parser.add_argument("--boundary-plot-resolution", type=int, default=1000)
    parser.add_argument("--snapshot-iters", type=int, nargs="*", default=[10])
    parser.add_argument("--brush-start-step", type=int, default=-1, help="Start-step override for brush visualizations.")
    return parser.parse_args()


def main():
    args = parse_args()
    args.snapshot_iters = set(args.snapshot_iters or [])
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    shaders = [
        (repo_root / "slang" / "swept_brush.slang", repo_root / "slang" / "__gen__swept_brush.slang"),
        (repo_root / "slang" / "swept_bilinear.slang", repo_root / "slang" / "__gen__swept_bilinear.slang"),
    ]
    for src, dst in shaders:
        compile_if_needed(src, dst)

    bristle_params = default_bristle_params()
    gt_integrand = SweptBrushIntegrandSlang(
        bristle_params=bristle_params,
        start_step=args.start_step,
        n_steps=args.num_steps,
    ).to(device)
    gt_fn = lambda x: gt_integrand(x)  # noqa: E731
    candidate = SweptBilinearIntegrandSlang(
        grid_size=args.grid_n,
        start_step=args.start_step,
        n_steps=args.num_steps,
        seed=args.seed,
    ).to(device)

    run_tag = f"grid{args.grid_n}_res{args.gt_resolution}_iter{args.num_iter}"
    run_name = args.results_name or run_tag
    run_dir = repo_root / "results" / "inverse_swept_surface_2d" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    gt_img = render_integrand_image(gt_integrand, args.gt_resolution, args.aa_target, jitter=True, device=device)
    gt_brush = render_brush_image(
        gt_integrand,
        args.gt_resolution,
        args.aa_target,
        jitter=True,
        device=device,
        brush_start_step=args.brush_start_step,
    )
    save_gray_image(gt_img, run_dir / "gt.png", invert=True)
    save_gray_image(gt_brush, run_dir / "gt_brush.png", invert=True)

    init_img = render_integrand_image(candidate, args.gt_resolution, args.aa_target, jitter=True, device=device)
    init_brush = render_brush_image(
        candidate,
        args.gt_resolution,
        args.aa_brush,
        jitter=True,
        device=device,
        brush_start_step=args.brush_start_step,
    )
    save_gray_image(init_img, run_dir / "initial.png", invert=True)
    save_gray_image(init_brush, run_dir / "initial_brush.png", invert=True)
    plot_comparison(gt_img, init_img, gt_brush, init_brush, title_mid="Initial", save_path=run_dir / "initial_comparison.png")

    boundary_cfg = BoundaryLossConfig(
        grid_size=args.boundary_grid_size,
        num_subdivision=args.boundary_num_subdivision,
        kde_k=args.boundary_kde_k,
        div_eps=args.boundary_div_eps,
        plot_resolution=args.boundary_plot_resolution,
        lipschitz_bounds=1e-6,
        mode="L2_test_fn",
        mode_aux_data=gt_fn,
        df_dx_mode="backward",
    )

    history, snapshots, brush_snapshots = train_integrand(
        candidate,
        gt_fn,
        args,
        device,
        boundary_cfg,
        run_dir,
    )

    final_img = render_integrand_image(candidate, args.gt_resolution, args.aa_final, jitter=True, device=device)
    final_brush = render_brush_image(
        candidate,
        args.gt_resolution,
        args.aa_final,
        jitter=True,
        device=device,
        brush_start_step=args.brush_start_step,
    )
    save_gray_image(final_img, run_dir / "final.png", invert=True)
    save_gray_image(final_brush, run_dir / "final_brush.png", invert=True)
    plot_comparison(gt_img, final_img, gt_brush, final_brush, title_mid="Final", save_path=run_dir / "final_comparison.png")

    plt.figure()
    plt.title("Loss History")
    plt.semilogy([h[0] for h in history], label="Pixel")
    plt.semilogy([h[1] for h in history], label="Boundary")
    plt.legend()
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.savefig(run_dir / "loss_history.png", dpi=300, bbox_inches="tight")
    plt.close()

    save_overview_figure(gt_img, init_img, final_img, gt_brush, init_brush, final_brush, run_dir / "overview.png", args.num_iter)

    create_gif(run_dir / "images", "image_*.png", "optimization.gif", duration=args.gif_duration)
    create_gif(run_dir / "brushes", "brush_*.png", "brush_optimization.gif", duration=args.gif_duration)


if __name__ == "__main__":
    main()
