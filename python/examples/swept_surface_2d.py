"""
Fit the Slang swept-brush target with the bilinear grid + step shader using
boundary supervision (ports the original `swept_surface_2d.py` experiment).
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
from python.utils.boundary import BoundaryLossConfig, boundary_loss  # noqa: E402
from python.utils.segments import points_on_grid  # noqa: E402
from python.integrands import (  # noqa: E402
    BinaryThresholdIntegrandSlang,
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
    total_pts = grid * grid
    samples_cpu = points_on_grid(grid, jitter=jitter, device=torch.device("cpu"))
    preds = []
    chunk = max_chunk_points if max_chunk_points is not None else total_pts
    for batch in torch.split(samples_cpu, chunk):
        batch_preds = integrand(batch.to(device))
        preds.append(batch_preds.detach().cpu())
    preds = torch.cat(preds, dim=0)
    img = reshape_and_average(preds, resolution, aa_factor)
    return img.to(device)


def save_tensor_image(img, path, cmap="gray", vmin=0.0, vmax=1.0):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(4, 4))
    plt.imshow(img.detach().cpu().numpy(), extent=[0, 1, 0, 1], origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
    plt.axis("off")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_comparison(target, candidate, title_mid="Candidate", save_path=None, vmin=0.0, vmax=1.0):
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(target.detach().cpu().numpy(), extent=[0, 1, 0, 1], origin="lower", cmap="gray", vmin=vmin, vmax=vmax)
    plt.title("Target")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(candidate.detach().cpu().numpy(), extent=[0, 1, 0, 1], origin="lower", cmap="gray", vmin=vmin, vmax=vmax)
    plt.title(title_mid)
    plt.axis("off")

    plt.subplot(1, 3, 3)
    diff_image = (candidate - target).abs().detach().cpu().numpy()
    plt.imshow(diff_image, extent=[0, 1, 0, 1], origin="lower", cmap="gray", vmin=0.0, vmax=vmax)
    plt.title("Difference")
    plt.axis("off")

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def create_optimization_gif(image_dir, duration=0.2):
    image_dir = Path(image_dir)
    image_files = sorted(image_dir.glob("iter_*.png"))
    if not image_files:
        print(f"No iter_*.png files found in {image_dir}, skipping GIF.")
        return
    images = [imageio.imread(str(file)) for file in image_files]
    gif_path = image_dir / "optimization.gif"
    image_dir.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(gif_path, images, duration=duration)
    print(f"GIF saved to {gif_path}")


def save_loss_plot(history, path):
    if not history:
        return
    iters = range(len(history))
    area = [h[0] for h in history]
    boundary = [h[1] for h in history]
    plt.figure()
    plt.title("Loss History")
    plt.semilogy(list(iters), area, label="Area")
    plt.semilogy(list(iters), boundary, label="Boundary")
    plt.legend()
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def save_overview_figure(swept_rgb, target, initial, intermediate, final_img, save_path):
    diff_image = (final_img - target).abs().detach().cpu().numpy()
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 3, 1)
    if swept_rgb is not None:
        plt.imshow(swept_rgb.detach().cpu().numpy(), extent=[0, 1, 0, 1], origin="lower")
        plt.title("Swept Brush")
    else:
        plt.axis("off")

    plt.subplot(2, 3, 2)
    plt.imshow(target.detach().cpu().numpy(), extent=[0, 1, 0, 1], origin="lower", cmap="gray")
    plt.title("Target")
    plt.axis("off")

    plt.subplot(2, 3, 3)
    plt.imshow(initial.detach().cpu().numpy(), extent=[0, 1, 0, 1], origin="lower", cmap="gray")
    plt.title("Initial")
    plt.axis("off")

    plt.subplot(2, 3, 5)
    plt.imshow(final_img.detach().cpu().numpy(), extent=[0, 1, 0, 1], origin="lower", cmap="gray")
    plt.title("Final")
    plt.axis("off")

    plt.subplot(2, 3, 6)
    plt.imshow(diff_image, extent=[0, 1, 0, 1], origin="lower", cmap="gray")
    plt.title("Difference")
    plt.axis("off")

    plt.subplot(2, 3, 4)
    if intermediate is not None:
        plt.imshow(intermediate.detach().cpu().numpy(), extent=[0, 1, 0, 1], origin="lower", cmap="gray")
        plt.title("Intermediate")
        plt.axis("off")
    else:
        plt.axis("off")

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def render_swept_path_visual(bristle_params, resolution, aa_factor, device):
    with torch.no_grad():
        path_integrand = SweptBrushIntegrandSlang(
            bristle_params=torch.tensor([[0.0, -2e-2, 2e-3]], dtype=torch.float32),
            n_bristles=1,
            n_steps=100,
            draw_path=True,
        ).to(device)
        brush_integrand = SweptBrushIntegrandSlang(
            bristle_params=bristle_params,
            n_steps=1,
        ).to(device)

        samples = points_on_grid(resolution * aa_factor, jitter=True).to(device)
        swept_path = torch.zeros(resolution, resolution, device=device)
        start_step = 10
        steps_per_segment = 100
        num_segments = 10
        for idx in range(num_segments):
            seg_steps = steps_per_segment
            if idx == num_segments - 1:
                seg_steps = steps_per_segment - start_step
            with torch.no_grad():
                path_integrand.p[1] = start_step + idx * steps_per_segment
                path_integrand.p[2] = seg_steps
            preds = path_integrand(samples)
            img = reshape_and_average(preds, resolution, aa_factor)
            swept_path = torch.maximum(swept_path, img)

        brush_img = reshape_and_average(brush_integrand(samples), resolution, aa_factor)
        swept_rgb = brush_img.unsqueeze(-1).repeat(1, 1, 3)
        highlight_color = torch.tensor([0.0, 1.0, 1.0], device=device)
        swept_rgb = torch.where(
            swept_path.unsqueeze(-1) > 0.0,
            swept_path.unsqueeze(-1) * highlight_color,
            swept_rgb,
        )
        return swept_rgb.detach().cpu()


def train_integrand(
    integrand,
    gt_fn,
    target_img_train,
    args,
    device,
    boundary_cfg,
    run_dir,
    snapshot_iters,
):
    optimizer = optim.Adam(integrand.parameters(), lr=args.lr)
    snapshot_iters = set(snapshot_iters or [])
    history = []
    saved_images = {}

    for step in range(args.num_iter):
        optimizer.zero_grad()

        samples = points_on_grid(args.train_resolution * args.aa_train, jitter=True).to(device)
        preds = integrand(samples)

        if args.use_test_fn:
            target_vals = gt_fn(samples)
            area_loss = (preds - target_vals).square().mean()
            train_img = reshape_and_average(preds, args.train_resolution, args.aa_train)
        else:
            if target_img_train is None:
                raise RuntimeError("target_img_train must be provided when --no-test-fn is used.")
            train_img = reshape_and_average(preds, args.train_resolution, args.aa_train)
            area_loss = (train_img - target_img_train).square().mean()

        boundary = boundary_loss(integrand, boundary_cfg)
        total_loss = boundary + area_loss
        total_loss.backward()
        optimizer.step()

        history.append((area_loss.item(), boundary.item()))

        if args.log_every and step % args.log_every == 0:
            print(f"Iter {step:04d} | loss={total_loss.item():.6f}")

        should_save = args.save_every > 0 and (step % args.save_every == 0 or step == args.num_iter - 1)
        if should_save:
            save_tensor_image(train_img, run_dir / f"iter_{step:04d}.png")
            if step in snapshot_iters:
                saved_images[step] = train_img.detach().cpu()
        elif step in snapshot_iters:
            saved_images[step] = train_img.detach().cpu()

    return history, saved_images


def parse_args():
    parser = argparse.ArgumentParser(description="Swept brush fitting demo.")
    parser.add_argument("--gt-resolution", type=int, default=512)
    parser.add_argument("--train-resolution", type=int, default=512)
    parser.add_argument("--aa-train", type=int, default=1, help="AA factor during training renders.")
    parser.add_argument("--aa-target", type=int, default=16, help="AA for GT renders/plots (matches AA_FACTOR_GT).")
    parser.add_argument("--aa-final", type=int, default=16, help="AA for the final visualization render (matches AA_FACTOR_GT).")
    parser.add_argument("--grid-n", type=int, default=450)
    parser.add_argument("--num-iter", type=int, default=400)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--save-every", type=int, default=1, help="Save interval for intermediate renders (matches legacy script).")
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
    parser.add_argument("--save-swept-path", action="store_true")
    parser.add_argument("--start-step", type=int, default=0)
    parser.add_argument("--num-steps", type=int, default=95)
    parser.add_argument("--no-test-fn", dest="use_test_fn", action="store_false")
    parser.set_defaults(use_test_fn=True)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    shaders = [
        (repo_root / "slang" / "binary_threshold.slang", repo_root / "slang" / "__gen__binary_threshold.slang"),
        (repo_root / "slang" / "swept_brush.slang", repo_root / "slang" / "__gen__swept_brush.slang"),
    ]
    for src, dst in shaders:
        compile_if_needed(src, dst)

    bristle_params = default_bristle_params()

    gt_integrand = SweptBrushIntegrandSlang(
        bristle_params=bristle_params,
        start_step=args.start_step,
        n_steps=args.num_steps,
    ).to(device)
    candidate = BinaryThresholdIntegrandSlang(grid_size=args.grid_n, seed=args.seed).to(device)
    gt_fn = lambda x: gt_integrand(x)  # noqa: E731

    run_tag = f"n{args.grid_n}_res{args.gt_resolution}_iter{args.num_iter}"
    run_name = args.results_name or run_tag
    run_dir = repo_root / "results" / "swept_surface_2d" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    gt_eval_img = render_integrand_image(gt_integrand, args.gt_resolution, args.aa_target, jitter=True, device=device)
    save_tensor_image(gt_eval_img, run_dir / "gt.png")

    initial_img = render_integrand_image(candidate, args.gt_resolution, args.aa_train, jitter=False, device=device)
    save_tensor_image(initial_img, run_dir / "initial.png")
    plot_comparison(gt_eval_img, initial_img, title_mid="Initial", save_path=run_dir / "initial_comparison.png")

    target_img_train = None
    if not args.use_test_fn:
        target_img_train = render_integrand_image(
            gt_integrand,
            args.train_resolution,
            args.aa_train,
            jitter=True,
            device=device,
        ).detach()

    boundary_cfg = BoundaryLossConfig(
        grid_size=args.boundary_grid_size,
        num_subdivision=args.boundary_num_subdivision,
        kde_k=args.boundary_kde_k,
        div_eps=args.boundary_div_eps,
        plot_resolution=args.boundary_plot_resolution,
        mode="L2_test_fn" if args.use_test_fn else "L2_img",
        mode_aux_data=gt_fn if args.use_test_fn else target_img_train,
        df_dx_mode="backward",
    )

    history, snapshots = train_integrand(
        candidate,
        gt_fn,
        target_img_train,
        args,
        device,
        boundary_cfg,
        run_dir,
        args.snapshot_iters,
    )

    final_img = render_integrand_image(candidate, args.gt_resolution, args.aa_final, jitter=False, device=device)
    save_tensor_image(final_img, run_dir / "final.png")
    plot_comparison(gt_eval_img, final_img, title_mid="Final", save_path=run_dir / "final_comparison.png")
    save_loss_plot(history, run_dir / "loss_history.png")

    swept_rgb = None
    if args.save_swept_path:
        swept_rgb = render_swept_path_visual(bristle_params, args.gt_resolution, args.aa_target, device)
        save_tensor_image(torch.mean(swept_rgb, dim=-1), run_dir / "swept_brush_bw.png")
        plt.imsave(run_dir / "swept_brush.png", swept_rgb.detach().cpu().numpy())

    intermediate_img = None
    if snapshots:
        first_step = sorted(snapshots.keys())[0]
        intermediate_img = snapshots[first_step]

    save_overview_figure(swept_rgb, gt_eval_img, initial_img, intermediate_img, final_img, run_dir / "overview.png")
    create_optimization_gif(run_dir, duration=args.gif_duration)


if __name__ == "__main__":
    main()
