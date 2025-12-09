"""
Port of the original `cel_shading.py` experiment.

Optimizes the thresholds of the Slang cel-shading shader so that its RGB output
matches a target render (with different thresholds) using pixel + boundary
losses. Saves intermediate renders plus a GIF of the optimization progress.
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
from python.integrands import CelShadingIntegrandSlang  # noqa: E402
from python.helpers import BoundaryLossConfig, boundary_loss, points_on_grid  # noqa: E402


def render_image(integrand, resolution, aa_factor, jitter, device):
    samples = points_on_grid(resolution * aa_factor, jitter=jitter).to(device)
    preds = integrand(samples).reshape(
        resolution,
        aa_factor,
        resolution,
        aa_factor,
        integrand.out_dim,
    )
    return preds.mean(dim=(1, 3))


def set_thresholds(integrand, thresholds):
    with torch.no_grad():
        integrand.p.data[4:7] = torch.tensor(thresholds, device=integrand.p.device)


def save_rgb_image(tensor, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = tensor.detach().cpu().numpy().clip(0.0, 1.0)
    plt.imsave(path, arr, origin="lower")


def plot_comparison(target, image, title, save_path):
    plt.figure(figsize=(12, 4))
    for idx, (label, data) in enumerate([("Target", target), (title, image), ("|diff|", (image - target).abs())]):
        plt.subplot(1, 3, idx + 1)
        plt.imshow(data.detach().cpu().numpy().clip(0.0, 1.0), origin="lower")
        plt.title(label)
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_iter_image(integrand, args, device, step, results_dir):
    img = render_image(integrand, args.gt_resolution, args.aa_eval, jitter=False, device=device)
    save_rgb_image(img, results_dir / f"iter_{step:04d}.png")


def train(integrand, args, device, boundary_cfg, target_img, results_dir):
    optimizer = optim.Adam(integrand.parameters(), lr=args.lr)
    for step in range(args.num_iter):
        optimizer.zero_grad()
        pts = points_on_grid(args.gt_resolution * args.aa_train, jitter=True).to(device)
        preds = integrand(pts).reshape(
            args.gt_resolution,
            args.aa_train,
            args.gt_resolution,
            args.aa_train,
            integrand.out_dim,
        ).mean(dim=(1, 3))

        area_loss = (preds - target_img).square().mean()
        boundary_term = boundary_loss(integrand, boundary_cfg)
        total_loss = area_loss + boundary_term
        total_loss.backward()
        optimizer.step()

        if step % args.log_every == 0:
            print(f"Iter {step:04d} | loss={total_loss.item():.6f}")

        if step % args.save_every == 0 or step == args.num_iter - 1:
            save_iter_image(integrand, args, device, step, results_dir)


def make_gif(results_dir, duration):
    frames = sorted(results_dir.glob("iter_*.png"))
    if not frames:
        return
    images = [imageio.imread(frame) for frame in frames]
    imageio.mimsave(results_dir / "optimization.gif", images, duration=duration)


def parse_args():
    parser = argparse.ArgumentParser(description="Cel shading threshold fitting example.")
    parser.add_argument("--gt-resolution", type=int, default=512)
    parser.add_argument("--aa-target", type=int, default=16)
    parser.add_argument("--aa-train", type=int, default=4)
    parser.add_argument("--aa-eval", type=int, default=4)
    parser.add_argument("--grid-n", type=int, default=150, help="Unused (kept for symmetry with other scripts).")
    parser.add_argument("--num-iter", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results-name", type=str, default=None)
    parser.add_argument("--gif-duration", type=float, default=0.2)
    parser.add_argument("--edge-sample-res", type=int, default=5000)
    parser.add_argument("--edge-num-subdivision", type=int, default=20)
    parser.add_argument("--edge-kde-k", type=int, default=14)
    parser.add_argument("--edge-div-eps", type=float, default=1e-15)
    parser.add_argument("--edge-plot-resolution", type=int, default=1000)
    parser.add_argument("--edge-mode", choices=["direct", "L1_img", "L2_img"], default="L1_img")
    parser.add_argument("--target-thresholds", type=float, nargs=3, default=[0.4, 0.6, 0.8])
    parser.add_argument("--init-thresholds", type=float, nargs=3, default=[0.2, 0.35, 0.45])
    parser.add_argument("--time", type=float, default=4.5)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    src = repo_root / "slang" / "cel_shading.slang"
    dst = repo_root / "slang" / "__gen__cel_shading.slang"
    compile_if_needed(src, dst)

    integrand = CelShadingIntegrandSlang(time=args.time).to(device)

    run_name = args.results_name or f"cel_shading_res{args.gt_resolution}_iter{args.num_iter}"
    results_dir = repo_root / "results" / "cel_shading" / run_name
    results_dir.mkdir(parents=True, exist_ok=True)

    orig_params = integrand.p.detach().clone()

    set_thresholds(integrand, args.target_thresholds)
    target_img = render_image(integrand, args.gt_resolution, args.aa_target, jitter=False, device=device).detach()
    save_rgb_image(target_img, results_dir / "target.png")

    set_thresholds(integrand, args.init_thresholds)
    initial_img = render_image(integrand, args.gt_resolution, args.aa_eval, jitter=False, device=device)
    save_rgb_image(initial_img, results_dir / "initial.png")
    save_rgb_image(initial_img, results_dir / "iter_0000.png")
    plot_comparison(target_img, initial_img, "Initial", results_dir / "initial_comparison.png")

    boundary_cfg = BoundaryLossConfig(
        grid_size=args.edge_sample_res,
        num_subdivision=args.edge_num_subdivision,
        div_eps=args.edge_div_eps,
        plot_resolution=args.edge_plot_resolution,
        kde_k=args.edge_kde_k,
        mode=args.edge_mode,
        mode_aux_data=target_img.detach(),
        df_dx_mode="forward",
    )

    train(integrand, args, device, boundary_cfg, target_img, results_dir)

    final_img = render_image(integrand, args.gt_resolution, args.aa_eval, jitter=False, device=device)
    save_rgb_image(final_img, results_dir / f"iter_{args.num_iter:04d}.png")
    plot_comparison(target_img, final_img, "Final", results_dir / "final_comparison.png")

    make_gif(results_dir, args.gif_duration)

    # restore original params (useful if the caller reuses the integrand)
    with torch.no_grad():
        integrand.p.copy_(orig_params)


if __name__ == "__main__":
    main()
