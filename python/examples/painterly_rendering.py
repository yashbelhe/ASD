"""
Painterly rendering example that optimizes a stack of parametric line strokes
to match a target image using area + boundary losses.

This is a trimmed port of the original `painterly_rendering.py` that keeps only
the pieces exercised by the line-based setup.
"""

import argparse
import os
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from PIL import Image

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))

from compiler.compile_shader import compile_if_needed  # noqa: E402
from python.integrands import VectorGraphicsRGBPaddedAccelIntegrandSlang  # noqa: E402
from python.helpers import (  # noqa: E402
    BoundaryLossConfig,
    boundary_loss_slang,
    points_on_grid,
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


def load_target_image(name: str, resolution: int, device: torch.device) -> torch.Tensor:
    presets = {
        "diamond": repo_root.parent / "data" / "diamond_input.png",
        "starry": repo_root.parent / "data" / "starry_wikimedia.jpg",
        "scream": repo_root.parent / "data" / "scream_wikimedia.jpg",
        "fallingwater": repo_root.parent / "data" / "fallingwater.jpg",
    }
    path = Path(name)
    if not path.exists():
        path = presets.get(name.lower())
    if path is None or not path.exists():
        raise FileNotFoundError(f"Could not locate image '{name}'.")
    img = Image.open(path).convert("RGB")
    if img.width != img.height:
        side = min(img.width, img.height)
        left = (img.width - side) // 2
        top = (img.height - side) // 2
        img = img.crop((left, top, left + side, top + side))
    img = img.resize((resolution, resolution), Image.LANCZOS)
    arr = np.array(img).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).to(device)
    return torch.flip(tensor, dims=[0])


def render_image(integrand, resolution, aa_factor, jitter, device):
    samples = points_on_grid(resolution * aa_factor, jitter=jitter).to(device)
    out = integrand(samples)
    return out.reshape(resolution, aa_factor, resolution, aa_factor, integrand.out_dim).mean(dim=(1, 3))


def save_tensor_image(path, image):
    plt.imsave(path, image.detach().cpu().numpy().clip(0, 1), origin="lower", dpi=300)


def main():
    parser = argparse.ArgumentParser(description="Painterly vector graphics optimization.")
    parser.add_argument(
        "--image",
        type=str,
        default="scream",
        help="Image preset or path (diamond, starry, scream, fallingwater).",
    )
    parser.add_argument(
        "--primitive",
        type=str,
        choices=["line", "bezier"],
        default="line",
        help="Primitive family to optimize.",
    )
    parser.add_argument("--n", type=int, default=30, help="Grid resolution for primitive initialization.")
    parser.add_argument("--accel-grid", type=int, default=100, help="Acceleration grid size.")
    parser.add_argument("--max-elements", type=int, default=200, help="Max primitives per acceleration cell.")
    parser.add_argument("--num-iter", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--area-loss", choices=["L1", "L2"], default="L2")
    parser.add_argument("--gt-resolution", type=int, default=512)
    parser.add_argument("--aa-train", type=int, default=1)
    parser.add_argument("--aa-eval", type=int, default=4)
    parser.add_argument("--edge-grid", type=int, default=2000)
    parser.add_argument("--num-subdivision", type=int, default=20)
    parser.add_argument("--kde-k", type=int, default=14)
    parser.add_argument("--div-eps", type=float, default=1e-15)
    parser.add_argument("--plot-resolution", type=int, default=1000)
    parser.add_argument("--optimize-opacity", action="store_true")
    parser.add_argument("--save-every", type=int, default=50)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    shader = repo_root / "slang" / "vector_graphics_rgb_padded_accel.slang"
    try:
        compile_if_needed(shader, shader.with_name("__gen__vector_graphics_rgb_padded_accel.slang"))
    except (ModuleNotFoundError, AssertionError) as exc:
        print(f"Warning: shader transformer unavailable or skipped ({exc}); using existing generated shader.")

    integrand = VectorGraphicsRGBPaddedAccelIntegrandSlang(
        n=args.n,
        grid_size=args.accel_grid,
        max_elements_per_cell=args.max_elements,
        primitive_type=args.primitive,
    ).to(device)

    target = load_target_image(args.image, args.gt_resolution, device)
    out_init = render_image(integrand, args.gt_resolution, args.aa_train, True, device)

    results_dir = repo_root / "results" / "painterly"
    run_dir = results_dir / f"{Path(args.image).stem}_n{args.n}_{args.primitive}"
    run_dir.mkdir(parents=True, exist_ok=True)
    save_tensor_image(run_dir / "initial.png", out_init)

    cfg = BoundaryLossConfig(
        grid_size=args.edge_grid,
        num_subdivision=args.num_subdivision,
        kde_k=args.kde_k,
        div_eps=args.div_eps,
        plot_resolution=args.plot_resolution,
        mode="L1_img" if args.area_loss == "L1" else "L2_img",
        mode_aux_data=target.detach(),
    )

    beta1 = 0.5
    beta2 = 1 - (1 - beta1) ** 2
    opacity_lr = args.lr * 10.0 if args.optimize_opacity else 0.0
    optimizer = optim.Adam(
        [
            {"params": integrand.control_points, "lr": args.lr * 0.5},
            {"params": integrand.stroke_widths, "lr": args.lr * 0.2},
            {"params": integrand.fill_colors, "lr": args.lr * 10.0},
            {"params": integrand.opacities, "lr": opacity_lr},
            {"params": [integrand.primitive_types, integrand.fill_types, integrand.other_fill_params], "lr": 0.0},
        ],
        betas=(beta1, beta2),
    )
    lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    loss_history = []

    for step in range(args.num_iter):
        pts = points_on_grid(args.gt_resolution * args.aa_train, jitter=True).to(device)
        preds = integrand(pts).reshape(
            args.gt_resolution,
            args.aa_train,
            args.gt_resolution,
            args.aa_train,
            integrand.out_dim,
        ).mean(dim=(1, 3))

        if args.area_loss == "L1":
            area_loss = (preds - target).abs().mean()
        else:
            area_loss = (preds - target).square().mean()

        edge_loss = boundary_loss_slang(integrand, cfg)
        total = area_loss + edge_loss
        total.backward()

        if not args.optimize_opacity and integrand.opacities.grad is not None:
            integrand.opacities.grad.zero_()

        optimizer.step()
        optimizer.zero_grad()
        lr_scheduler.step()

        loss_history.append(area_loss.item())
        if step % 10 == 0:
            print(f"Iter {step:04d} | area={area_loss.item():.6f}")

        if step % args.save_every == 0:
            save_tensor_image(run_dir / f"iter_{step:04d}.png", preds)

    final = render_image(integrand, args.gt_resolution, args.aa_eval, True, device)
    plot_comparison(target, final, initial=out_init, save_path=run_dir / "comparison.pdf")
    save_tensor_image(run_dir / "final.png", final)

    plt.figure()
    plt.title("Area loss")
    plt.semilogy(loss_history)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.savefig(run_dir / "loss_history.pdf", dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
