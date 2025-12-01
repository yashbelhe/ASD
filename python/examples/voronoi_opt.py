"""
Voronoi texture optimization example (non-accel, non-warp version).

Matches the behavior of the original `voronoi_opt.py` script by optimizing a
Voronoi diagram so that its rasterized colors match a target RGB image using a
pixel loss plus the boundary integral loss.
"""

import argparse
from pathlib import Path
import sys

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from PIL import Image

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))

from compiler.compile_shader import compile_if_needed  # noqa: E402
from python.integrands import VoronoiSimpleIntegrandSlang  # noqa: E402
from python.helpers import (  # noqa: E402
    BoundaryLossConfig,
    boundary_loss_slang,
    points_on_grid,
)


def plot_comparison(target, current, initial=None, title_mid="Current", save_path=None):
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(target.detach().cpu().numpy(), extent=[0, 1, 0, 1], origin="lower")
    plt.title("Target")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(current.detach().cpu().numpy(), extent=[0, 1, 0, 1], origin="lower")
    plt.title(title_mid)
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
    name_to_file = {
        "diamond": "diamond_input.png",
        "starry": "starry_wikimedia.jpg",
        "scream": "scream_wikimedia.jpg",
    }
    if Path(name).is_file():
        img_path = Path(name)
    else:
        filename = name_to_file.get(name.lower(), name)
        candidates = [
            repo_root / "data" / filename,
            repo_root.parent / "data" / filename,
        ]
        for cand in candidates:
            if cand.exists():
                img_path = cand
                break
        else:
            raise FileNotFoundError(f"Could not resolve target image for '{name}'.")

    img = Image.open(img_path).convert("RGB")
    img = img.resize((resolution, resolution), Image.LANCZOS)
    arr = np.array(img).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).to(device)
    tensor = torch.flip(tensor, dims=[0])  # match original flipUD
    return tensor


def render_image(integrand, resolution, aa_factor, jitter, device):
    samples = points_on_grid(resolution * aa_factor, jitter=jitter).to(device)
    out = integrand(samples)
    out = out.reshape(resolution, aa_factor, resolution, aa_factor, integrand.out_dim)
    return out.mean(dim=(1, 3))


def save_intermediate(image, path):
    plt.imsave(path, image.detach().cpu().numpy().clip(0, 1), origin="lower", dpi=300)


def create_gif(save_dir, pattern="result_iteration_*.png", output_name="optimization.gif", duration=0.2):
    image_files = sorted(save_dir.glob(pattern))
    if not image_files:
        return
    images = [imageio.imread(path) for path in image_files]
    imageio.mimsave(save_dir / output_name, images, duration=duration)


def main():
    parser = argparse.ArgumentParser(description="Optimize a Voronoi diagram to match a target image.")
    parser.add_argument("--image", type=str, default="scream", help="Image preset ('diamond', 'starry', 'scream') or path.")
    parser.add_argument("--grid-size", type=int, default=30, help="Number of cells per axis for the Voronoi sites.")
    parser.add_argument("--num-iter", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--pixel-loss", choices=["L1", "L2"], default="L2")
    parser.add_argument("--gt-resolution", type=int, default=512)
    parser.add_argument("--aa-train", type=int, default=4, help="AA factor used during training renders.")
    parser.add_argument("--aa-eval", type=int, default=16, help="AA factor for final renders.")
    parser.add_argument("--edge-sample-res", type=int, default=5000)
    parser.add_argument("--num-subdivision", type=int, default=20)
    parser.add_argument("--kde-k", type=int, default=14)
    parser.add_argument("--div-eps", type=float, default=1e-15)
    parser.add_argument("--plot-resolution", type=int, default=1000)
    parser.add_argument("--jitter-scale", type=float, default=0.8, help="Sampling jitter for Voronoi points.")
    parser.add_argument("--seed", type=int, default=12)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--warp", action="store_true", help="Use the domain-warped Voronoi shader.")
    parser.add_argument("--gif-duration", type=float, default=0.2, help="Frame duration for the optimization GIF.")
    parser.add_argument("--no-gif", action="store_true")
    args = parser.parse_args()

    shader_file = "voronoi_simple_warp.slang" if args.warp else "voronoi_simple.slang"
    src = repo_root / "slang" / shader_file
    dst = repo_root / "slang" / f"__gen__{shader_file}"
    compile_if_needed(src, dst)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    integrand = VoronoiSimpleIntegrandSlang(
        grid_size=args.grid_size,
        jitter_scale=args.jitter_scale,
        seed=args.seed,
        warp=args.warp,
    ).to(device)

    target_img = load_target_image(args.image, args.gt_resolution, device)
    out_init = render_image(integrand, args.gt_resolution, args.aa_train, True, device)

    results_dir = repo_root / "results" / "voronoi_opt"
    warp_tag = "warp" if args.warp else "nowarp"
    run_dir = results_dir / f"{args.image}_N{args.grid_size}_{warp_tag}"
    run_dir.mkdir(parents=True, exist_ok=True)
    save_intermediate(out_init, run_dir / "initial.png")

    boundary_cfg = BoundaryLossConfig(
        grid_size=args.edge_sample_res,
        num_subdivision=args.num_subdivision,
        lipschitz_bounds=1e-6,
        div_eps=args.div_eps,
        plot_resolution=args.plot_resolution,
        kde_k=args.kde_k,
        mode="L1_img" if args.pixel_loss == "L1" else "L2_img",
        mode_aux_data=target_img.detach(),
    )

    optimizer = optim.Adam(
        integrand.parameters(),
        lr=args.lr,
        betas=(0.5, 1 - (1 - 0.5) ** 2),
    )
    lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    loss_history = []
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

        if args.pixel_loss == "L1":
            pixel_loss = (preds - target_img).abs().mean()
        else:
            pixel_loss = (preds - target_img).square().mean()

        edge_loss = boundary_loss_slang(integrand, boundary_cfg)
        total_loss = pixel_loss + edge_loss
        total_loss.backward()

        optimizer.step()
        lr_scheduler.step()

        loss_history.append(pixel_loss.item())
        if step % 1 == 0:
            print(f"Iter {step:04d} | pixel={pixel_loss.item():.6f} | edge={edge_loss.item():.6f}")

        if step % args.save_every == 0:
            save_intermediate(preds, run_dir / f"result_iteration_{step:04d}.png")

    final = render_image(integrand, args.gt_resolution, args.aa_eval, True, device)
    plot_comparison(
        target_img,
        final,
        initial=out_init,
        title_mid="Final",
        save_path=run_dir / "final.pdf",
    )
    save_intermediate(final, run_dir / "final.png")

    plt.figure()
    plt.title("Pixel loss history")
    plt.semilogy(loss_history)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.savefig(run_dir / "loss_history.pdf", dpi=300, bbox_inches="tight")
    plt.close()

    if not args.no_gif:
        create_gif(run_dir, output_name="optimization.gif", duration=args.gif_duration)


if __name__ == "__main__":
    main()
