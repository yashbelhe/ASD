"""
Optimize the Slang `binary_threshold` shader (bilinear grid + step) so its
output matches the SIGGRAPH logo baseline, mirroring the original
`compare_sigmoid_relu.py` experiment (pixel + boundary losses, n=150 grid).
"""

import argparse
from pathlib import Path
import sys

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from PIL import Image
import torchvision.transforms as transforms

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))

from compiler.compile_shader import compile_if_needed  # noqa: E402
from python.integrands import BinaryThresholdIntegrandSlang  # noqa: E402
from python.helpers import BoundaryLossConfig, boundary_loss, points_on_grid  # noqa: E402


def reshape_and_average(values, resolution, aa_factor):
    return values.reshape(resolution, aa_factor, resolution, aa_factor).mean(dim=(1, 3))


def render_integrand_image(integrand, resolution, aa_factor, jitter, device):
    samples = points_on_grid(resolution * aa_factor, jitter=jitter).to(device)
    preds = integrand(samples)
    return reshape_and_average(preds, resolution, aa_factor)


def load_target_image(image_path, resolution, device):
    img = Image.open(image_path).convert("L")
    tensor = transforms.ToTensor()(img)
    tensor = transforms.Resize((resolution, resolution), antialias=True)(tensor)
    tensor = tensor.to(device)[0]
    return (tensor > 0.5).float()


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
    plt.imshow(diff_image, extent=[0, 1, 0, 1], origin="lower", cmap="gray", vmin=vmin, vmax=vmax)
    plt.title("Difference")
    plt.axis("off")

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def train_integrand(integrand, args, device, results_dir, boundary_cfg, target_img):
    optimizer = optim.Adam(integrand.parameters(), lr=args.lr)
    for step in range(args.num_iter):
        optimizer.zero_grad()

        samples = points_on_grid(args.gt_resolution * args.aa_train, jitter=True).to(device)
        preds = reshape_and_average(integrand(samples), args.gt_resolution, args.aa_train)
        area_loss = (preds - target_img).square().mean()
        boundary_term = boundary_loss(integrand, boundary_cfg)
        total_loss = boundary_term
        total_loss.backward()
        optimizer.step()

        if step % args.log_every == 0:
            print(f"Iter {step:04d} | loss={total_loss.item():.6f}")

        if args.save_every > 0 and step % args.save_every == 0 and step > 0:
            eval_img = render_integrand_image(integrand, args.gt_resolution, args.aa_eval, jitter=False, device=device)
            save_tensor_image(eval_img, results_dir / f"iter_{step:04d}.png")

def create_optimization_gif(output_path, duration=0.2):
    image_dir = Path(output_path)
    image_files = sorted(image_dir.glob("iter_*.png"))

    if not image_files:
        print(f"No image files found in {image_dir} to create GIF.")
        return

    images = [imageio.imread(str(file)) for file in image_files]
    gif_path = image_dir / "optimization.gif"
    image_dir.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(gif_path, images, duration=duration)
    print(f"GIF created at {gif_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Match the SIGGRAPH logo using the Slang binary-threshold shader.")
    parser.add_argument("--target-image", type=str, default="data/siggraph_bw.tif")
    parser.add_argument("--gt-resolution", type=int, default=4096, help="Rendering resolution.")
    parser.add_argument("--aa-train", type=int, default=1, help="AA factor during training renders.")
    parser.add_argument("--aa-eval", type=int, default=1, help="AA factor for evaluation renders.")
    parser.add_argument("--grid-n", type=int, default=150, help="Number of stored grid points per axis.")
    parser.add_argument("--num-iter", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--save-every", type=int, default=25, help="Save interval for intermediate renders.")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=40)
    parser.add_argument("--results-name", type=str, default=None, help="Optional custom name for the results folder.")
    parser.add_argument("--gif-duration", type=float, default=0.2)
    parser.add_argument("--boundary-grid-size", type=int, default=5000)
    parser.add_argument("--boundary-num-subdivision", type=int, default=20)
    parser.add_argument("--boundary-kde-k", type=int, default=14)
    parser.add_argument("--boundary-div-eps", type=float, default=1e-15)
    parser.add_argument("--boundary-plot-resolution", type=int, default=1000)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    src = repo_root / "slang" / "binary_threshold.slang"
    dst = repo_root / "slang" / "__gen__binary_threshold.slang"
    compile_if_needed(src, dst)

    target_path = Path(args.target_image)
    if not target_path.is_absolute():
        candidates = [
            target_path,
            repo_root / target_path,
            repo_root.parent / target_path,
        ]
    else:
        candidates = [target_path]
    for cand in candidates:
        if cand.exists():
            target_path = cand
            break
    else:
        raise FileNotFoundError(f"Could not find target image at '{args.target_image}'.")

    run_tag = target_path.stem
    run_name = args.results_name or f"{run_tag}_n{args.grid_n}_res{args.gt_resolution}_iter{args.num_iter}"
    run_dir = repo_root / "results" / "binary_fit" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    integrand = BinaryThresholdIntegrandSlang(grid_size=args.grid_n, seed=args.seed).to(device)
    target_img = load_target_image(target_path, args.gt_resolution, device)

    boundary_cfg = BoundaryLossConfig(
        grid_size=args.boundary_grid_size,
        num_subdivision=args.boundary_num_subdivision,
        kde_k=args.boundary_kde_k,
        div_eps=args.boundary_div_eps,
        plot_resolution=args.boundary_plot_resolution,
        mode="L2_img",
        mode_aux_data=target_img.detach(),
        df_dx_mode="backward",
    )

    initial_img = render_integrand_image(integrand, args.gt_resolution, args.aa_eval, jitter=False, device=device)
    save_tensor_image(initial_img, run_dir / "initial.png")
    save_tensor_image(initial_img, run_dir / "iter_0000.png")
    plot_comparison(target_img, initial_img, title_mid="Initial Image", save_path=run_dir / "initial_comparison.png")

    train_integrand(integrand, args, device, run_dir, boundary_cfg, target_img)

    final_img = render_integrand_image(integrand, args.gt_resolution, args.aa_eval, jitter=False, device=device)
    plot_comparison(target_img, final_img, title_mid="Optimized Image", save_path=run_dir / "final_comparison.png")
    save_tensor_image(final_img, run_dir / f"iter_{args.num_iter:04d}.png")

    create_optimization_gif(run_dir, duration=args.gif_duration)


if __name__ == "__main__":
    main()
