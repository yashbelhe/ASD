"""
Pure PyTorch optimization of isotropic 2D Gaussians to fit an RGB target image.
Initialization and learning rates mirror the vector-graphics circle example.
"""

import argparse
import math
from pathlib import Path
import sys

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))

from python.utils.segments import points_on_grid  # noqa: E402


def load_target_image(name: str, resolution: int, device: torch.device) -> torch.Tensor:
    presets = {
        "diamond": repo_root / "data" / "diamond_input.png",
        "starry": repo_root / "data" / "starry_wikimedia.jpg",
        "scream": repo_root / "data" / "scream_wikimedia.jpg",
        "fallingwater": repo_root / "data" / "fallingwater.jpg",
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


def save_tensor_image(path, image):
    plt.imsave(path, image.detach().cpu().numpy().clip(0, 1), origin="lower", dpi=300)


class IsotropicGaussianPainter(nn.Module):
    def __init__(self, num_gaussians: int, min_sigma=0.01, max_sigma=0.25, seed=42):
        super().__init__()
        self.num_gaussians = num_gaussians
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        if seed is not None:
            torch.manual_seed(seed)

        centers = []
        radii = []
        colors = []
        opacities = []
        grid_dim = max(1, math.ceil(math.sqrt(num_gaussians)))
        cell_width = 1.0 / grid_dim
        count = 0
        for i in range(grid_dim):
            for j in range(grid_dim):
                if count >= num_gaussians:
                    break
                cx = (i + 0.5) / grid_dim + (torch.rand(1).item() - 0.5) * 0.3 * cell_width
                cy = (j + 0.5) / grid_dim + (torch.rand(1).item() - 0.5) * 0.3 * cell_width
                cx = min(max(cx, 0.0), 1.0)
                cy = min(max(cy, 0.0), 1.0)
                radius = (0.1 + torch.rand(1).item() * 0.3) * cell_width
                opacity = 0.8 + torch.rand(1).item() * 0.2
                color = torch.rand(3).tolist()
                centers.append((cx, cy))
                radii.append(radius)
                colors.append(color)
                opacities.append(opacity)
                count += 1
            if count >= num_gaussians:
                break

        self.centers = nn.Parameter(torch.tensor(centers, dtype=torch.float32))
        self.radii = nn.Parameter(torch.tensor(radii, dtype=torch.float32).unsqueeze(-1))
        self.colors = nn.Parameter(torch.tensor(colors, dtype=torch.float32))
        self.opacities = nn.Parameter(torch.tensor(opacities, dtype=torch.float32).unsqueeze(-1))
        self.clamp_parameters()

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        pts = points.unsqueeze(1)  # (S, 1, 2)
        centers = self.centers.unsqueeze(0)  # (1, N, 2)
        diff = pts - centers
        dist2 = (diff * diff).sum(dim=-1)  # (S, N)
        sigma = self.radii.squeeze(-1).clamp(self.min_sigma, self.max_sigma)
        gaussians = torch.exp(-dist2 / (2.0 * (sigma.unsqueeze(0) ** 2) + 1e-8))
        weighted_colors = self.colors * self.opacities.squeeze(-1).unsqueeze(-1)
        rgb = (gaussians.unsqueeze(-1) * weighted_colors.unsqueeze(0)).sum(dim=1)
        return rgb.clamp(0.0, 1.0)

    @torch.no_grad()
    def clamp_parameters(self):
        self.centers.clamp_(0.0, 1.0)
        self.radii.clamp_(self.min_sigma, self.max_sigma)
        self.colors.clamp_(0.0, 1.0)
        self.opacities.clamp_(0.05, 1.0)


def main():
    parser = argparse.ArgumentParser(description="Optimize isotropic Gaussian blobs to match an image.")
    parser.add_argument("--image", type=str, default="scream", help="Image preset or absolute path.")
    parser.add_argument("--num-gaussians", type=int, default=100)
    parser.add_argument("--train-resolution", type=int, default=256)
    parser.add_argument("--save-resolution", type=int, default=512)
    parser.add_argument("--num-iter", type=int, default=500)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--center-lr-scale", type=float, default=0.5)
    parser.add_argument("--radius-lr-scale", type=float, default=0.2)
    parser.add_argument("--color-lr-scale", type=float, default=0.3)
    parser.add_argument("--opacity-lr-scale", type=float, default=0.1)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--aa-train", type=int, default=1)
    parser.add_argument("--aa-eval", type=int, default=2)
    parser.add_argument("--min-sigma", type=float, default=0.01)
    parser.add_argument("--max-sigma", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = IsotropicGaussianPainter(
        args.num_gaussians,
        args.min_sigma,
        args.max_sigma,
        seed=args.seed,
    ).to(device)
    target = load_target_image(args.image, args.train_resolution, device)

    results_dir = repo_root / "results" / "gaussian_fit"
    run_dir = results_dir / f"{Path(args.image).stem}_N{args.num_gaussians}"
    run_dir.mkdir(parents=True, exist_ok=True)

    optimizer = optim.Adam(
        [
            {"params": [model.centers], "lr": args.lr * args.center_lr_scale},
            {"params": [model.radii], "lr": args.lr * args.radius_lr_scale},
            {"params": [model.colors], "lr": args.lr * args.color_lr_scale},
            {"params": [model.opacities], "lr": args.lr * args.opacity_lr_scale},
        ],
        betas=(0.9, 0.99),
    )
    loss_history = []

    def render(resolution, aa_factor, jitter):
        pts = points_on_grid(resolution * aa_factor, jitter=jitter).to(device)
        preds = model(pts).reshape(resolution, aa_factor, resolution, aa_factor, 3).mean(dim=(1, 3))
        return preds

    initial_frame = run_dir / "initial.png"
    with torch.no_grad():
        init_img = render(args.save_resolution, args.aa_eval, False)
    save_tensor_image(initial_frame, init_img)
    saved_frames = [initial_frame]

    for step in range(args.num_iter):
        pts = points_on_grid(args.train_resolution * args.aa_train, jitter=True).to(device)
        preds = model(pts).reshape(
            args.train_resolution,
            args.aa_train,
            args.train_resolution,
            args.aa_train,
            3,
        ).mean(dim=(1, 3))
        loss = (preds - target).square().mean()
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()
        model.clamp_parameters()

        loss_history.append(loss.item())
        if step % 10 == 0:
            print(f"Iter {step:04d} | loss={loss.item():.6f}")
        if step % args.save_every == 0:
            with torch.no_grad():
                frame = render(args.save_resolution, args.aa_eval, False)
            frame_path = run_dir / f"iter_{step:04d}.png"
            save_tensor_image(frame_path, frame)
            saved_frames.append(frame_path)

    final_path = run_dir / "final.png"
    with torch.no_grad():
        final_img = render(args.save_resolution, args.aa_eval, False)
    save_tensor_image(final_path, final_img)
    saved_frames.append(final_path)

    plt.figure()
    plt.title("Gaussian fit loss")
    plt.semilogy(loss_history)
    plt.xlabel("Iteration")
    plt.ylabel("MSE")
    plt.savefig(run_dir / "loss_history.pdf", dpi=300, bbox_inches="tight")
    plt.close()

    frames = [imageio.imread(path) for path in saved_frames if path.exists()]
    if frames:
        imageio.mimsave(run_dir / "optimization.gif", frames, duration=0.2)


if __name__ == "__main__":
    main()
