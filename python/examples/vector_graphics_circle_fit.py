"""
Optimize a stack of circles rendered by the vector-graphics Slang shader to
match a target RGB image. Saves both a full-resolution GIF and an optional
zoomed GIF around a user-specified region.
"""

import argparse
from pathlib import Path
import sys

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torch.optim as optim

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))

from compiler.compile_shader import compile_if_needed  # noqa: E402
from python.integrands import VectorGraphicsRGBPaddedAccelIntegrandSlang  # noqa: E402
from python.utils.boundary import BoundaryLossConfig, boundary_loss  # noqa: E402
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


def render_image(integrand, resolution, aa_factor, jitter, device):
    samples = points_on_grid(resolution * aa_factor, jitter=jitter).to(device)
    out = integrand(samples)
    return out.reshape(resolution, aa_factor, resolution, aa_factor, integrand.out_dim).mean(dim=(1, 3))


def save_tensor_image(path, image):
    plt.imsave(path, image.detach().cpu().numpy().clip(0, 1), origin="lower", dpi=300)


def clamp_circle_params(integrand, min_radius=0.01, max_radius=0.3):
    if integrand.primitive_type != "circle":
        return
    with torch.no_grad():
        control = integrand.control_points.view(-1, 6)
        integrand.stroke_widths.clamp_(min_radius, max_radius)
        control[:, 0:2].clamp_(0.0, 1.0)
        control[:, 2] = integrand.stroke_widths.view(-1)
        integrand.fill_colors.clamp_(0.0, 1.0)
        integrand.opacities.clamp_(0.05, 1.0)


def render_zoom_image(integrand, resolution, aa_factor, center, extent, device, jitter=False):
    """Render the integrand over a zoomed window centered at `center`."""
    x_center, y_center = center
    half = extent / 2.0
    xmin = max(x_center - half, 0.0)
    xmax = min(x_center + half, 1.0)
    ymin = max(y_center - half, 0.0)
    ymax = min(y_center + half, 1.0)
    samples = points_on_grid(resolution * aa_factor, jitter=jitter).to(device)
    xs = xmin + (xmax - xmin) * samples[:, 0]
    ys = ymin + (ymax - ymin) * samples[:, 1]
    region_samples = torch.stack([xs, ys], dim=1)
    out = integrand(region_samples)
    return out.reshape(resolution, aa_factor, resolution, aa_factor, integrand.out_dim).mean(dim=(1, 3))


def main():
    parser = argparse.ArgumentParser(description="Optimize circles with the vector-graphics shader.")
    parser.add_argument("--image", type=str, default="scream", help="Image preset or path.")
    parser.add_argument("--num-circles", type=int, default=100, help="Number of circle primitives.")
    parser.add_argument("--accel-grid", type=int, default=80, help="Acceleration grid resolution.")
    parser.add_argument("--max-elements", type=int, default=128, help="Max primitives per acceleration cell.")
    parser.add_argument("--num-iter", type=int, default=400)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--center-lr-scale", type=float, default=0.5)
    parser.add_argument("--radius-lr-scale", type=float, default=0.2)
    parser.add_argument("--color-lr-scale", type=float, default=0.3)
    parser.add_argument("--opacity-lr-scale", type=float, default=0.1)
    parser.add_argument("--gt-resolution", type=int, default=512)
    parser.add_argument("--aa-train", type=int, default=1)
    parser.add_argument("--aa-eval", type=int, default=4)
    parser.add_argument("--edge-grid", type=int, default=2000)
    parser.add_argument("--num-subdivision", type=int, default=20)
    parser.add_argument("--kde-k", type=int, default=11)
    parser.add_argument("--div-eps", type=float, default=1e-15)
    parser.add_argument("--edge-weight", type=float, default=0.02)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument(
        "--zoom-center",
        type=float,
        nargs=2,
        metavar=("X", "Y"),
        default=None,
        help="Optional (x, y) center for a zoomed GIF region.",
    )
    parser.add_argument("--zoom-size", type=float, default=0.1, help="Physical size of the zoom window.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    shader = repo_root / "slang" / "vector_graphics_rgb_padded_accel.slang"
    try:
        compile_if_needed(shader, shader.with_name("__gen__vector_graphics_rgb_padded_accel.slang"))
    except (ModuleNotFoundError, AssertionError) as exc:
        print(f"Warning: shader transformer unavailable or skipped ({exc}); using existing generated shader.")

    integrand = VectorGraphicsRGBPaddedAccelIntegrandSlang(
        primitive_type="circle",
        grid_size=args.accel_grid,
        max_elements_per_cell=args.max_elements,
        num_primitives=args.num_circles,
    ).to(device)

    target = load_target_image(args.image, args.gt_resolution, device)
    with torch.no_grad():
        out_init = render_image(integrand, args.gt_resolution, args.aa_eval, True, device)

    results_dir = repo_root / "results" / "circle_fit"
    run_dir = results_dir / f"{Path(args.image).stem}_N{args.num_circles}"
    run_dir.mkdir(parents=True, exist_ok=True)
    initial_frame = run_dir / "initial.png"
    save_tensor_image(initial_frame, out_init)

    cfg = BoundaryLossConfig(
        grid_size=args.edge_grid,
        num_subdivision=args.num_subdivision,
        kde_k=args.kde_k,
        div_eps=args.div_eps,
        mode="L2_img",
        mode_aux_data=target.detach(),
    )

    optimizer = optim.Adam(
        [
            {"params": [integrand.control_points], "lr": args.lr * args.center_lr_scale},
            {"params": [integrand.stroke_widths], "lr": args.lr * args.radius_lr_scale},
            {"params": [integrand.fill_colors], "lr": args.lr * args.color_lr_scale},
            {"params": [integrand.opacities], "lr": args.lr * args.opacity_lr_scale},
        ],
        betas=(0.9, 0.99),
    )

    loss_history = []
    saved_frames = [initial_frame]
    zoom_frames = []
    zoom_center = tuple(args.zoom_center) if args.zoom_center is not None else None
    if zoom_center is not None:
        with torch.no_grad():
            zoom_initial = render_zoom_image(
                integrand,
                args.gt_resolution,
                args.aa_eval,
                zoom_center,
                args.zoom_size,
                device,
                jitter=False,
            )
        zoom_initial_path = run_dir / "initial_zoom.png"
        save_tensor_image(zoom_initial_path, zoom_initial)

    for step in range(args.num_iter):
        pts = points_on_grid(args.gt_resolution * args.aa_train, jitter=True).to(device)
        preds = integrand(pts).reshape(
            args.gt_resolution,
            args.aa_train,
            args.gt_resolution,
            args.aa_train,
            integrand.out_dim,
        ).mean(dim=(1, 3))

        area_loss = (preds - target).square().mean()
        edge_loss = boundary_loss(integrand, cfg)
        total = area_loss + args.edge_weight * edge_loss
        total.backward()

        optimizer.step()
        optimizer.zero_grad()
        clamp_circle_params(integrand)

        loss_history.append(area_loss.item())
        if step % 10 == 0:
            print(f"Iter {step:04d} | area={area_loss.item():.6f} | edge={edge_loss.item():.6f}")
        if step % args.save_every == 0:
            frame_path = run_dir / f"iter_{step:04d}.png"
            with torch.no_grad():
                frame_img = render_image(integrand, args.gt_resolution, args.aa_eval, True, device)
            save_tensor_image(frame_path, frame_img)
            saved_frames.append(frame_path)
            if zoom_center is not None:
                with torch.no_grad():
                    zoom = render_zoom_image(
                        integrand,
                        args.gt_resolution,
                        args.aa_eval,
                        zoom_center,
                        args.zoom_size,
                        device,
                        jitter=False,
                    )
                zoom_path = run_dir / f"iter_{step:04d}_zoom.png"
                save_tensor_image(zoom_path, zoom)
                zoom_frames.append(zoom_path)

    with torch.no_grad():
        final = render_image(integrand, args.gt_resolution, args.aa_eval, True, device)
    final_frame = run_dir / "final.png"
    save_tensor_image(final_frame, final)
    saved_frames.append(final_frame)
    if zoom_center is not None:
        with torch.no_grad():
            zoom = render_zoom_image(
                integrand,
                args.gt_resolution,
                args.aa_eval,
                zoom_center,
                args.zoom_size,
                device,
                jitter=False,
            )
        zoom_path = run_dir / "final_zoom.png"
        save_tensor_image(zoom_path, zoom)
        zoom_frames.append(zoom_path)

    plt.figure()
    plt.title("Area loss")
    plt.semilogy(loss_history)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.savefig(run_dir / "loss_history.pdf", dpi=300, bbox_inches="tight")
    plt.close()

    frames = [imageio.imread(path) for path in saved_frames if path.exists()]
    if frames:
        imageio.mimsave(run_dir / "optimization.gif", frames, duration=0.2)
    if zoom_center is not None:
        zoom_imgs = [imageio.imread(path) for path in zoom_frames if path.exists()]
        if zoom_imgs:
            imageio.mimsave(run_dir / "optimization_zoom.gif", zoom_imgs, duration=0.2)


if __name__ == "__main__":
    main()
