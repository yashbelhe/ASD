"""
Multi-view triangle mesh optimization using the vector-graphics Slang shader.

This ports the original `triangle_rasterizer_opt.py` training loop into the new
repository layout (python/examples) with CLI args, caching, and result dumps.
"""

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))

from compiler.compile_shader import compile_if_needed  # noqa: E402
from python.integrands import TriangleRasterizerIntegrandSlang  # noqa: E402
from python.helpers import BoundaryLossConfig, boundary_loss_slang, points_on_grid  # noqa: E402


def resolve_path(path_str):
    path = Path(path_str)
    if path.is_absolute():
        return path
    for root in (repo_root, repo_root.parent):
        candidate = root / path
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not locate '{path_str}'.")


def render_with_samples(integrand, samples, resolution, aa_factor):
    preds = integrand(samples)
    return preds.reshape(resolution, aa_factor, resolution, aa_factor, integrand.out_dim).mean(dim=(1, 3))


def save_tensor_image(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(path, image.detach().cpu().numpy().clip(0.0, 1.0), origin="lower", dpi=300)


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
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def average_edge_length(verts, faces):
    face_verts = verts[faces]
    v0, v1, v2 = face_verts[:, 0], face_verts[:, 1], face_verts[:, 2]
    a = (v1 - v2).norm(dim=1)
    b = (v0 - v2).norm(dim=1)
    c = (v0 - v1).norm(dim=1)
    return (a + b + c).sum() / faces.shape[0] / 3.0


def try_remesh(integrand, step, interval):
    if interval <= 0 or (step + 1) % interval != 0:
        return False
    try:
        import gpytoolbox as gpy  # type: ignore
    except ImportError:
        print("gpytoolbox not available; skipping remeshing.")
        return False

    with torch.no_grad():
        verts = integrand.vertices.detach().cpu()
        faces = integrand.faces.detach().cpu()
        target_length = average_edge_length(verts, faces).cpu().item() * 0.5
        v_new, f_new = gpy.remesh_botsch(
            verts.numpy().astype(np.double),
            faces.numpy().astype(np.int32),
            5,
            target_length,
            True,
        )
        v_tensor = torch.from_numpy(v_new)
        f_tensor = torch.from_numpy(f_new)
        integrand.reset_mesh(v_tensor, f_tensor)
    print("Remeshed surface to", f_tensor.shape[0], "faces.")
    return True


def build_optimizer(param, lr):
    try:
        from largesteps.optimize import AdamUniform  # type: ignore

        print("Using largesteps.optimize.AdamUniform.")
        return AdamUniform([param], lr)
    except Exception as exc:  # pragma: no cover - fallback path
        print(f"AdamUniform unavailable ({exc}); falling back to torch.optim.Adam.")
        return torch.optim.Adam([param], lr=lr)


def parse_args():
    parser = argparse.ArgumentParser(description="Optimize a triangle mesh via the Slang rasterizer.")
    parser.add_argument("--target-ply", type=str, default="data/scenes/suzanne/meshes/target.ply")
    parser.add_argument("--source-ply", type=str, default="data/scenes/suzanne/meshes/source.ply")
    parser.add_argument("--target-scale", type=float, default=0.5, help="Normalization scale for the target mesh.")
    parser.add_argument("--source-scale", type=float, default=0.125, help="Normalization scale for the source mesh.")
    parser.add_argument("--grid-size", type=int, default=1000, help="Acceleration grid size.")
    parser.add_argument("--max-elements", type=int, default=400, help="Max primitives per grid cell.")
    parser.add_argument("--num-views", type=int, default=500, help="Number of random camera views.")
    parser.add_argument("--views-per-step", type=int, default=50, help="How many view grads to accumulate per step.")
    parser.add_argument("--num-iters", type=int, default=16000, help="Total per-view iterations.")
    parser.add_argument("--gt-resolution", type=int, default=512)
    parser.add_argument("--aa-train", type=int, default=1)
    parser.add_argument("--aa-eval", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--initial-view", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=0, help="Save intermediate renders every N optimizer steps.")
    parser.add_argument("--remesh-interval", type=int, default=4000, help="Set <=0 to disable remeshing.")
    parser.add_argument("--boundary-grid-size", type=int, default=2000)
    parser.add_argument("--boundary-num-subdivision", type=int, default=20)
    parser.add_argument("--boundary-kde-k", type=int, default=14)
    parser.add_argument("--boundary-div-eps", type=float, default=1e-15)
    parser.add_argument("--boundary-plot-resolution", type=int, default=1000)
    parser.add_argument("--boundary-lipschitz", type=float, default=1e-6)
    parser.add_argument("--boundary-mode", type=str, default="L1_img")
    parser.add_argument("--results-root", type=str, default="results/triangle_rasterizer_opt")
    parser.add_argument("--results-name", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    shader = repo_root / "slang" / "vector_graphics_rgb_padded_accel.slang"
    try:
        compile_if_needed(shader, shader.with_name("__gen__vector_graphics_rgb_padded_accel.slang"))
    except (ModuleNotFoundError, AssertionError) as exc:
        print(f"Warning: shader transformer unavailable or skipped ({exc}); using existing generated shader.")

    target_path = resolve_path(args.target_ply)
    source_path = resolve_path(args.source_ply)

    integrand_target = TriangleRasterizerIntegrandSlang(
        target_path,
        grid_size=args.grid_size,
        max_elements_per_cell=args.max_elements,
        scale=args.target_scale,
        res=args.gt_resolution,
        num_view=args.num_views,
        seed=args.seed,
    ).to(device)
    integrand_target.requires_grad_(False)

    integrand = TriangleRasterizerIntegrandSlang(
        source_path,
        grid_size=args.grid_size,
        max_elements_per_cell=args.max_elements,
        scale=args.source_scale,
        res=args.gt_resolution,
        num_view=args.num_views,
        seed=args.seed,
    ).to(device)

    results_root = repo_root / args.results_root
    run_name = args.results_name or f"{source_path.stem}_to_{target_path.stem}_views{args.num_views}_iter{args.num_iters}"
    run_dir = results_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    eval_samples = points_on_grid(args.gt_resolution * args.aa_eval, jitter=False).to(device)
    target_samples = points_on_grid(args.gt_resolution * args.aa_train, jitter=True).to(device)

    integrand_target.set_active_view(args.initial_view)
    integrand.set_active_view(args.initial_view)

    target_imgs = []
    with torch.no_grad():
        for view_idx in range(args.num_views):
            integrand_target.set_active_view(view_idx)
            img = render_with_samples(
                integrand_target,
                target_samples,
                args.gt_resolution,
                args.aa_train,
            ).detach().cpu()
            target_imgs.append(img)
    integrand_target.set_active_view(args.initial_view)
    target_img0 = target_imgs[args.initial_view]

    init_img = render_with_samples(integrand, target_samples, args.gt_resolution, args.aa_train).detach()
    save_tensor_image(run_dir / "initial.png", init_img)
    plot_comparison(target_img0, init_img.detach().cpu(), save_path=run_dir / "initial_comparison.png")
    del target_samples

    cfg = BoundaryLossConfig(
        grid_size=args.boundary_grid_size,
        num_subdivision=args.boundary_num_subdivision,
        kde_k=args.boundary_kde_k,
        div_eps=args.boundary_div_eps,
        plot_resolution=args.boundary_plot_resolution,
        lipschitz_bounds=[args.boundary_lipschitz] * args.boundary_num_subdivision,
        mode=args.boundary_mode,
        df_dx_mode="backward",
    )

    optimizer = build_optimizer(integrand.u, args.lr)
    optimizer.zero_grad()

    loss_history = []
    accum_counter = 0
    block_loss = 0.0

    for step in range(args.num_iters):
        if try_remesh(integrand, step, args.remesh_interval):
            optimizer = build_optimizer(integrand.u, args.lr)
            optimizer.zero_grad()

        view_idx = step % args.num_views
        integrand.set_active_view(view_idx)
        area_samples = points_on_grid(args.gt_resolution * args.aa_train, jitter=True).to(device)
        preds = render_with_samples(integrand, area_samples, args.gt_resolution, args.aa_train)
        target_img_gpu = target_imgs[view_idx].to(device)
        pixel_loss = (preds - target_img_gpu).square().mean()
        cfg.mode_aux_data = target_img_gpu.detach()
        edge_loss = boundary_loss_slang(integrand, cfg)
        cfg.mode_aux_data = None
        del target_img_gpu
        total_loss = pixel_loss + edge_loss
        total_loss.backward()

        accum_counter += 1
        block_loss += pixel_loss.item()

        if accum_counter == args.views_per_step or step == args.num_iters - 1:
            optimizer.step()
            optimizer.zero_grad()
            avg_loss = block_loss / accum_counter
            loss_history.append(avg_loss)
            print(f"Iter {step + 1:05d} | view {view_idx:04d} | pixel={avg_loss:.6f} | edge={edge_loss.item():.6f}")

            if args.save_every > 0 and len(loss_history) % args.save_every == 0:
                prev = integrand.active_view
                integrand.set_active_view(args.initial_view)
                snapshot = render_with_samples(integrand, eval_samples, args.gt_resolution, args.aa_eval)
                save_tensor_image(run_dir / f"iter_{step + 1:05d}.png", snapshot)
                integrand.set_active_view(prev)

            accum_counter = 0
            block_loss = 0.0

    integrand.set_active_view(args.initial_view)
    final_img = render_with_samples(integrand, eval_samples, args.gt_resolution, args.aa_eval).detach()
    save_tensor_image(run_dir / "final.png", final_img)
    plot_comparison(target_img0.detach().cpu(), final_img.detach().cpu(), initial=init_img.detach().cpu(), save_path=run_dir / "final_comparison.png")

    torch.save(
        {
            "vertices": integrand.vertices.detach().cpu(),
            "faces": integrand.faces.detach().cpu(),
        },
        run_dir / "optimized_mesh.pt",
    )

    plt.figure()
    plt.semilogy(loss_history)
    plt.xlabel("Optimizer step")
    plt.ylabel("Pixel loss")
    plt.savefig(run_dir / "loss_history.png", dpi=300, bbox_inches="tight")
    plt.close()

    np.savetxt(run_dir / "loss_history.txt", np.array(loss_history))
    print("Results written to", run_dir)


if __name__ == "__main__":
    main()
