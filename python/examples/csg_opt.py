import os
import math
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import sys

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))

from compiler.compile_shader import compile_if_needed  # noqa: E402
from python.integrands import BaseIntegrandSlang  # noqa: E402
from python.utils.boundary import BoundaryLossConfig, boundary_loss  # noqa: E402
from python.utils.segments import points_on_grid  # noqa: E402
import slangtorch  # noqa: E402


def create_3d_slice_samples(resolution, aa_factor, plane_value, constant_axis=2):
    """Create a 3D slice of points where one axis is held constant."""
    area_samples = points_on_grid(resolution * aa_factor, jitter=True, dim=2)
    if constant_axis == 0:
        area_samples = torch.cat([torch.ones_like(area_samples[:, :1]) * plane_value, area_samples], dim=1)
        axis_name = "x"
    elif constant_axis == 1:
        area_samples = torch.cat(
            [area_samples[:, :1], torch.ones_like(area_samples[:, :1]) * plane_value, area_samples[:, 1:]], dim=1
        )
        axis_name = "y"
    else:
        area_samples = torch.cat([area_samples, torch.ones_like(area_samples[:, :1]) * plane_value], dim=1)
        axis_name = "z"
    return area_samples, axis_name


def plot_comparison(out_gt, out_perturbed, save_path=None, middle_title="Perturbed Image"):
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(out_gt.detach().cpu().numpy(), extent=[0, 1, 0, 1], origin="lower")
    plt.title("Original Image")
    plt.axis("off")
    plt.subplot(1, 3, 2)
    plt.imshow(out_perturbed.detach().cpu().numpy(), extent=[0, 1, 0, 1], origin="lower")
    plt.title(middle_title)
    plt.axis("off")
    plt.subplot(1, 3, 3)
    diff_image = (out_perturbed - out_gt).abs().detach().cpu().numpy()
    plt.imshow(diff_image, extent=[0, 1, 0, 1], origin="lower")
    plt.title("Difference Image")
    plt.axis("off")
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()
    plt.close()


class CSGIntegrandSlang(BaseIntegrandSlang):
    def __init__(self):
        super().__init__()
        self.shader = slangtorch.loadModule("slang/__gen__csg.slang")

        params = []
        params.append(1.0)  # sphere_radius
        params.extend([0.0, 0.0, 0.0])  # sphere_position
        params.append(0.7)  # cube_size
        params.extend([0.0, 0.0, 0.0])  # cube_position
        params.append(0.6)  # cylinder_z_radius
        params.extend([0.0, 0.0, 0.0])  # cylinder_z_position
        params.append(0.6)  # cylinder_x_radius
        params.extend([0.0, 0.0, 0.0])  # cylinder_x_position
        params.append(0.6)  # cylinder_y_radius
        params.extend([0.0, 0.0, 0.0])  # cylinder_y_position

        self.p = nn.Parameter(torch.tensor(params))


def main():
    # Use pre-generated shader; do not re-run transformer for CSG until preproc handling is fixed

    # Config
    USE_TEST_FN = True
    EDGE_LOSS_MODE = "L2_test_fn" if USE_TEST_FN else "L2_img"
    GT_RESOLUTION = 512
    AA_FACTOR_GT = 1
    LR = 1e-2
    NUM_ITER = 400
    AREA_SAMPLE_RES = 128
    EDGE_SAMPLE_RES = 500
    NUM_SUBDIVISION = 20
    KDE_K = 14

    # Integrands
    test_fn = CSGIntegrandSlang()
    integrand = CSGIntegrandSlang()
    torch.manual_seed(0)
    with torch.no_grad():
        integrand.p += torch.randn_like(integrand.p) * 0.07

    # Initial visualization on a slice
    plane_value = 0.25
    constant_axis = 2
    area_samples, _ = create_3d_slice_samples(GT_RESOLUTION, AA_FACTOR_GT, plane_value, constant_axis)
    out_gt = test_fn(area_samples).reshape(GT_RESOLUTION, AA_FACTOR_GT, GT_RESOLUTION, AA_FACTOR_GT).mean(dim=(1, 3))
    out_init = integrand(area_samples).reshape(GT_RESOLUTION, AA_FACTOR_GT, GT_RESOLUTION, AA_FACTOR_GT).mean(dim=(1, 3))

    results_dir = repo_root / "results" / "csg_opt"
    results_dir.mkdir(parents=True, exist_ok=True)
    plot_comparison(out_gt, out_init, save_path=results_dir / "initial.png", middle_title="Initial Image")
    torch.save(integrand.state_dict(), results_dir / "initial.pt")

    optimizer = optim.Adam(integrand.parameters(), lr=LR, betas=(0.9, 1 - (1 - 0.9) ** 2))
    lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.05 ** (1 / NUM_ITER))

    boundary_cfg = BoundaryLossConfig(
        dim=3,
        grid_size=EDGE_SAMPLE_RES,
        num_subdivision=NUM_SUBDIVISION,
        kde_k=KDE_K,
        mode=EDGE_LOSS_MODE,
        mode_aux_data=test_fn,
        df_dx_mode="backward",
    )

    for i in range(NUM_ITER):
        torch.save(integrand.state_dict(), results_dir / f"step_{i}.pt")
        optimizer.zero_grad()

        area_samples = points_on_grid(AREA_SAMPLE_RES * AA_FACTOR_GT, jitter=True, dim=3)
        out = integrand(area_samples)
        out_gt_batch = test_fn(area_samples)
        loss = (out - out_gt_batch).square().mean()

        edge_loss = boundary_loss(integrand, boundary_cfg)
        total_loss = edge_loss  # or loss + edge_loss if area term desired
        total_loss.backward()

        optimizer.step()
        lr_scheduler.step()

        if i % 10 == 0:
            print(f"Iter {i}: loss={loss.item():.6f}, edge={edge_loss.item():.6f}")

    # Final visualization
    area_samples, _ = create_3d_slice_samples(GT_RESOLUTION, AA_FACTOR_GT, plane_value, constant_axis)
    out_final = integrand(area_samples).reshape(GT_RESOLUTION, AA_FACTOR_GT, GT_RESOLUTION, AA_FACTOR_GT).mean(dim=(1, 3))
    plot_comparison(out_gt, out_final, save_path=results_dir / "final.png", middle_title="Final Image")
    torch.save(integrand.state_dict(), results_dir / "final.pt")


if __name__ == "__main__":
    main()
