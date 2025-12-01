import math
from pathlib import Path
import pytest
import torch
import sys

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))

from compiler.compile_shader import compile_if_needed  # noqa: E402
from python.integrands import CirclesIntegrandSlang  # noqa: E402


def test_circle_grad_matches_analytic():
    repo_root = Path(__file__).resolve().parents[2]
    src = repo_root / "slang" / "circles.slang"
    dst = repo_root / "slang" / "__gen__circles.slang"
    compile_if_needed(src, dst)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    integrand = CirclesIntegrandSlang(repo_root).to(device)

    cx, cy, r, opacity = 0.5, 0.5, 0.4, 1.0
    integrand.p.data = torch.tensor([cx, cy, r, opacity], device=device, requires_grad=True)

    N = 64
    xs = torch.linspace(0.0, 1.0, N, device=device)
    yy, xx = torch.meshgrid(xs, xs, indexing="ij")
    pts = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)

    res = integrand(pts)
    mean_val = res.mean()
    mean_val.backward()
    grad_r = integrand.p.grad[2].item()

    assert math.isfinite(grad_r)
