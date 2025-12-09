import os

import matplotlib.pyplot as plt
import numpy as np
import skimage
import skimage.io
import torch
from .utils.segments import points_on_grid, get_segments_on_grid, snap_segment_to_discontinuity
from .utils.boundary import (
    BoundaryLossConfig,
    boundary_loss,
    edge_loss_slang,
)
from .utils.finite_diff import (
    compute_and_plot_fwd_grad,
    efficient_finite_diff_grad,
    plot_fwd_grad_ours_and_fd,
)
# Set default device to CUDA if available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(DEVICE)  # This sets the default device for all new tensors


import slangtorch
BLOCK_SIZE_1D=32
def launch_1d(x, LEN):
  x.launchRaw(
  blockSize=(BLOCK_SIZE_1D, 1, 1),
  gridSize=(LEN // BLOCK_SIZE_1D + 1, 1, 1)
)
  
def SlangShaderForwardGrad(x, d_x, p, d_p, impl_idx, force_sign, shader, ret_const, ret_impl):
    if impl_idx is None:
        impl_idx = torch.zeros(x.shape[0], dtype=torch.int32) - 1
    r = torch.full_like(x[:,0], -1.0, requires_grad=True)
    impl_fn = torch.full_like(x[:,0], 1000.0, requires_grad=True)
    out_idx = torch.zeros(x.shape[0], dtype=torch.int32)
    d_r = torch.zeros_like(r)
    d_impl_fn = torch.zeros_like(impl_fn)
    
    launch_1d(shader.run.fwd(
      x=(x, d_x), p=(p, d_p), r=(r, d_r), impl_fn=(impl_fn, d_impl_fn), impl_idx=impl_idx, out_idx=out_idx, force_sign=force_sign,
      ret_const=ret_const, ret_impl=ret_impl
    ), LEN=x.shape[0])

    return d_r, d_impl_fn
  
  
def SlangShaderForwardGradRGB(x, d_x, p, d_p, impl_idx, force_sign, shader, ret_const, ret_impl):
    if impl_idx is None:
        impl_idx = torch.zeros(x.shape[0], dtype=torch.int32) - 1
    r = torch.full_like(x[:,0].unsqueeze(-1).repeat(1,3), -1.0, requires_grad=True)
    impl_fn = torch.full_like(x[:,0], 1000.0, requires_grad=True)
    out_idx = torch.zeros(x.shape[0], dtype=torch.int32)
    d_r = torch.zeros_like(r)
    d_impl_fn = torch.zeros_like(impl_fn)
    
    launch_1d(shader.run.fwd(
      x=(x, d_x), p=(p, d_p), r=(r, d_r), impl_fn=(impl_fn, d_impl_fn), impl_idx=impl_idx, out_idx=out_idx, force_sign=force_sign,
      ret_const=ret_const, ret_impl=ret_impl
    ), LEN=x.shape[0])

    return d_r, d_impl_fn
  


  
class SlangShader(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, p, impl_idx, force_sign, shader, ret_const, ret_impl):
        if impl_idx is None:
            impl_idx = torch.zeros(x.shape[0], dtype=torch.int32) - 1
        r = torch.full_like(x[:,0], -1.0, requires_grad=True)
        impl_fn = torch.full_like(x[:,0], 1000.0, requires_grad=True)
        # r = torch.full_like(x[:,0], torch.inf, requires_grad=True)
        out_idx = torch.zeros(x.shape[0], dtype=torch.int32)

        launch_1d(shader.run(
            x=x, p=p, r=r, impl_fn=impl_fn, impl_idx=impl_idx, out_idx=out_idx, force_sign=force_sign,
            ret_const=ret_const, ret_impl=ret_impl
        ), LEN=x.shape[0])
        ctx.save_for_backward(x, p, r, impl_fn, impl_idx, out_idx)
        ctx.shader = shader
        ctx.ret_const = ret_const
        ctx.ret_impl = ret_impl
        ctx.force_sign = force_sign

        return r, impl_fn, impl_idx, out_idx

    @staticmethod
    def backward(ctx, grad_output=None, grad_impl_fn=None, grad_impl_idx=None, grad_out_idx=None):
        x, p, r, impl_fn, impl_idx, out_idx = ctx.saved_tensors
        shader = ctx.shader
        ret_const = ctx.ret_const
        ret_impl = ctx.ret_impl
        force_sign = ctx.force_sign
        d_x = torch.zeros_like(x)
        d_p = torch.zeros_like(p)

        impl_fn = torch.full_like(x[:,0], 1000.0, requires_grad=True)
        
        if grad_output is not None:
            d_r = grad_output.contiguous()
        else:
            d_r = torch.zeros_like(r)
        
        if grad_impl_fn is not None:
            d_impl_fn = grad_impl_fn.contiguous()
        else:
            d_impl_fn = torch.zeros_like(impl_fn)
        
        # TODO: In backward pass for discontinuity, force_sign should be -1, ret_const should be False
        # and ret_impl should be True. maybe add assertions for this
        
        launch_1d(shader.run.bwd(
        x=(x, d_x), p=(p, d_p), r=(r, d_r), impl_fn=(impl_fn, d_impl_fn), impl_idx=impl_idx, out_idx=out_idx, force_sign=force_sign,
        ret_const=ret_const, ret_impl=ret_impl
        ), LEN=x.shape[0])

        return d_x, d_p, None, None, None, None, None

class SlangShaderRGB(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, p, impl_idx, force_sign, shader, ret_const, ret_impl):
        if impl_idx is None:
            impl_idx = torch.zeros(x.shape[0], dtype=torch.int32) - 1
        r = torch.full_like(x[:,0].unsqueeze(-1).repeat(1,3), -1.0, requires_grad=True)
        impl_fn = torch.full_like(x[:,0], 1000.0, requires_grad=True)
        # r = torch.full_like(x[:,0], torch.inf, requires_grad=True)
        out_idx = torch.zeros(x.shape[0], dtype=torch.int32)

        launch_1d(shader.run(
            x=x, p=p, r=r, impl_fn=impl_fn, impl_idx=impl_idx, out_idx=out_idx, force_sign=force_sign,
            ret_const=ret_const, ret_impl=ret_impl
        ), LEN=x.shape[0])
        ctx.save_for_backward(x, p, r, impl_fn, impl_idx, out_idx)
        ctx.shader = shader
        ctx.ret_const = ret_const
        ctx.ret_impl = ret_impl
        ctx.force_sign = force_sign

        return r, impl_fn, impl_idx, out_idx

    @staticmethod
    def backward(ctx, grad_output=None, grad_impl_fn=None, grad_impl_idx=None, grad_out_idx=None):
        x, p, r, impl_fn, impl_idx, out_idx = ctx.saved_tensors
        shader = ctx.shader
        ret_const = ctx.ret_const
        ret_impl = ctx.ret_impl
        force_sign = ctx.force_sign
        d_x = torch.zeros_like(x)
        d_p = torch.zeros_like(p)

        impl_fn = torch.full_like(x[:,0], 1000.0, requires_grad=True)
        
        if grad_output is not None:
            d_r = grad_output.contiguous()
        else:
            d_r = torch.zeros_like(r)
        
        if grad_impl_fn is not None:
            d_impl_fn = grad_impl_fn.contiguous()
        else:
            d_impl_fn = torch.zeros_like(impl_fn)
        
        # TODO: In backward pass for discontinuity, force_sign should be -1, ret_const should be False
        # and ret_impl should be True. maybe add assertions for this
        
        launch_1d(shader.run.bwd(
        x=(x, d_x), p=(p, d_p), r=(r, d_r), impl_fn=(impl_fn, d_impl_fn), impl_idx=impl_idx, out_idx=out_idx, force_sign=force_sign,
        ret_const=ret_const, ret_impl=ret_impl
        ), LEN=x.shape[0])

        return d_x, d_p, None, None, None, None, None







class Camera:
    def __init__(self, fov=100, aspect_ratio=1.0):
        self.fov = fov
        self.aspect_ratio = aspect_ratio
        self.reset()

    def reset(self):
        self.lookfrom = torch.tensor([0.0, 0.5, -10.0], device=DEVICE)
        self.lookat = torch.tensor([0.0, 0.0, 0.0], device=DEVICE)
        
        angle = torch.tensor(torch.pi / 16, device=DEVICE)
        self.vup = torch.tensor([torch.sin(angle), torch.cos(angle), 0.0], device=DEVICE)
        
        theta = torch.tensor(self.fov * (torch.pi / 180.0), device=DEVICE)
        half_height = torch.tan(theta / 2.0)
        half_width = self.aspect_ratio * half_height
        
        self.cam_origin = self.lookfrom.clone()
        
        w = (self.lookfrom - self.lookat) / torch.norm(self.lookfrom - self.lookat)
        u = torch.cross(self.vup, w, dim=0)
        u = u / torch.norm(u)
        v = torch.cross(w, u, dim=0)
        
        self.cam_lower_left_corner = self.cam_origin - half_width * u - half_height * v - w
        self.cam_horizontal = 2 * half_width * u
        self.cam_vertical = 2 * half_height * v

    def serialize(self):
        params = torch.zeros(3 + 3 + 3 + 3, device=DEVICE)
        params[:3] = self.cam_origin
        params[3:6] = self.cam_lower_left_corner
        params[6:9] = self.cam_horizontal
        params[9:] = self.cam_vertical
        return params
    
def imwrite(img, filename, gamma = 2.2, normalize = False):
    """
    Save an image to a file with the origin at the lower left corner.
    Taken from https://github.com/BachiLi/diffvg/blob/85802a71fbcc72d79cb75716eb4da4392fd09532/pydiffvg/image.py
    """
    directory = os.path.dirname(filename)
    if directory != '' and not os.path.exists(directory):
        os.makedirs(directory)

    # set origin to lower left corner
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu()
        if img.dim() == 3:
            img = img.flip(0)
        elif img.dim() == 4:
            img = img.flip(1)
    else:
        img = np.flipud(img)

    if not isinstance(img, np.ndarray):
        img = img.data.numpy()
    if normalize:
        img_rng = np.max(img) - np.min(img)
        if img_rng > 0:
            img = (img - np.min(img)) / img_rng
    img = np.clip(img, 0.0, 1.0)
    if img.ndim==2:
        img=np.expand_dims(img,2)
    img[:, :, :3] = np.power(img[:, :, :3], 1.0/gamma)
    skimage.io.imsave(filename, (img * 255).astype(np.uint8))

def vidwrite(imgs, filename, fps=1):
    import tempfile
    import subprocess
    import os
    import numpy as np
    
    # Create a temporary directory to store frames
    with tempfile.TemporaryDirectory() as tmpdir:
        # Save each frame as a PNG file
        for i, img in enumerate(imgs):
            frame_path = os.path.join(tmpdir, f"frame_{i:05d}.png")
            imwrite(img, frame_path)
        
        # Use ffmpeg to create a GIF from the frames
        cmd = [
            "ffmpeg", 
            "-y",  # Overwrite output file if it exists
            "-f", "image2",  # Force input format to image2
            "-framerate", str(fps),  # Set the frame rate
            "-i", os.path.join(tmpdir, "frame_%05d.png"),  # Input pattern
            filename  # Output file
        ]
        
        subprocess.run(cmd, check=True)
    
