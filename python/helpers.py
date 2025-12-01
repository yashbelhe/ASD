import numpy as np
import skimage
import skimage.io
import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from dataclasses import dataclass
# Set default device to CUDA if available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(DEVICE)  # This sets the default device for all new tensors

def points_on_grid(GRID_SIZE, jitter=False, dim=2, device=None):
    # Create linspace for each dimension
    coords = [torch.linspace(0, 1, GRID_SIZE + 1, device=device)[:-1] for _ in range(dim)]

    # Create meshgrid for all dimensions
    meshgrids = torch.meshgrid(*coords, indexing='xy')

    # Apply jitter if requested
    if jitter:
        meshgrids = [grid + torch.rand_like(grid) / GRID_SIZE for grid in meshgrids]

    # Stack and flatten all coordinates
    points = torch.stack([grid.flatten() for grid in meshgrids], dim=1)

    return points

# https://github.com/getkeops/keops/issues/73 is FUCKING AWESOME!!!
from pykeops.torch import LazyTensor
def ranges_slices(batch):
  """Helper function for the diagonal ranges function."""
  Ns = batch.bincount()
  indices = Ns.cumsum(0)
  ranges = torch.cat((0 * indices[:1], indices))
  ranges = torch.stack((ranges[:-1], ranges[1:])).t().int().contiguous().to(batch.device)
  slices = (1 + torch.arange(len(Ns))).int().to(batch.device)
  
  return ranges, slices


def diagonal_ranges(batch_x = None, batch_y = None):
  """Encodes the block-diagonal structure associated to a batch vector."""
  
  if batch_x is None and batch_y is None: return None
  
  ranges_x, slices_x = ranges_slices(batch_x)
  ranges_y, slices_y = ranges_slices(batch_y)
  
  return ranges_x, slices_x, ranges_y, ranges_y, slices_y, ranges_x 


class KDEKeOpsKNN(nn.Module):
    def __init__(self, do_KDE=True):
        super().__init__()
        self.do_KDE = do_KDE
        if not do_KDE:
            assert False
  
    def forward(self, x, min_t_idx, K, sz=5):
        min_t_idx = min_t_idx.to(torch.int64)
        from pykeops.torch import LazyTensor
        with torch.no_grad():
            _, NI = x.shape
            # Do KDE based on this: https://math.stackexchange.com/questions/4028633/which-is-the-algorithm-for-knn-density-estimator
            # K = 20
            # elems, counts = min_t_idx.unique(return_counts=True)
            p = torch.zeros(x.shape[0], device=x.device)

            m = min_t_idx > 0 # assume idx 0 and below is out of domain bounds

            x_ = x[m]
            min_t_idx_m = min_t_idx[m]
            
            # Handle 2D and 3D cases separately
            if NI == 2:
                # Do KDE independently for each if statement and sz^2 grid cell
                # (this reduces the memory requirements)
                y = (x_ * sz).long() # 2D indices
                y_f = y[:,0] * sz + y[:,1] # 1D indices
                y_f += (min_t_idx_m) * sz**2 # different grids for different discontinuous neurons
            elif NI == 3:
                # 3D version - map 3D grid cells to 1D indices
                y = (x_ * sz).long() # 3D indices
                y_f = y[:,0] * sz**2 + y[:,1] * sz + y[:,2] # 1D indices
                y_f += (min_t_idx_m) * sz**3 # different grids for different discontinuous neurons
            else:
                assert False, f"Only 2D and 3D are supported, got dimension {NI}"
            
            sort_idx = torch.argsort(y_f)
            
            inv_sort_idx = torch.zeros_like(sort_idx)
            inv_sort_idx[sort_idx] = torch.arange(sort_idx.shape[0], device=sort_idx.device)
            assert torch.all(x_[sort_idx][inv_sort_idx] == x_)

            x_, y_f = x_[sort_idx], y_f[sort_idx]

            # Symbolic matrix of squared distances:
            x_i = LazyTensor(x_[:,None,:])    # (B*N, 1, D)
            y_j = LazyTensor(x_[None,:,:])    # (1, B*M, D)
            
            D_ij = ((x_i - y_j)**2).sum(-1)  # (B*N, B*M)
                
            # Apply a block-diagonal sparsity mask:
            D_ij.ranges = diagonal_ranges(y_f, y_f)

            # KDE_MODE = 'Epanechnikov'
            KDE_MODE = 'KNN'
            if KDE_MODE == 'KNN':
                # K-NN search with heterogeneous batches:
                idx_i = D_ij.argKmin(K, dim=1)

                knn = x_[idx_i[:,-1]]
                knn_dist = (x_ - knn).square().sum(axis=-1).sqrt()

                # If the kth nearest neighbor is the el
                w = knn_dist
                p_m = torch.zeros(x_.shape[0], device=x.device)
                if NI - 1 == 1:
                    p_m = w * 2.0 / (K - 1)
                elif NI - 1 == 2:
                    p_m = w**2 * torch.pi / (K - 1)
                elif NI - 1 == 3:
                    p_m = 4/3 * w**3 * torch.pi / (K - 1)
            elif KDE_MODE == 'Epanechnikov':
                assert False, "This is not correct"
                # K-NN search with heterogeneous batches:
                idx_i = D_ij.argKmin(K, dim=1)
                
                # Get distances to all k neighbors
                knn_dists = torch.zeros((x_.shape[0], K), device=x.device)
                for k in range(K):
                    knn = x_[idx_i[:,k]]
                    knn_dists[:,k] = (x_ - knn).square().sum(axis=-1).sqrt()
                
                # Use distance to kth neighbor as bandwidth (same as KNN)
                w = knn_dists[:,-1]
                
                # Calculate Epanechnikov kernel values for each neighbor
                # K(u) = (3/4)(1-u²) for |u| ≤ 1, 0 otherwise
                u = knn_dists / w.unsqueeze(-1)  # Normalize distances by kth neighbor distance
                kernel_vals = torch.where(u <= 1, 0.75 * (1 - u**2), torch.zeros_like(u))
                
                # For inverse probability, use the same formula as KNN
                # but scale by the sum of kernel values to maintain consistent scale
                # Note: kernel_vals are reduced by 0.75, so we need to scale up by 4/3
                kernel_sum = kernel_vals.sum(dim=1)
                if NI - 1 == 1:
                    p_m = w * 2.0 / (K - 1) * (K / kernel_sum) * (4/3)
                elif NI - 1 == 2:
                    p_m = w**2 * torch.pi / (K - 1) * (K / kernel_sum) * (4/3)
                elif NI - 1 == 3:
                    p_m = 4/3 * w**3 * torch.pi / (K - 1) * (K / kernel_sum) * (4/3)
            p[m] = p_m[inv_sort_idx]
            # print(p[m])
            return p



class KDETorchKNN(nn.Module):
    def __init__(self, uniform_KDE=False):
        super().__init__()
        self.uniform_KDE = uniform_KDE
        if uniform_KDE:
            print("WARNING: We are NOT doing KDE to compute probabilites in the boundary integral, this can make results worse.")
  

    def forward(self, x, min_t_idx, K, sz=None):
        from pykeops.torch import LazyTensor
        _, NI = x.shape
        p = None
        with torch.no_grad():
            # Do KDE based on this: https://math.stackexchange.com/questions/4028633/which-is-the-algorithm-for-knn-density-estimator
            # K = 20
            elems, counts = min_t_idx.unique(return_counts=True)
            p = torch.zeros(x.shape[0], device=x.device)
            for e, c in zip(elems, counts):
                m = min_t_idx == e # TODO: is this the same as counts?
                N = torch.sum(m)
                # p[m] = 1/c; continue
                if N < K or NI - 1 == 0:
                    # for 1d functions, the probability is the same for all samples landing on the same discontinuity
                    p[m] = 1/c
                    continue
                if not self.uniform_KDE:
                    dist = (x[m].unsqueeze(1) - x[m].unsqueeze(0)).square().sum(axis=2).sqrt()
                    knn_dist, _ = torch.kthvalue(dist, k=K)
                    w = knn_dist
                    if NI - 1 == 1:
                        p[m] = w * 2.0 / (K - 1)
                    elif NI - 1 == 2:
                        p[m] = w**2 * torch.pi / (K - 1)
                    elif NI - 1 == 3:
                        p[m] = 4/3 * w**3 * torch.pi / (K - 1)
                else:
                    p[m] = 1/c

        return p

def l2_norm(x):
    return torch.norm(x, p=2, dim=1)

def target_diff_segment(segments, target_fn):
    # segments: [N, 2, 2 or 3]
    return target_fn(segments[:, 0]) - target_fn(segments[:, 1])

def diff_segment(segments):
    # segments: [N, 2, 2 or 3]
    return segments[:, 0] - segments[:, 1]

def segment_grad_norm(segments, target_fn):
    num = target_diff_segment(segments, target_fn)
    den = diff_segment(segments)
    return num.abs() / l2_norm(den)

# TODO: Figure out whether uniform is better or random or stratified
def get_segments_on_grid(GRID_SIZE, dim=2):
    # Sample points on grid and create random directions on unit sphere/circle
    grid = points_on_grid(GRID_SIZE, jitter=True, dim=dim)
    
    if dim == 2:
        # For 2D: sample on unit circle
        theta = torch.rand_like(grid[:,0]) * torch.pi * 2
        # random_dir = torch.stack([torch.cos(theta), torch.sin(theta)], dim=1)/GRID_SIZE
        # random_dir = torch.stack([torch.cos(theta), torch.sin(theta)], dim=1)/GRID_SIZE*1.42
        random_dir = torch.stack([torch.cos(theta), torch.sin(theta)], dim=1)/GRID_SIZE * torch.rand_like(grid[:,0]).unsqueeze(-1)
    elif dim == 3:
        # For 3D: sample on unit sphere using spherical coordinates
        theta = torch.rand_like(grid[:,0]) * torch.pi * 2  # azimuthal angle
        phi = torch.acos(2 * torch.rand_like(grid[:,0]) - 1)  # polar angle
        
        x = torch.sin(phi) * torch.cos(theta)
        y = torch.sin(phi) * torch.sin(theta)
        z = torch.cos(phi)
        
        random_dir = torch.stack([x, y, z], dim=1)/GRID_SIZE
    else:
        assert False, "Only 2D and 3D are supported for get_segments_on_grid"
        # For higher dimensions, use normalized random vectors
        random_dir = torch.randn(grid.shape, device=grid.device)
        random_dir = random_dir / torch.norm(random_dir, dim=1, keepdim=True) / GRID_SIZE
    
    segments_0 = torch.stack([grid, grid + random_dir], dim=1)
    return segments_0


def snap_segment_to_discontinuity(segments, target_fn, LIPSCHITZ_BOUNDS, NUM_SUBDIVISION, max_segments=10**7):
    num_points = 0
    for i in range(NUM_SUBDIVISION):
        num_points += segments.shape[0] * 2
        # print(f"[{i}] Num active segments: {len(segments)}")
        seg_grad_norms = segment_grad_norm(segments, target_fn)
        segments_mask = seg_grad_norms > LIPSCHITZ_BOUNDS[i]
        segments = segments[segments_mask]
        midpoints = segments.mean(axis=1)
        if i == NUM_SUBDIVISION-1:
            break
        segments_first = torch.stack([segments[:,0], midpoints], dim=1)
        segments_second = torch.stack([midpoints, segments[:,1]], dim=1)
        segments = torch.cat([segments_first, segments_second], dim=0)
        if segments.shape[0] > max_segments:
            # Randomly select max_segments points if there are too many
            indices = torch.randperm(segments.shape[0], device=segments.device)[:max_segments]
            segments = segments[indices]
            print(f"WARNING: Reduced number of segments from {segments.shape[0]} to {segments.shape[0]}")
    # print(f"Num sampled points: {num_points}")
    return segments

# def snap_segment_to_discontinuity(segments, target_fn, LIPSCHITZ_BOUNDS, NUM_SUBDIVISION, max_segments=10**7):
#     num_points = 0
#     completed_segments = []
#     for i in range(NUM_SUBDIVISION):
#         num_points += segments.shape[0] * 2
#         if segments.shape[0] == 0:
#             break
#         # print(f"[{i}] Num active segments: {len(segments)}")
#         seg_grad_norms = segment_grad_norm(segments, target_fn)
#         segments_mask = seg_grad_norms > LIPSCHITZ_BOUNDS[i]
#         segments = segments[segments_mask]
#         midpoints = segments.mean(axis=1)
#         if i == NUM_SUBDIVISION-1:
#             break
#         segments_first = torch.stack([segments[:,0], midpoints], dim=1)
#         segments_second = torch.stack([midpoints, segments[:,1]], dim=1)
#         segments = torch.cat([segments_first, segments_second], dim=0)
#         if segments.shape[0] > max_segments:
#             # Randomly select max_segments points if there are too many
#             indices = torch.randperm(segments.shape[0], device=segments.device)[:max_segments]
#             segments = segments[indices]
#             print(f"WARNING: Reduced number of segments from {segments.shape[0]} to {segments.shape[0]}")
#         segment_lengths = (segments[:,0] - segments[:,1]).norm(dim=1)
#         done_mask = segment_lengths < 1e-6
#         completed_segments.append(segments[done_mask])
#         segments = segments[~done_mask]
#         # print(segments.shape[0])
#     completed_segments.append(segments)
#     print(f"Num sampled points: {num_points}")
#     return torch.cat(completed_segments, dim=0)

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

class PathGuidingDataStructure2D():
    def __init__(self, n=50, ema_alpha=0.9):
        self.data = torch.ones((n, n))
        self.ema_alpha = ema_alpha
    
    def sample(self, num_samples):
        """
        Generate samples from the 2D probability distribution represented by self.data.
        
        Args:
            num_samples: Target number of samples to generate
            
        Returns:
            Tensor of shape (N, 2) with values in range [0, 1]
        """
        # Normalize the PDF to get a proper probability distribution
        pdf = self.data / self.data.sum()
        n = pdf.shape[0]
        
        # Calculate how many samples to allocate per cell based on the PDF
        cell_samples = (pdf.flatten() * num_samples).long()
        
        # Get indices of cells with samples
        valid_indices = torch.nonzero(cell_samples > 0).squeeze(1)
        
        # Calculate y and x indices for all valid cells
        y_indices = valid_indices // n
        x_indices = valid_indices % n
        
        # Get the number of samples for each valid cell
        valid_cell_samples = cell_samples[valid_indices]
        
        # Calculate offsets for creating the sample arrays
        sample_offsets = torch.zeros_like(valid_indices)
        sample_offsets[1:] = torch.cumsum(valid_cell_samples[:-1], dim=0)
        total_samples = valid_cell_samples.sum().item()
        
        # Generate random values for all samples at once
        rand_values = torch.rand((total_samples, 2), device=pdf.device)
        
        # Create indices for each sample to map to its cell
        # For each valid cell, repeat its index by the number of samples it should have
        cell_indices = torch.arange(valid_indices.size(0), device=pdf.device)
        repeated_indices = torch.repeat_interleave(cell_indices, valid_cell_samples)
        
        # Use the repeated indices to gather the x and y coordinates
        cell_x = torch.gather(x_indices, 0, repeated_indices)
        cell_y = torch.gather(y_indices, 0, repeated_indices)
        
        # Calculate the final sample positions
        all_samples = torch.zeros((total_samples, 2), device=pdf.device)
        all_samples[:, 0] = (rand_values[:, 0] + cell_x) / n
        all_samples[:, 1] = (rand_values[:, 1] + cell_y) / n
        
        # Shuffle to avoid correlation between adjacent samples
        perm = torch.randperm(total_samples, device=all_samples.device)
        all_samples = all_samples[perm]

        return all_samples
    
    def sample_segments(self, num_samples):
        """
        Sample segments centered at random points with random orientations.
        """
        # Get random points using the existing sample method
        centers = self.sample(num_samples)
        
        # Generate random angles and calculate segment endpoints
        n = self.data.shape[0]
        theta = 2 * torch.pi * torch.rand(centers.shape[0], device=centers.device)
        # half_length = 1.0 / n
        # half_length = 0.7071067811865476 / n
        half_length = 0.5 / n
        
        # Calculate offsets and create segments
        dx, dy = half_length * torch.cos(theta), half_length * torch.sin(theta)
        start_points = centers - torch.stack([dx, dy], dim=1)
        end_points = centers + torch.stack([dx, dy], dim=1)
        
        return torch.stack([start_points, end_points], dim=1)

    def update_data(self, samples, values):
        """
        Update the data structure based on new samples and their values.
        
        Args:
            samples: Tensor of shape (num_samples, 2) with values in range [0, 1]
            values: Tensor of shape (num_samples) with values to update the data structure with
        """
        # Convert samples to indices in the 2D grid
        n = self.data.shape[0]
        x_indices = (samples[:, 0] * n).long()
        y_indices = (samples[:, 1] * n).long()

        # Clip indices to valid range to prevent out-of-bounds access
        x_indices = torch.clamp(x_indices, 0, n-1)
        y_indices = torch.clamp(y_indices, 0, n-1)
        
        # Calculate absolute values of the input values
        abs_values = torch.abs(values)
        
        # Create flat indices for 2D grid to use with scatter_add_
        flat_indices = y_indices * n + x_indices
        
        # Create tensors to hold the updates and counts
        updates = torch.zeros_like(self.data).view(-1)
        counts = torch.zeros_like(self.data).view(-1)
        
        # Use scatter_add_ to accumulate values at the specific grid cells
        updates.scatter_add_(0, flat_indices, (1 - self.ema_alpha) * abs_values)
        counts.scatter_add_(0, flat_indices, torch.ones_like(abs_values))
        
        # Reshape back to 2D
        updates = updates.view(n, n)
        counts = counts.view(n, n)

        counts = counts.clamp(min=1)
        
        # Apply the EMA update rule only to cells that received samples
        # mask = counts > 0
        self.data = self.ema_alpha * self.data + updates / counts
        # self.data[mask] = self.ema_alpha * self.data[mask] + updates[mask] / counts[mask]
    


def edge_loss_slang(integrand, dim=2, GRID_SIZE=2000, plot_segments=False, fwd_grad=(False, -1), NUM_SUBDIVISION=20, LIPSCHITZ_BOUNDS=1e-6, DIV_EPS=1e-15, PLOT_RESOLUTION=1000, KDE_K=9, mode='direct', mode_aux_data=None, path_guiding_num_samples=0, path_guiding_ds=None, df_dx_mode='forward', mask_fn=None, custom_segments=None, custom_x=None):
    assert mode in ['direct', 'L2_img', 'L2_test_fn', 'L1_img', 'L1_test_fn']

    if type(LIPSCHITZ_BOUNDS) == float:
        LIPSCHITZ_BOUNDS = [LIPSCHITZ_BOUNDS] * NUM_SUBDIVISION
    if custom_x is None:
        # Given two version of the integrand (piecewise continuous and constant),
        # first detect discontinuities and then compute edge gradient
        if custom_segments is not None:
            assert GRID_SIZE is None
            segments = custom_segments
        else:
            segments = get_segments_on_grid(GRID_SIZE, dim=dim) # [N', 2, ID] ID is input dimension
        if path_guiding_num_samples > 0:
            segments2 = path_guiding_ds.sample_segments(path_guiding_num_samples)
            segments = torch.cat([segments, segments2], dim=0)

        integrand_pw_constant = lambda x: integrand(x, ret_const=True)
        integrand_pw_continuous = lambda x: integrand(x, ret_const=False)

        # Snap segments to discontinuities
        segments = snap_segment_to_discontinuity(segments, integrand_pw_constant, LIPSCHITZ_BOUNDS, NUM_SUBDIVISION) # [N', 2, ID]
        x = segments.mean(axis=1) # [N, ID]
    else:
        x = custom_x

    # Filter out points that are outside the domain
    x_in_domain = torch.all(x > 0, dim=1) & torch.all(x < 1, dim=1) # [N']
    x = x[x_in_domain] # [N, ID]

    # Set a maximum number of x to support
    if x.shape[0] > 10**7:
        # Randomly select 10^7 points if there are too many
        indices = torch.randperm(x.shape[0], device=x.device)[:10**7]
        x = x[indices]
        x_in_domain = x_in_domain[indices]
        print(f"WARNING: AFTER SNAPPING Reduced number of points from {x.shape[0]} to {x.shape[0]}")

    x_in_domain = torch.ones_like(x[:,0])

    x.requires_grad = True
    N = len(x)

    # Find which discontinuity each segment crosses (map N to len(f_list))
    f, f_min_idx = integrand(x, ret_impl=True)

    # df_dp = torch.autograd.grad(f.sum(), integrand.p, create_graph=False, retain_graph=True)[0].detach()  # [N, 3]
    # print(df_dp)
    # sdfs

    # if the boundary is not associated with an if statement, ignore it
    f_min_mask = f_min_idx >= 0
    # x = x[f_min_mask]
    # f_min_idx = f_min_idx[f_min_mask]
    # f = f[f_min_mask]
    # x_in_domain = x_in_domain[f_min_mask]
    x_in_domain = torch.logical_or(x_in_domain, f_min_mask) # this is our active mask

    if plot_segments:
        points_ = points_on_grid(PLOT_RESOLUTION, dim=dim) # [N, ID]

        values = integrand(points_).detach()
        if len(values.shape) > 1 and values.shape[1] == 3:
            values = values.reshape(PLOT_RESOLUTION, PLOT_RESOLUTION, 3)
        else:
            values = values.reshape(PLOT_RESOLUTION, PLOT_RESOLUTION)

        if dim == 2:
            # values = integrand_pw_constant(points_).reshape(PLOT_RESOLUTION, PLOT_RESOLUTION)
            # # Find all unique values and assign sequential indices from 0
            # unique_values = torch.unique(values)
            # # Create a mapping tensor where each position corresponds to a unique value's index
            # mapping = torch.arange(len(unique_values), device=values.device)
            # # Use searchsorted to find indices for each value in the original tensor
            # indices = torch.searchsorted(unique_values, values.flatten()).reshape(values.shape)
            # values = indices

            # print(torch.unique(f_min_idx))

            # values_unique = torch.unique(values)
            # print(f_min_idx.unique())
            x_ = x[f_min_idx >= 0]
            f_ = f[f_min_idx >= 0]
            # df_dx = torch.autograd.grad(f_.sum(), x_, create_graph=False, retain_graph=True, allow_unused=True)
            # print(df_dx.std(), df_dx.max(), df_dx.min())
            f_min_idx_ = f_min_idx[f_min_idx >= 0]

            # print(f_min_idx_.unique())

            plt.figure(figsize=(8,8))
            plt.imshow(values.cpu(), origin='lower', extent=[0,1,0,1])
            plt.colorbar()
    

            plt.scatter(x_[:, 0].detach().cpu(), x_[:, 1].detach().cpu(), c=f_min_idx_.detach().cpu(), cmap='tab10', s=0.5)
            # plt.scatter(x_[:, 0].detach().cpu(), x_[:, 1].detach().cpu(), c='blue', s=0.1)
            # plt.colorbar(label='Discontinuity Index')
           # plt.scatter(x[:, 0].cpu(), x[:, 1].cpu(), c='red', s=0.5)
            plt.show()
        else:
            print("TODO: Plotting 3D integrand")
    
    # with torch.autograd.set_detect_anomaly(True):

    # # Compute the weight of the samples using KDE for normalization
    with torch.no_grad():
        # kde = KDETorchKNN(uniform_KDE=True)
        # kde = KDETorchKNN(uniform_KDE=False)
        # for KDEKeOpsKNN, need min f_min_idx to be 1
        f_min_idx = (x_in_domain.type(f_min_idx.dtype) * (f_min_idx + 1))
        kde = KDEKeOpsKNN(do_KDE=True)
        p = kde(x, f_min_idx, K=KDE_K) # [N]
        # if you use KDEKeOpsKNN, need to subtract 1 from f_min_idx
        f_min_idx -= 1
        # print(p.std(), p.max(), p.min())
    
    # print(f.std(), f.max(), f.min())
    # print(x.shape)

    if df_dx_mode == 'forward':
        ####### CAUTION: CANNOT USE THIS FOR SHADERS WHERE P IS COMPUTED ON DEMAND INSTEAD OF A PROPERTY #######
        # Use forward AD to compute df_dx for each dimension
        # This works for both 2D and 3D
        dim = x.shape[1]  # Get the dimensionality from input
        df_dx_components = []
        
        for i in range(dim):
            dx = torch.zeros_like(x)
            dx[:, i] = 1.0
            df_dxi = integrand.forward_grad(x, dx, torch.zeros_like(p), ret_impl=True)[1]
            df_dx_components.append(df_dxi)
        
        # Stack the gradients along dimension 1
        df_dx = torch.stack(df_dx_components, dim=1)
    else:
        # Forward mode AD not working for circles or for sphere raymarching
        # BE VERY VERY CAREFUL WITH TURNING THIS ON, FAILS INSIDIOUSLY WITH WRONG GRADIENTS (raymarching_opt.py)
        df_dx = torch.autograd.grad(f.sum(), x, create_graph=False, retain_graph=True)[0].detach()  # [N, 3]
    # print(df_dx)
    # print(df_dx.std(), df_dx.max(), df_dx.min())
    
    # Normalize the gradients
    df_dx_norm = torch.norm(df_dx, p=2, dim=1)  # [N]
    with torch.no_grad():
        df_dx_norm = torch.clamp(df_dx_norm, min=1e-1, max=1e4)

    # Compute the change in output across the discontinuity
    with torch.no_grad():
        out_true = integrand(x, impl_idx=f_min_idx, force_sign=0)
        out_false = integrand(x, impl_idx=f_min_idx, force_sign=1)
        delta_out = (out_true - out_false).detach()  # [N, OD] where OD is output dimension

    p = p.unsqueeze(-1); f = f.unsqueeze(-1); df_dx_norm = df_dx_norm.unsqueeze(-1)
    if len(delta_out.shape) == 1:
        delta_out = delta_out.unsqueeze(-1)
    
    if mask_fn is not None:
        mask = mask_fn(x)
        p *= mask
    
    if fwd_grad[0]:
        dp = torch.zeros_like(p)
        if type(fwd_grad[1]) == int:
            dp[fwd_grad[1]] = 1.0
        else:
            dp = fwd_grad[1]
        dx = torch.zeros_like(x)
        df_dp0 = integrand.forward_grad(x, dx, dp, ret_impl=True)[1]
        df_dp0 = df_dp0.unsqueeze(-1)
        # print(p.std(), df_dp0.std(), delta_out.std(), df_dx_norm.std())
        # print(p.max(), df_dp0.max(), delta_out.max(), df_dx_norm.max())
        # print(p.min(), df_dp0.min(), delta_out.min(), df_dx_norm.min())

        dout_dp0 = (p * delta_out * df_dp0 / (df_dx_norm + DIV_EPS))
        # print(dout_dp0.std(), dout_dp0.max(), dout_dp0.min())
        # print(df_dx_norm.std(), df_dx_norm.max(), df_dx_norm.min())
        return dout_dp0, x
    else:
        if mode == 'L2_img' or mode == 'L1_img':
            gt_img = mode_aux_data
            err_size = gt_img.shape[0]
            # Scale x coordinates from [0,1] to indices in the L2_ERR grid
            # Clamp to ensure we stay within bounds
            x_indices = (((x[:, 1])) * err_size).long().clamp(0, err_size - 1)
            y_indices = (((x[:, 0])) * err_size).long().clamp(0, err_size - 1)
            gt_vals = gt_img[x_indices, y_indices]
            
            ######## Two sample Monte Carlo Estimator  (one integral for entire domain); no pixel antialiasing (or pixel color computation) needed #########
            if mode == 'L1_img':
                delta_out = ((out_true - gt_vals).abs() - (out_false - gt_vals).abs()).detach()
            else:
                delta_out = ((out_true - gt_vals)**2 - (out_false - gt_vals)**2).detach()

            ######## Full pixel antialiasing (standard chain rule with one integral per pixel) #########
            # delta_out = (out_true - out_false).detach() * point_errors
        elif mode == 'L2_test_fn' or mode == 'L1_test_fn':
            gt_vals = mode_aux_data(x)
            ### For infinite res supervision, use infinite res L2 loss ###
            if mode == "L2_test_fn":
                delta_out = ((out_true - gt_vals)**2 - (out_false - gt_vals)**2).detach()
            else:
                delta_out = ((out_true - gt_vals).abs() - (out_false - gt_vals).abs()).detach()
        
        if len(delta_out.shape) == 1:
            delta_out = delta_out.unsqueeze(-1)
        
        # Update the path guiding data structure
        if path_guiding_num_samples > 0:
            path_guiding_ds.update_data(x, (p * delta_out).abs().sum(dim=1))

        # Check for NaN or Inf values in key variables
        if torch.isnan(p).any() or torch.isinf(p).any():
            print("Warning: NaN or Inf detected in p")
        
        if torch.isnan(delta_out).any() or torch.isinf(delta_out).any():
            print("Warning: NaN or Inf detected in delta_out")
            
        if torch.isnan(f).any() or torch.isinf(f).any():
            print("Warning: NaN or Inf detected in f")
            
        if torch.isnan(df_dx_norm).any() or torch.isinf(df_dx_norm).any():
            print("Warning: NaN or Inf detected in df_dx_norm")
        # print()
        # # Print min and max values of key tensors
        # print(f"p min: {p.min().item():.6e}, p max: {p.max().item():.6e}")
        # print(f"delta_out min: {delta_out.min().item():.6e}, delta_out max: {delta_out.max().item():.6e}")
        # print(f"f min: {f.min().item():.6e}, f max: {f.max().item():.6e}")
        # print(f"df_dx_norm min: {df_dx_norm.min().item():.6e}, df_dx_norm max: {df_dx_norm.max().item():.6e}")
        # print(f"num samples = {x.shape[0]}")
        # print(f.requires_grad)
        # dfdp = torch.autograd.grad(f.sum(), integrand.p, create_graph=False, retain_graph=True)[0].detach()  # [N, 3]
        # print(dfdp)
        # sdfs

        # TODO: there is some insanely weird bug in slang which makes the checkerboard derivative weird, need to hack around it for now
        if integrand.__class__.__name__ == "CheckerboardIntegrandSlang":
            return (p * delta_out * integrand.p[0] / (df_dx_norm + DIV_EPS)).sum()

        ##### For all three modes, compute the final (undifferentiated) integrand #####
        out = (p * delta_out * f / (df_dx_norm + DIV_EPS)).sum()
        return out


@dataclass
class BoundaryLossConfig:
    dim: int = 2
    grid_size: int = 2000
    plot_segments: bool = False
    fwd_grad: tuple = (False, -1)
    num_subdivision: int = 20
    lipschitz_bounds: float = 1e-6
    div_eps: float = 1e-15
    plot_resolution: int = 1000
    kde_k: int = 9
    mode: str = "direct"
    mode_aux_data: any = None
    path_guiding_num_samples: int = 0
    path_guiding_ds: any = None
    df_dx_mode: str = "forward"
    mask_fn: any = None
    custom_segments: any = None
    custom_x: any = None


def boundary_loss(integrand, cfg: BoundaryLossConfig):
    """Wrapper around edge_loss_slang using a config object."""
    return edge_loss_slang(
        integrand,
        dim=cfg.dim,
        GRID_SIZE=cfg.grid_size,
        plot_segments=cfg.plot_segments,
        fwd_grad=cfg.fwd_grad,
        NUM_SUBDIVISION=cfg.num_subdivision,
        LIPSCHITZ_BOUNDS=cfg.lipschitz_bounds,
        DIV_EPS=cfg.div_eps,
        PLOT_RESOLUTION=cfg.plot_resolution,
        KDE_K=cfg.kde_k,
        mode=cfg.mode,
        mode_aux_data=cfg.mode_aux_data,
        path_guiding_num_samples=cfg.path_guiding_num_samples,
        path_guiding_ds=cfg.path_guiding_ds,
        df_dx_mode=cfg.df_dx_mode,
        mask_fn=cfg.mask_fn,
        custom_segments=cfg.custom_segments,
        custom_x=cfg.custom_x,
    )


def boundary_loss_slang(integrand, cfg: BoundaryLossConfig):
    """Alias for boundary_loss."""
    return boundary_loss(integrand, cfg)



def efficient_finite_diff_grad(integrand, param_idx, batch_size=1024*128, fd_grid_size=256, fd_aa_size=40, fd_eps=1e-3, out_dim=3):
    """
    Compute finite difference gradient for a specific parameter index with batched processing.
    
    Args:
        integrand: The integrand function to evaluate
        param_idx: Index of the parameter to compute gradient for
        batch_size: Number of points to process in each batch
        fd_grid_size: Base grid size for finite differencing
        fd_aa_size: Anti-aliasing size (points per grid cell)
        fd_eps: Epsilon for finite difference
        out_dim: Output dimension of the integrand
        
    Returns:
        Tensor containing the gradient image
    """
    # Total number of points in each dimension
    total_size = fd_grid_size * fd_aa_size
    
    # Create coordinate arrays
    x_coords = torch.linspace(0, 1, total_size+1, device=DEVICE)[:-1] + 1 / (2 * total_size)
    y_coords = torch.linspace(0, 1, total_size+1, device=DEVICE)[:-1] + 1 / (2 * total_size)

    # Initialize separate result tensors for each output dimension
    results = [torch.zeros(fd_grid_size, fd_grid_size, device=DEVICE) for _ in range(out_dim)]
    
    # Get the parameter to differentiate
    param = list(integrand.parameters())[0]  # Assuming only one parameter tensor
    
    # Store original parameter value
    orig_value = param.data[param_idx].clone().detach()
    
    # Calculate total number of points and batches
    total_points = total_size * total_size
    num_batches = (total_points + batch_size - 1) // batch_size

    with torch.no_grad():
        # Process in batches
        for batch_idx in range(num_batches):
            # Determine batch indices
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, total_points)
            
            # Convert linear indices to 2D coordinates
            linear_indices = torch.arange(start_idx, end_idx, device=DEVICE)
            y_indices = linear_indices // total_size
            x_indices = linear_indices % total_size
            
            # Get actual coordinate values
            x_batch = x_coords[x_indices]
            y_batch = y_coords[y_indices]
            points_batch = torch.stack([x_batch, y_batch], dim=1)
            
            # Compute f(p + eps)
            param.data[param_idx] = orig_value + fd_eps
            int_plus = integrand(points_batch)
            
            # Compute f(p - eps)
            param.data[param_idx] = orig_value - fd_eps
            int_minus = integrand(points_batch)
            
            # Compute finite difference
            fd_values = (int_plus - int_minus) / (2 * fd_eps)

            # Map back to grid cells for anti-aliasing
            grid_x = x_indices // fd_aa_size
            grid_y = y_indices // fd_aa_size
            
            # Use scatter_add to accumulate values for anti-aliasing
            flat_indices = grid_y * fd_grid_size + grid_x
            if out_dim == 1:
                results[0].view(-1).scatter_reduce_(0, flat_indices, fd_values, reduce='mean')
            else:
                # Process each dimension separately
                for dim in range(out_dim):
                    # Get values for this dimension
                    dim_values = fd_values[:, dim]
                    # Use scatter_reduce to compute mean for this dimension
                    results[dim].view(-1).scatter_reduce_(0, flat_indices, dim_values, reduce='mean')
    
    # Restore original parameter value
    param.data[param_idx] = orig_value
    
    # Scale by pixel area to maintain proper integration
    pixel_area = 1.0 / (fd_grid_size * fd_grid_size)
    for dim in range(out_dim):
        results[dim] *= pixel_area
    
    # Stack results if multi-dimensional
    if out_dim > 1:
        result = torch.stack(results, dim=-1)
    else:
        result = results[0].unsqueeze(-1)
    
    return result

def plot_fwd_grad_ours_and_fd(dout_dp_ours, x_ours, fd_grad_img, vmax=0.01,grid_size=101, save_path=None, plot_error=False, show_plot=True):
    grid_bins = torch.linspace(0, 1, grid_size+1)
    indices = torch.stack([
        torch.clamp(torch.searchsorted(grid_bins, x_ours[:,1])-1, 0, grid_size-1),
        torch.clamp(torch.searchsorted(grid_bins, x_ours[:,0])-1, 0, grid_size-1)
    ], dim=1)
    grid_values = torch.zeros(grid_size * grid_size)
    # Flatten indices before scatter add
    flat_indices = indices[:,0] * grid_size + indices[:,1]
    grid_values.scatter_add_(0, flat_indices, dout_dp_ours)
    grid_values = grid_values.reshape(grid_size, grid_size)
    # print(grid_values)
    # sdfs

    # vmax = 1e-7
    # vmax = 1e-5
    # vmax = fd_grad_img.abs().max() / 4
    vmax = fd_grad_img.abs().max()

    # print(grid_values.shape, fd_grad_img.shape)


    if plot_error:
        plt.figure(figsize=(24,8))
        plt.subplot(1,3,1)
        plt.imshow(grid_values.detach().cpu(), origin='lower', extent=[0,1,0,1], cmap='bwr_r', vmin=-vmax, vmax=vmax)
        plt.colorbar(label='dout_dp')
        plt.title(f'Ours: {dout_dp_ours.sum()}')
        plt.subplot(1,3,2)
        plt.imshow(fd_grad_img.detach().cpu(), origin='lower', extent=[0,1,0,1], cmap='bwr_r', vmin=-vmax, vmax=vmax)
        plt.colorbar(label='dout_dp')
        plt.title(f'FD: {fd_grad_img.sum()}')
        plt.subplot(1,3,3)
        plt.title('Absolute Error (x10)')
        err = (grid_values - fd_grad_img).abs()
        plt.imshow(err.detach().cpu(), origin='lower', extent=[0,1,0,1], cmap='viridis', vmax=vmax)
        plt.colorbar(label='|error| x 10')
        plt.title('Absolute Error (x10)')
    
    else:

        plt.figure(figsize=(16,8))
        plt.subplot(1,2,1)
        plt.imshow(grid_values.detach().cpu(), origin='lower', extent=[0,1,0,1], cmap='bwr_r', vmin=-vmax, vmax=vmax)
        # plt.colorbar(label='dout_dp')
        plt.title(f'Ours: {dout_dp_ours.sum()}')
        plt.axis('off')
        plt.subplot(1,2,2)
        plt.imshow(fd_grad_img.detach().cpu(), origin='lower', extent=[0,1,0,1], cmap='bwr_r', vmin=-vmax, vmax=vmax)
        # plt.colorbar(label='dout_dp')
        plt.title(f'FD: {fd_grad_img.sum()}')
        plt.axis('off')
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300)
    if show_plot:
        plt.show()
    plt.close()


def compute_and_plot_fwd_grad(integrand, p_idx, plot_error=False, FD_GRID_SIZE=256, FD_AA_SIZE=16, FD_EPS=1e-3, OUR_GRID_SIZE=5000, integrand_class_name=None, show_plot=True):
    fd_grad_img = efficient_finite_diff_grad(integrand, param_idx=p_idx, fd_grid_size=FD_GRID_SIZE, fd_aa_size=FD_AA_SIZE, fd_eps=FD_EPS, out_dim=integrand.out_dim)

    # our_grad_edge, x_edge = edge_loss_slang(integrand, GRID_SIZE=OUR_GRID_SIZE, plot_segments=True, fwd_grad=(True, p_idx), df_dx_mode='forward')
    our_grad_edge, x_edge = edge_loss_slang(integrand, GRID_SIZE=OUR_GRID_SIZE, plot_segments=False, fwd_grad=(True, p_idx), df_dx_mode='forward')
    vmax=None

    # print(our_grad_edge.min(), our_grad_edge.max(), our_grad_edge.std())

    x_area_samples = points_on_grid(OUR_GRID_SIZE, jitter=True)
    dp = torch.zeros_like(integrand.p)
    dp[p_idx] = 1.0
    dx = torch.zeros_like(x_area_samples)
    our_grad_area = integrand.forward_grad(x_area_samples, dx, dp, ret_impl=False)[0]
    our_grad_area *= 1/OUR_GRID_SIZE**2

    # print(our_grad_area.shape, our_grad_edge.abs().max())

    # print(our_grad_area.shape, our_grad_edge.shape)
    if len(our_grad_edge) == 0:
        our_grad = our_grad_area
        x = x_area_samples
    else:
        if len(our_grad_edge.shape) == 1:
            our_grad_edge = our_grad_edge.unsqueeze(-1)
        if len(our_grad_area.shape) == 1:
            our_grad_area = our_grad_area.unsqueeze(-1)
        our_grad = torch.cat([our_grad_edge, our_grad_area], dim=0)
        x = torch.cat([x_edge, x_area_samples], dim=0)


    print(f"p_idx: {p_idx}, ours: {our_grad.sum()}, fd: {fd_grad_img.sum()}")

    fd_grad_img = fd_grad_img.sum(dim=-1)
    our_grad = our_grad.sum(dim=-1)

    plot_fwd_grad_ours_and_fd(our_grad, x, fd_grad_img, vmax=vmax, grid_size=FD_GRID_SIZE, save_path=f"results/{integrand_class_name}/{p_idx}.pdf", plot_error=plot_error, show_plot=show_plot)

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
    
