import torch
import matplotlib.pyplot as plt

from .device import DEVICE
from .segments import points_on_grid
from .boundary import BoundaryLossConfig, boundary_loss
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

    vmax = fd_grad_img.abs().max()

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

    cfg = BoundaryLossConfig(
        grid_size=OUR_GRID_SIZE,
        plot_segments=False,
        fwd_grad=(True, p_idx),
        df_dx_mode='forward',
    )
    our_grad_edge, x_edge = boundary_loss(integrand, cfg)
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
