import torch
import torch.nn as nn
import slangtorch
from helpers import *

class BaseIntegrandSlang(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x, impl_idx=None, force_sign=-1, ret_const=False, ret_impl=False):
        r, impl_fn, impl_idx, out_idx = SlangShader.apply(x, self.p, impl_idx, force_sign, self.shader, ret_const, ret_impl)
        if ret_const:
            return out_idx
        elif ret_impl:
            return impl_fn, impl_idx
        return r

    def forward_grad(self, x, d_x, d_p, impl_idx=None, force_sign=-1, ret_const=False, ret_impl=False):
        d_r, d_impl_fn = SlangShaderForwardGrad(x, d_x, self.p, d_p, impl_idx, force_sign, self.shader, ret_const, ret_impl)
        return d_r, d_impl_fn

class BaseIntegrandSlangRGB(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x, impl_idx=None, force_sign=-1, ret_const=False, ret_impl=False):
        r, impl_fn, impl_idx, out_idx = SlangShaderRGB.apply(x, self.p, impl_idx, force_sign, self.shader, ret_const, ret_impl)
        if ret_const:
            return out_idx
        elif ret_impl:
            return impl_fn, impl_idx
        return r

    def forward_grad(self, x, d_x, d_p, impl_idx=None, force_sign=-1, ret_const=False, ret_impl=False):
        d_r, d_impl_fn = SlangShaderForwardGradRGB(x, d_x, self.p, d_p, impl_idx, force_sign, self.shader, ret_const, ret_impl)
        return d_r, d_impl_fn


class LineIntegrandSlang(BaseIntegrandSlang):
    def __init__(self):
        super().__init__()
        self.shader = slangtorch.loadModule("slang/line.slang")
        self.p = nn.Parameter(torch.tensor([0.5, -0.5, 0.1, 1.0, 1.0]))

class NLinesIntegrandSlang(BaseIntegrandSlang):
    def __init__(self):
        super().__init__()
        self.shader = slangtorch.loadModule("slang/n_lines.slang")
        self.p = nn.Parameter(torch.tensor([0.5, -0.5, 0.1, -1.0, 1.0, 0.3, -0.8, 0.5, 1.0, 1.0]))
        # self.p = nn.Parameter(torch.tensor([0.5, -0.5, 0.1, 1.0, 1.0, 0.5, -0.5, 0.3, 1.0, 1.0]))

class NLinesGeneratedIntegrandSlang(BaseIntegrandSlang):
    def __init__(self):
        super().__init__()
        self.shader = slangtorch.loadModule("slang/n_lines_generated.slang")
        # self.p = nn.Parameter(torch.tensor([0.5, -0.5, 0.1, 1.0, 1.0, 0.3, -0.8, 0.5, 4.0, 4.0]))
        # self.p = nn.Parameter(torch.tensor([0.5, -0.5, 0.1, 1.0, 1.0, 0.5, -0.5, 0.3, 1.0, 1.0]))
        self.p = nn.Parameter(torch.tensor([0.5, -0.5, 0.1, 1.0, 1.0, 0.3, -0.8, 0.5, 4.0, 4.0]))
        # Initialize 20 lines that intersect [0,1]^2
        # Each line has 5 parameters: [a, b, c, r0, r1]
        # ax + by + c = 0 defines the line
        # r0 is random between -1 and 1, r1 kept at 1.0
        # self.p = nn.Parameter(torch.tensor([
        #     # Horizontal lines
        #     0.0, 1.0, -0.2, -0.73, 1.0,    # y = 0.2
        #     0.0, 1.0, -0.4, 0.85, 1.0,     # y = 0.4
        #     0.0, 1.0, -0.6, -0.32, 1.0,    # y = 0.6
        #     0.0, 1.0, -0.8, 0.91, 1.0,     # y = 0.8
            
        #     # Vertical lines  
        #     1.0, 0.0, -0.2, -0.45, 1.0,    # x = 0.2
        #     1.0, 0.0, -0.4, 0.67, 1.0,     # x = 0.4
        #     1.0, 0.0, -0.6, -0.88, 1.0,    # x = 0.6
        #     1.0, 0.0, -0.8, 0.23, 1.0,     # x = 0.8
            
        #     # Diagonal lines (positive slope)
        #     1.0, 1.0, -0.5, -0.51, 1.0,    # x + y = 0.5
        #     1.0, 1.0, -1.0, 0.78, 1.0,     # x + y = 1.0
        #     1.0, 1.0, -1.5, -0.94, 1.0,    # x + y = 1.5
            
        #     # Diagonal lines (negative slope)
        #     1.0, -1.0, -0.2, 0.44, 1.0,    # x - y = 0.2
        #     1.0, -1.0, 0.0, -0.63, 1.0,    # x - y = 0
        #     1.0, -1.0, 0.2, 0.82, 1.0,     # x - y = -0.2
            
        #     # Additional angled lines
        #     0.5, 1.0, -0.3, -0.29, 1.0,    # 0.5x + y = 0.3
        #     0.5, 1.0, -0.7, 0.55, 1.0,     # 0.5x + y = 0.7
        #     1.0, 0.5, -0.4, -0.71, 1.0,    # x + 0.5y = 0.4
        #     1.0, 0.5, -0.8, 0.39, 1.0,     # x + 0.5y = 0.8
        #     0.7, 1.0, -0.5, -0.83, 1.0,    # 0.7x + y = 0.5
        #     1.0, 0.7, -0.6, 0.61, 1.0      # x + 0.7y = 0.6
        # ]))


class PerlinIntegrandSlang(BaseIntegrandSlang):
    def __init__(self):
        super().__init__()
        self.shader = slangtorch.loadModule("slang/perlin.slang")
        self.p = nn.Parameter(torch.tensor([0.2]))

class PerlinGeneratedIntegrandSlang(BaseIntegrandSlang):
    def __init__(self):
        super().__init__()
        self.shader = slangtorch.loadModule("slang/perlin_generated.slang")
        self.p = nn.Parameter(torch.tensor([0.2]))

class FractalPerlinIntegrandSlang(BaseIntegrandSlang):
    def __init__(self):
        super().__init__()
        self.shader = slangtorch.loadModule("slang/fractal_perlin.slang")
        # Parameters: [threshold, freq_scale, persistence, lacunarity]
        self.p = nn.Parameter(torch.tensor([0.2, 8.0, 0.5, 2.0]))

class MandelbrotIntegrandSlang(BaseIntegrandSlang):
    def __init__(self):
        super().__init__()
        self.shader = slangtorch.loadModule("slang/mandelbrot.slang")
        # Parameter: threshold value for the set boundary
        self.p = nn.Parameter(torch.tensor([0.5]))

class TriangleIntegrandSlang(BaseIntegrandSlang):
    def __init__(self):
        super().__init__()
        self.shader = slangtorch.loadModule("slang/triangle.slang")
        # Parameters: [v0x, v0y, v1x, v1y, v2x, v2y]
        self.p = nn.Parameter(torch.tensor([
            0.2, 0.2,  # v0
            0.8, 0.2,  # v1
            0.5, 0.7   # v2
        ]))

class TriangleGeneratedIntegrandSlang(BaseIntegrandSlang):
    def __init__(self):
        super().__init__()
        self.shader = slangtorch.loadModule("slang/triangle_generated.slang")
        # Parameters: [v0x, v0y, v1x, v1y, v2x, v2y]
        self.p = nn.Parameter(torch.tensor([
            0.2, 0.2,  # v0
            0.8, 0.2,  # v1
            0.5, 0.7   # v2
        ]))

class NTrianglesIntegrandSlang(BaseIntegrandSlang):
    def __init__(self):
        super().__init__()
        self.shader = slangtorch.loadModule("slang/n_triangles.slang")
        # Parameters: [v0x, v0y, v1x, v1y, v2x, v2y, ...] for each triangle
        # self.p = nn.Parameter(torch.tensor([
        #     # First triangle
        #     0.2, 0.2,  # v0
        #     0.5, 0.2,  # v1
        #     0.3, 0.5,  # v2
        #     # Second triangle
        #     0.6, 0.5,  # v0
        #     0.8, 0.3,  # v1
        #     0.7, 0.7   # v2
        # ]))
        self.p = nn.Parameter(torch.tensor([
            # First triangle
            0.2, 0.2,  # v0
            0.7, 0.5,  # v1
            0.3, 0.5,  # v2
            # Second triangle
            0.6, 0.5,  # v0
            0.8, 0.3,  # v1
            0.7, 0.7   # v2
        ]))

        

class NTrianglesGeneratedIntegrandSlang(BaseIntegrandSlang):
    def __init__(self):
        super().__init__()
        self.shader = slangtorch.loadModule("slang/n_triangles_generated.slang")
        # self.shader = slangtorch.loadModule("slang_gen/n_triangles.slang")
        # Parameters: [v0x, v0y, v1x, v1y, v2x, v2y, ...] for each triangle
        # self.p = nn.Parameter(torch.tensor([
        #     # First triangle
        #     0.2, 0.2,  # v0
        #     0.7, 0.5,  # v1
        #     0.3, 0.5,  # v2
        #     # Second triangle
        #     0.6, 0.5,  # v0
        #     0.8, 0.3,  # v1
        #     0.7, 0.7   # v2
        # ]))

        self.p = nn.Parameter(torch.tensor([
            # First triangle (from previous example)
            0.2, 0.2,  # v0
            0.7, 0.5,  # v1
            0.3, 0.5,  # v2
            # Second triangle (from previous example) 
            0.6, 0.5,  # v0
            0.8, 0.3,  # v1
            0.7, 0.7,  # v2
            # Additional triangles
            0.1, 0.6,  # v0
            0.3, 0.8,  # v1
            0.2, 0.9,  # v2
            0.4, 0.3,  # v0
            0.5, 0.4,  # v1
            0.3, 0.4,  # v2
            0.8, 0.8,  # v0
            0.9, 0.6,  # v1
            0.7, 0.9,  # v2
            0.1, 0.1,  # v0
            0.3, 0.1,  # v1
            0.2, 0.3,  # v2
            0.8, 0.1,  # v0
            0.9, 0.2,  # v1
            0.9, 0.1,  # v2
            0.5, 0.8,  # v0
            0.6, 0.9,  # v1
            0.4, 0.9,  # v2
            0.4, 0.6,  # v0
            0.6, 0.7,  # v1
            0.5, 0.6,  # v2
            0.7, 0.4,  # v0
            0.9, 0.4,  # v1
            0.8, 0.5   # v2
        ]))

class Test0IntegrandSlang(BaseIntegrandSlang):
    def __init__(self):
        super().__init__()
        self.shader = slangtorch.loadModule("slang_gen/transformer_test0.slang")
        self.p = nn.Parameter(torch.tensor([
            0.2
        ]))

class Test1IntegrandSlang(BaseIntegrandSlang):
    def __init__(self):
        super().__init__()
        self.shader = slangtorch.loadModule("slang_gen/transformer_test1.slang")
        self.p = nn.Parameter(torch.tensor([
            0.2, 0.4, 0.6, 0.8,
            0.25, 0.45, 0.65, 0.85,
        ]))

class BugReportSlang(BaseIntegrandSlang):
    def __init__(self):
        super().__init__()
        self.shader = slangtorch.loadModule("slang_gen/bug_report.slang")
        self.p = nn.Parameter(torch.tensor([
            0.2, 0.4, 0.6, 0.8,
            0.25, 0.45, 0.65, 0.85,
        ]))

class CirclesIntegrandSlang(BaseIntegrandSlang):
    def __init__(self):
        super().__init__()
        self.shader = slangtorch.loadModule("slang_gen/circles.slang")
        self.p = nn.Parameter(torch.tensor([
            0.2, 0.4, 0.3, 0.4,
            0.5, 0.7, 0.2, 0.2,
            0.6, 0.6, 0.3, 0.1,
            0.8, 0.3, 0.2, 0.2,
            0.3, 0.8, 0.15, 0.1,
            0.7, 0.7, 0.18, 0.2,
            0.4, 0.2, 0.12, 0.2,
            0.9, 0.7, 0.16, 0.2,
            0.15, 0.6, 0.19, 0.7,
            0.5, 0.5, 2.0, 1.0,
        ]))
        self.out_dim = 1

class WavyCircleIntegrandSlang(BaseIntegrandSlang):
    def __init__(self):
        super().__init__()
        self.shader = slangtorch.loadModule("slang_gen/wavy_circle.slang")
        self.p = nn.Parameter(torch.tensor([
            0.5, 0.5, 0.3, 1.0, 0.1, 10.0
        ]))
        self.out_dim = 1

class DoubleLineIntegrandSlang(BaseIntegrandSlang):
    def __init__(self, LINE_SPACING=0.1, CURVE_AMPLITUDE=0.1, FREQ=10.0, A1=0.1, A2=0.1, A3=0.1):
        super().__init__()
        self.shader = slangtorch.loadModule("slang_gen/double_line_integrand.slang")
        self.p = nn.Parameter(torch.tensor([
            0.5,
            LINE_SPACING,
            CURVE_AMPLITUDE,
            FREQ,
            A1,
            A2,
            A3
        ]))
        self.out_dim = 1

integrand = DoubleLineIntegrandSlang()

# class FractalTreeIntegrandSlang(BaseIntegrandSlang):
#     def __init__(self):
#         super().__init__()
#         self.shader = slangtorch.loadModule("slang_gen/fractal_tree.slang")
#         self.p = nn.Parameter(torch.tensor([
#             0.2, 0.4, 0.3, 0.4,
#             8.0, 0.4, 3.0, 1.0,
#         ]))
#         self.out_dim = 3


class CapsuleIntegrandSlang(BaseIntegrandSlang):
    def __init__(self):
        super().__init__()
        self.shader = slangtorch.loadModule("slang_gen/capsule.slang")
        self.p = nn.Parameter(torch.tensor([
            0.5, 0.2, 0.3, 0.7, 0.1, 0.2
        ]))

class HeartIntegrandSlang(BaseIntegrandSlang):
    def __init__(self):
        super().__init__()
        import math
        self.shader = slangtorch.loadModule("slang_gen/heart.slang")
        self.p = nn.Parameter(torch.tensor([
            0.25, 0.75, 0.0, 1.0, 0.5, 0.5, math.sqrt(2.0)/4.0
        ]))


class VoronoiSimpleIntegrandSlang(BaseIntegrandSlangRGB):
    def __init__(self, N=10, seed=None, jitter_scale=0.8, is_accel=False):
        super().__init__()
        self.is_accel = is_accel
        if self.is_accel:
            self.shader = slangtorch.loadModule("slang_gen/voronoi_simple_accel.slang")
        else:
            self.shader = slangtorch.loadModule("slang_gen/voronoi_simple.slang")
            # self.shader = slangtorch.loadModule("slang_gen/voronoi_simple_warp.slang")
        self.grid_size = N
        
        # Initialize with metadata
        if self.is_accel:
            # params_list = [float(N)]
            params_list = []
        else:
            params_list = [float(N)]
        
        # Initialize random number generator
        torch.manual_seed(12)
        
        # For each cell in the grid
        for i in range(N):  # i = x index
            for j in range(N):  # j = y index
                # Calculate cell center
                cell_center_x = (i + 0.5) / N
                cell_center_y = (j + 0.5) / N
                
                # Add more significant jitter for more randomness
                # Higher jitter_scale = more random appearance
                x = cell_center_x + (torch.rand(1).item() - 0.5) * jitter_scale / N
                y = cell_center_y + (torch.rand(1).item() - 0.5) * jitter_scale / N
                
                # Ensure point stays within the grid cell (important for acceleration structure)
                x = max(i/N, min((i+1)/N - 1e-6, x))
                y = max(j/N, min((j+1)/N - 1e-6, y))
                
                # Generate random color with more variation
                r = torch.rand(1).item()
                g = torch.rand(1).item()
                b = torch.rand(1).item()
                
                # Add to parameters list
                params_list.extend([x, y, r, g, b])
                
                # Debug print
                # print(f"Cell ({i},{j}): Point at ({x:.4f}, {y:.4f}), Color: ({r:.2f}, {g:.2f}, {b:.2f})")
        
        # Convert to tensor
        self.p = nn.Parameter(torch.tensor(params_list, dtype=torch.float32))
        self.out_dim = 3
    
    # @property
    # def p(self):
    #     """Return the parameters tensor"""
    #     return self.point_params

class TunnelIntegrandSlang(BaseIntegrandSlangRGB):
    def __init__(self):
        super().__init__()
        import math
        self.shader = slangtorch.loadModule("slang_gen/tunnel.slang")
        self.p = nn.Parameter(torch.tensor([
            3.0, 48.0, 0.08
        ]))
        self.out_dim = 3

class Sincos3dIntegrandSlang(BaseIntegrandSlangRGB):
    def __init__(self):
        super().__init__()
        self.shader = slangtorch.loadModule("slang_gen/sincos3d.slang")
        self.p = nn.Parameter(torch.tensor([
            0.2, 0.4, 0.6, 0.8,
            0.25, 0.45, 0.65, 0.85,
        ]))
        self.out_dim = 3


# class VoronoiSimpleIntegrandSlang(BaseIntegrandSlang):
#     def __init__(self):
#         super().__init__()
#         self.shader = slangtorch.loadModule("slang_gen/voronoi_simple.slang")
#         torch.manual_seed(12)
#         # N = 30
#         # N = 30
#         N = 4
#         # N = 16
#         grid = torch.linspace(0, 1, N+1)[:-1]
#         X, Y = torch.meshgrid(grid, grid, indexing='ij')
#         points = torch.stack([X.flatten(), Y.flatten()], dim=1)
#         points += torch.rand_like(points)/N
#         colors = torch.rand(N*N)
#         params = torch.zeros(N*N*3)
#         params[::3] = points[:,0]
#         params[1::3] = points[:,1] 
#         params[2::3] = colors
#         self.p = nn.Parameter(params)
#         self.out_dim = 1
#         # self.p = nn.Parameter(torch.tensor([
#         #     0.1, 0.5, 1.0, 0.9, 0.45, 0.0,
#         #     0.12, 0.56, 0.89, 0.34, 0.76, 0.09, 0.43, 0.65,
#         #     0.21, 0.98, 0.54, 0.33, 0.87, 0.11, 0.69, 0.44,
#         #     0.77, 0.31, 0.55, 0.90, 0.22, 0.66, 0.45, 0.88,
#         #     # 0.34, 0.82, 0.15, 0.95, 0.41, 0.28, 0.63, 0.85,
#         #     # 0.19, 0.52, 0.93, 0.37, 0.71, 0.13, 0.48, 0.61
#         # ]))

# class NCapsulesIntegrandSlang(BaseIntegrandSlang):
#     def __init__(self):
#         super().__init__()
#         self.shader = slangtorch.loadModule("slang_gen/n_capsules.slang")
#         self.p = nn.Parameter(torch.tensor([
#             0.5, 0.2, 0.3, 0.7, 0.1, 0.2, 0.5,
#             0.3, 0.4, 0.6, 0.8, 0.15, 0.1, 0.4,
#             0.7, 0.6, 0.8, 0.3, 0.12, 0.08, 0.6,
#             0.2, 0.8, 0.4, 0.9, 0.1, 0.15, 0.3,
#             0.8, 0.2, 0.9, 0.1, 0.08, 0.12, 0.5,
#             0.4, 0.7, 0.2, 0.5, 0.14, 0.11, 0.45
#         ]))



# class VectorGraphicsIntegrandSlang(BaseIntegrandSlang):
#     def __init__(self):
#         super().__init__()
#         self.shader = slangtorch.loadModule("slang_gen/vector_graphics.slang")
        
#         # Helper functions to create primitives with different fill types
#         def create_bezier_constant(p0x, p0y, p1x, p1y, p2x, p2y, width, color):
#             return [
#                 0.0,        # Type identifier (Bezier)
#                 p0x, p0y,   # p0 (start point)
#                 p1x, p1y,   # p1 (control point)
#                 p2x, p2y,   # p2 (end point)
#                 width,      # width
#                 0.0,        # Fill type (constant)
#                 color       # Color
#             ]
            
#         def create_bezier_gradient(p0x, p0y, p1x, p1y, p2x, p2y, width, 
#                                   grad_start_x, grad_start_y, grad_end_x, grad_end_y, 
#                                   start_color, end_color):
#             return [
#                 0.0,                # Type identifier (Bezier)
#                 p0x, p0y,           # p0 (start point)
#                 p1x, p1y,           # p1 (control point)
#                 p2x, p2y,           # p2 (end point)
#                 width,              # width
#                 1.0,                # Fill type (gradient)
#                 grad_start_x, grad_start_y,  # Gradient start
#                 grad_end_x, grad_end_y,      # Gradient end
#                 start_color,        # Start color
#                 end_color           # End color
#             ]
            
#         def create_triangle_constant(p0x, p0y, p1x, p1y, p2x, p2y, color):
#             return [
#                 1.0,        # Type identifier (Triangle)
#                 p0x, p0y,   # p0
#                 p1x, p1y,   # p1
#                 p2x, p2y,   # p2
#                 0.0,        # Fill type (constant)
#                 color       # Color
#             ]
            
#         def create_triangle_gradient(p0x, p0y, p1x, p1y, p2x, p2y,
#                                     grad_start_x, grad_start_y, grad_end_x, grad_end_y,
#                                     start_color, end_color):
#             return [
#                 1.0,                # Type identifier (Triangle)
#                 p0x, p0y,           # p0
#                 p1x, p1y,           # p1
#                 p2x, p2y,           # p2
#                 1.0,                # Fill type (gradient)
#                 grad_start_x, grad_start_y,  # Gradient start
#                 grad_end_x, grad_end_y,      # Gradient end
#                 start_color,        # Start color
#                 end_color           # End color
#             ]
            
#         def create_ellipse_constant(cx, cy, rx, ry, color):
#             return [
#                 2.0,        # Type identifier (Ellipse)
#                 cx, cy,     # center
#                 rx, ry,     # radii
#                 0.0,        # Fill type (constant)
#                 color       # Color
#             ]
            
#         def create_ellipse_gradient(cx, cy, rx, ry,
#                                    grad_start_x, grad_start_y, grad_end_x, grad_end_y,
#                                    start_color, end_color):
#             return [
#                 2.0,                # Type identifier (Ellipse)
#                 cx, cy,             # center
#                 rx, ry,             # radii
#                 1.0,                # Fill type (gradient)
#                 grad_start_x, grad_start_y,  # Gradient start
#                 grad_end_x, grad_end_y,      # Gradient end
#                 start_color,        # Start color
#                 end_color           # End color
#             ]
            
#         def create_circle_constant(cx, cy, r, color):
#             return [
#                 3.0,        # Type identifier (Circle)
#                 cx, cy,     # center
#                 r,          # radius
#                 0.0,        # Fill type (constant)
#                 color       # Color
#             ]
            
#         def create_circle_gradient(cx, cy, r,
#                                   grad_start_x, grad_start_y, grad_end_x, grad_end_y,
#                                   start_color, end_color):
#             return [
#                 3.0,                # Type identifier (Circle)
#                 cx, cy,             # center
#                 r,                  # radius
#                 1.0,                # Fill type (gradient)
#                 grad_start_x, grad_start_y,  # Gradient start
#                 grad_end_x, grad_end_y,      # Gradient end
#                 start_color,        # Start color
#                 end_color           # End color
#             ]
            
#         def create_line_constant(ax, ay, bx, by, width, color):
#             return [
#                 4.0,        # Type identifier (Line)
#                 ax, ay,     # point a
#                 bx, by,     # point b
#                 width,      # width
#                 0.0,        # Fill type (constant)
#                 color       # Color
#             ]
            
#         def create_line_gradient(ax, ay, bx, by, width,
#                                 grad_start_x, grad_start_y, grad_end_x, grad_end_y,
#                                 start_color, end_color):
#             return [
#                 4.0,                # Type identifier (Line)
#                 ax, ay,             # point a
#                 bx, by,             # point b
#                 width,              # width
#                 1.0,                # Fill type (gradient)
#                 grad_start_x, grad_start_y,  # Gradient start
#                 grad_end_x, grad_end_y,      # Gradient end
#                 start_color,        # Start color
#                 end_color           # End color
#             ]
        
#         # Create a parameter tensor with one of each primitive type and fill type
#         params = []
        
#         # Add a Bezier curve with constant fill
#         params.extend(create_bezier_constant(0.2, 0.2, 0.1, 0.5, 0.8, 0.8, 0.01, 0.3))
#         # # params.extend(create_bezier_constant(0.2, 0.2, 0.5, 0.8, 0.8, 0.2, 0.03, 0.9))
#         params.extend(create_bezier_constant(0.1, 0.2, 0.8, 0.8, 0.8, 0.2, 0.03, 0.9))
        
#         # Add a Triangle with linear gradient
#         # params.extend(create_triangle_gradient(0.3, 0.3, 0.7, 0.4, 0.5, 0.7, 
#                                             #   0.3, 0.3, 0.5, 0.7, 0.3, 0.8))
        
#         params.extend(create_triangle_constant(0.3, 0.3, 0.7, 0.4, 0.5, 0.7, 0.8))
        
#         # # Add an Ellipse with constant fill
#         # params.extend(create_ellipse_constant(0.5, 0.5, 0.3, 0.4, 0.7))
        
#         # # Add a Circle with constant fill
#         params.extend(create_circle_constant(0.2, 0.5, 0.1, 0.5))
        
#         # # Add a Circle with linear gradient
#         # params.extend(create_circle_gradient(0.3, 0.7, 0.1, 
#         #                                     0.2, 0.6, 0.4, 0.8, 0.2, 0.6))
        
#         # # Add a Line with constant fill
#         params.extend(create_line_constant(0.1, 0.1, 0.9, 0.9, 0.02, 0.85))
        
#         self.p = nn.Parameter(torch.tensor(params))


# Helper functions to create primitives with different fill types
def create_bezier_constant(p0x, p0y, p1x, p1y, p2x, p2y, width, r, g, b):
    return [
        0.0,        # Type identifier (Bezier)
        p0x, p0y,   # p0 (start point)
        p1x, p1y,   # p1 (control point)
        p2x, p2y,   # p2 (end point)
        width,      # width
        0.0,        # Fill type (constant)
        r, g, b     # Color (RGB)
    ]
    
def create_bezier_gradient(p0x, p0y, p1x, p1y, p2x, p2y, width, 
                            grad_start_x, grad_start_y, grad_end_x, grad_end_y, 
                            start_r, start_g, start_b, end_r, end_g, end_b):
    return [
        0.0,                # Type identifier (Bezier)
        p0x, p0y,           # p0 (start point)
        p1x, p1y,           # p1 (control point)
        p2x, p2y,           # p2 (end point)
        width,              # width
        1.0,                # Fill type (gradient)
        grad_start_x, grad_start_y,  # Gradient start
        grad_end_x, grad_end_y,      # Gradient end
        start_r, start_g, start_b,   # Start color (RGB)
        end_r, end_g, end_b          # End color (RGB)
    ]
    
def create_triangle_constant(p0x, p0y, p1x, p1y, p2x, p2y, r, g, b):
    return [
        1.0,        # Type identifier (Triangle)
        p0x, p0y,   # p0
        p1x, p1y,   # p1
        p2x, p2y,   # p2
        0.0,        # Fill type (constant)
        r, g, b     # Color (RGB)
    ]
    
def create_triangle_gradient(p0x, p0y, p1x, p1y, p2x, p2y,
                            grad_start_x, grad_start_y, grad_end_x, grad_end_y,
                            start_r, start_g, start_b, end_r, end_g, end_b):
    return [
        1.0,                # Type identifier (Triangle)
        p0x, p0y,           # p0
        p1x, p1y,           # p1
        p2x, p2y,           # p2
        1.0,                # Fill type (gradient)
        grad_start_x, grad_start_y,  # Gradient start
        grad_end_x, grad_end_y,      # Gradient end
        start_r, start_g, start_b,   # Start color (RGB)
        end_r, end_g, end_b          # End color (RGB)
    ]
    
def create_ellipse_constant(cx, cy, rx, ry, r, g, b):
    return [
        2.0,        # Type identifier (Ellipse)
        cx, cy,     # center
        rx, ry,     # radii
        0.0,        # Fill type (constant)
        r, g, b     # Color (RGB)
    ]
    
def create_ellipse_gradient(cx, cy, rx, ry,
                            grad_start_x, grad_start_y, grad_end_x, grad_end_y,
                            start_r, start_g, start_b, end_r, end_g, end_b):
    return [
        2.0,                # Type identifier (Ellipse)
        cx, cy,             # center
        rx, ry,             # radii
        1.0,                # Fill type (gradient)
        grad_start_x, grad_start_y,  # Gradient start
        grad_end_x, grad_end_y,      # Gradient end
        start_r, start_g, start_b,   # Start color (RGB)
        end_r, end_g, end_b          # End color (RGB)
    ]
    
def create_circle_constant(cx, cy, r, r_color, g, b):
    return [
        3.0,        # Type identifier (Circle)
        cx, cy,     # center
        r,          # radius
        0.0,        # Fill type (constant)
        r_color, g, b  # Color (RGB)
    ]
    
def create_circle_gradient(cx, cy, r,
                            grad_start_x, grad_start_y, grad_end_x, grad_end_y,
                            start_r, start_g, start_b, end_r, end_g, end_b):
    return [
        3.0,                # Type identifier (Circle)
        cx, cy,             # center
        r,                  # radius
        1.0,                # Fill type (gradient)
        grad_start_x, grad_start_y,  # Gradient start
        grad_end_x, grad_end_y,      # Gradient end
        start_r, start_g, start_b,   # Start color (RGB)
        end_r, end_g, end_b          # End color (RGB)
    ]
    
def create_line_constant(ax, ay, bx, by, width, r, g, b):
    return [
        4.0,        # Type identifier (Line)
        ax, ay,     # point a
        bx, by,     # point b
        width,      # width
        0.0,        # Fill type (constant)
        r, g, b     # Color (RGB)
    ]
    
def create_line_gradient(ax, ay, bx, by, width,
                        grad_start_x, grad_start_y, grad_end_x, grad_end_y,
                        start_r, start_g, start_b, end_r, end_g, end_b):
    return [
        4.0,                # Type identifier (Line)
        ax, ay,             # point a
        bx, by,             # point b
        width,              # width
        1.0,                # Fill type (gradient)
        grad_start_x, grad_start_y,  # Gradient start
        grad_end_x, grad_end_y,      # Gradient end
        start_r, start_g, start_b,   # Start color (RGB)
        end_r, end_g, end_b          # End color (RGB)
    ]


class VectorGraphicsRGBIntegrandSlang(BaseIntegrandSlangRGB):
    def __init__(self):
        super().__init__()
        self.shader = slangtorch.loadModule("slang_gen/vector_graphics_rgb.slang")
        
        # Create a parameter tensor with one of each primitive type and fill type
        params = []
        # params.extend(create_bezier_constant(0.2, 0.2, 0.1, 0.5, 0.8, 0.8, 0.01, 1.0, 1.0, 0.0))
        # params.extend(create_bezier_constant(0.1, 0.2, 0.8, 0.8, 0.8, 0.2, 0.03, 0.9, 0.9, 0.9))
        # params.extend(create_triangle_constant(0.3, 0.3, 0.7, 0.4, 0.5, 0.7, 0.0, 1.0, 0.0))
        # params.extend(create_ellipse_constant(0.1, 0.1, 0.3, 0.4, 0.7, 0.1, 0.2))
        params.extend(create_circle_constant(0.5122, 0.52345, 0.3, 0.0, 1.0, 1.0))
        self.p = nn.Parameter(torch.tensor(params))
        self.out_dim = 3


class VectorGraphicsRGB2IntegrandSlang(BaseIntegrandSlangRGB):
    def __init__(self):
        super().__init__()
        self.shader = slangtorch.loadModule("slang_gen/vector_graphics_rgb.slang")
        
        # Create a parameter tensor with one of each primitive type and fill type
        params = []
        # Set random seed for reproducibility
        torch.manual_seed(42)
        
        # Helper function to generate random colors
        def random_color():
            return torch.rand(3).tolist()
        
        # Create 10 triangles with random positions and colors
        # Define grid size
        n = 5  # 3x3 grid = 9 triangles
        jitter_factor = 1.0  # Default jitter is 100% of cell width
        
        # Create triangles in a grid pattern
        for i in range(n):
            for j in range(n):
                # Calculate base point from grid position
                base_x = (i + 0.5) / n  # Center in grid cell
                base_y = (j + 0.5) / n  # Center in grid cell
                
                # Add configurable random jitter
                cell_width = 1.0 / n
                base_x += (torch.rand(1).item() - 0.5) * jitter_factor * cell_width
                base_y += (torch.rand(1).item() - 0.5) * jitter_factor * cell_width
                
                # Create more balanced triangles by using angles instead of random displacements
                radius = 0.4 / n  # Control the size of the triangle
                
                # Generate three points around the base point using angles
                angles = torch.rand(3) * 2 * 3.14159  # Random angles in [0, 2π]
                
                # Sort angles to make more regular triangles
                angles, _ = torch.sort(angles)
                
                # Calculate triangle vertices using polar coordinates
                p0x = base_x + radius * torch.cos(angles[0]).item()
                p0y = base_y + radius * torch.sin(angles[0]).item()
                p1x = base_x + radius * torch.cos(angles[1]).item()
                p1y = base_y + radius * torch.sin(angles[1]).item()
                p2x = base_x + radius * torch.cos(angles[2]).item()
                p2y = base_y + radius * torch.sin(angles[2]).item()
                
                # Ensure the triangle stays within [0,1] bounds
                p0x = min(max(p0x, 0.0), 1.0)
                p0y = min(max(p0y, 0.0), 1.0)
                p1x = min(max(p1x, 0.0), 1.0)
                p1y = min(max(p1y, 0.0), 1.0)
                p2x = min(max(p2x, 0.0), 1.0)
                p2y = min(max(p2y, 0.0), 1.0)
                
                # Add the triangle with random color
                params.extend(create_triangle_constant(p0x, p0y, p1x, p1y, p2x, p2y, *random_color()))
                
                # Add a circle with jittered position in the same grid cell
                circle_x = (i + 0.5) / n + (torch.rand(1).item() - 0.5) * jitter_factor * cell_width
                circle_y = (j + 0.5) / n + (torch.rand(1).item() - 0.5) * jitter_factor * cell_width
                circle_r = torch.rand(1).item() * 0.1 / n + 0.05 / n  # Random radius
                
                # Ensure circle stays within bounds
                circle_x = min(max(circle_x, circle_r), 1.0 - circle_r)
                circle_y = min(max(circle_y, circle_r), 1.0 - circle_r)
                
                # Add the circle with random color
                params.extend(create_circle_constant(circle_x, circle_y, circle_r, *random_color()))
                
                # Add an ellipse with jittered position in the same grid cell
                ellipse_x = (i + 0.5) / n + (torch.rand(1).item() - 0.5) * jitter_factor * cell_width
                ellipse_y = (j + 0.5) / n + (torch.rand(1).item() - 0.5) * jitter_factor * cell_width
                ellipse_rx = torch.rand(1).item() * 0.12 / n + 0.04 / n  # Random x-radius
                ellipse_ry = torch.rand(1).item() * 0.12 / n + 0.04 / n  # Random y-radius
                
                # Ensure ellipse stays within bounds
                ellipse_x = min(max(ellipse_x, ellipse_rx), 1.0 - ellipse_rx)
                ellipse_y = min(max(ellipse_y, ellipse_ry), 1.0 - ellipse_ry)
                
                # Add the ellipse with random color
                params.extend(create_ellipse_constant(ellipse_x, ellipse_y, ellipse_rx, ellipse_ry, *random_color()))

                
        
        # # Create 10 ellipses with random positions, sizes and colors
        # for _ in range(10):
        #     cx, cy = torch.rand(2).tolist()
        #     # Random radii between 0.02 and 0.1
        #     rx, ry = (torch.rand(2) * 0.08 + 0.02).tolist()
        #     # Ensure ellipse stays within bounds
        #     cx = min(max(cx, rx), 1.0 - rx)
        #     cy = min(max(cy, ry), 1.0 - ry)
        #     params.extend(create_ellipse_constant(cx, cy, rx, ry, *random_color()))
        
        # # Create 10 circles with random positions, sizes and colors
        # for _ in range(10):
        #     cx, cy = torch.rand(2).tolist()
        #     # Random radius between 0.02 and 0.1
        #     r = torch.rand(1).item() * 0.08 + 0.02
        #     # Ensure circle stays within bounds
        #     cx = min(max(cx, r), 1.0 - r)
        #     cy = min(max(cy, r), 1.0 - r)
        #     params.extend(create_circle_constant(cx, cy, r, *random_color()))
        # params.extend(create_bezier_constant(0.2, 0.2, 0.1, 0.5, 0.8, 0.8, 0.01, 1.0, 1.0, 0.0))
        # params.extend(create_bezier_constant(0.1, 0.2, 0.8, 0.8, 0.8, 0.2, 0.03, 0.9, 0.9, 0.9))
        # params.extend(create_triangle_constant(0.3, 0.3, 0.7, 0.4, 0.5, 0.7, 0.0, 1.0, 0.0))
        # params.extend(create_ellipse_constant(0.1, 0.1, 0.3, 0.4, 0.7, 0.1, 0.2))
        # params.extend(create_circle_constant(0.5, 0.5, 0.3, 0.0, 1.0, 1.0))
        self.p = nn.Parameter(torch.tensor(params))
        self.out_dim = 3