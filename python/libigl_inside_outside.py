# %%
import numpy as np
import os
import plyfile
import igl

class MeshInsideOutsideTest:
    """
    A class for performing inside-outside testing on 3D meshes using libigl's fast winding number.
    """
    
    def __init__(self, ply_file):
        """
        Initialize the inside-outside tester with a mesh file.
        
        Args:
            ply_file: Path to a PLY file containing the mesh
        """
        self.vertices, self.faces = self._load_and_normalize_mesh(ply_file)
        self.mesh_stats = self._compute_mesh_stats()
        
    def _load_and_normalize_mesh(self, ply_file):
        """
        Load a mesh from a PLY file and normalize it to fit in [0,1]^3.
        """
        # Only support PLY files
        _, ext = os.path.splitext(ply_file)
        if ext.lower() != '.ply':
            raise ValueError(f"Only PLY files are supported, got: {ext}")
        
        # Load the PLY file using plyfile
        plydata = plyfile.PlyData.read(ply_file)
        
        # Extract vertices
        vertex_data = plydata['vertex']
        vertices = np.vstack([
            vertex_data['x'],
            vertex_data['y'],
            vertex_data['z']
        ]).T
        
        # Extract faces - assuming they're stored as lists of vertex indices
        face_data = plydata['face']
        faces = np.stack([f[0] for f in face_data], axis=0)
        
        # Get the bounding box
        min_coords = np.min(vertices, axis=0)
        max_coords = np.max(vertices, axis=0)
        
        # Calculate the translation to center the mesh
        translation = -min_coords
        
        # Calculate the scale to fit in [0,1]^3
        scale = 1.0 / np.max(max_coords - min_coords)
        
        # Apply the transformation
        vertices = (vertices + translation) * scale
        
        return vertices, faces
    
    def _compute_mesh_stats(self):
        """Compute basic statistics about the mesh vertices."""
        stats = {
            'x': {
                'min': np.min(self.vertices[:, 0]),
                'max': np.max(self.vertices[:, 0]),
                'mean': np.mean(self.vertices[:, 0]),
                'std': np.std(self.vertices[:, 0])
            },
            'y': {
                'min': np.min(self.vertices[:, 1]),
                'max': np.max(self.vertices[:, 1]),
                'mean': np.mean(self.vertices[:, 1]),
                'std': np.std(self.vertices[:, 1])
            },
            'z': {
                'min': np.min(self.vertices[:, 2]),
                'max': np.max(self.vertices[:, 2]),
                'mean': np.mean(self.vertices[:, 2]),
                'std': np.std(self.vertices[:, 2])
            },
            'overall': {
                'min': np.min(self.vertices),
                'max': np.max(self.vertices),
                'mean': np.mean(self.vertices),
                'std': np.std(self.vertices)
            }
        }
        return stats
    
    def __call__(self, points):
        """
        Check if points are inside the mesh.
        
        Args:
            points: numpy array or torch tensor of shape (N, 3) containing N 3D points
            
        Returns:
            numpy array or torch tensor of shape (N,) with boolean values (True if inside)
        """
        # Check if points is a torch tensor
        is_torch_tensor = False
        original_device = None
        original_dtype = None
        
        if hasattr(points, 'device') and hasattr(points, 'dtype'):  # It's a torch tensor
            is_torch_tensor = True
            original_device = points.device
            original_dtype = points.dtype
            points = points.detach().cpu().numpy()
        
        # Ensure all inputs have the correct data type (float32)
        vertices = np.asfortranarray(self.vertices.astype(np.float32))
        faces = np.asfortranarray(self.faces.astype(np.int32))
        points = np.asfortranarray(points.astype(np.float32))
        
        # Use winding number method
        winding = igl.fast_winding_number_for_meshes(vertices, faces, points)
        
        # Points with winding number > 0.5 are inside
        inside = winding > 0.5
        
        # Convert back to torch tensor if input was a torch tensor
        if is_torch_tensor:
            import torch
            inside = torch.tensor(inside, dtype=torch.float32, device=original_device)
            # print(inside.shape)
            # swdfs
        
        return inside
    
    def compute_slice(self, resolution, plane_value, constant_axis=2):
        """
        Compute a 2D slice of the mesh at a specific plane.
        
        Args:
            resolution: Resolution of the 2D grid
            plane_value: Value along the constant axis where to slice
            constant_axis: Which axis to keep constant (0=x, 1=y, 2=z)
        
        Returns:
            2D numpy array with boolean values (True if inside)
            List of axis names for the non-constant axes
        """
        # Create a 2D grid
        grid = np.linspace(0, 1, resolution)
        xx, yy = np.meshgrid(grid, grid)
        
        # Create 3D points based on which axis is constant
        points = np.zeros((resolution * resolution, 3))
        
        if constant_axis == 0:  # x is constant
            points[:, 0] = plane_value
            points[:, 1] = xx.flatten()
            points[:, 2] = yy.flatten()
            axis_names = ['y', 'z']
        elif constant_axis == 1:  # y is constant
            points[:, 0] = xx.flatten()
            points[:, 1] = plane_value
            points[:, 2] = yy.flatten()
            axis_names = ['x', 'z']
        else:  # z is constant
            points[:, 0] = xx.flatten()
            points[:, 1] = yy.flatten()
            points[:, 2] = plane_value
            axis_names = ['x', 'y']
        
        # Check which points are inside the mesh
        inside = self(points)
        
        # Reshape to 2D grid
        return inside.reshape(resolution, resolution), axis_names
    
    def print_stats(self):
        """Print statistics about the mesh."""
        stats = self.mesh_stats
        print(f"X-axis: min={stats['x']['min']:.4f}, mean={stats['x']['mean']:.4f}, max={stats['x']['max']:.4f}, std={stats['x']['std']:.4f}")
        print(f"Y-axis: min={stats['y']['min']:.4f}, mean={stats['y']['mean']:.4f}, max={stats['y']['max']:.4f}, std={stats['y']['std']:.4f}")
        print(f"Z-axis: min={stats['z']['min']:.4f}, mean={stats['z']['mean']:.4f}, max={stats['z']['max']:.4f}, std={stats['z']['std']:.4f}")
        print(f"Overall: min={stats['overall']['min']:.4f}, mean={stats['overall']['mean']:.4f}, max={stats['overall']['max']:.4f}, std={stats['overall']['std']:.4f}")
