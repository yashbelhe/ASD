import utils;

[AutoPyBindCUDA]
[CUDAKernel]
void setup_unified_acceleration_structure(
    TensorView<int> primitive_types,      // [num_primitives] tensor of primitive types
    TensorView<float> control_points,     // [num_primitives, 6] tensor of control points
    TensorView<float> stroke_widths,      // [num_primitives, 1] tensor of stroke widths
    TensorView<int> primitive_indices,    // [num_primitives] tensor of primitive indices
    TensorView<int> grid_tensor,          // [grid_size * grid_size * (max_elements_per_cell + 1)] flattened grid
    int grid_size,                        // Size of the grid
    int max_elements_per_cell             // Maximum elements per cell
) {
    uint3 dispatch_id = cudaBlockIdx() * cudaBlockDim() + cudaThreadIdx();
    int prim_idx = dispatch_id.x;
    
    // Return if this thread is beyond the number of primitives
    if (prim_idx >= primitive_indices.size(0)) return;
    
    // Get the actual primitive index
    int primitive_idx = primitive_indices[prim_idx];
    
    // Get primitive type
    int primitive_type = primitive_types[prim_idx];
    
    // Initialize bounding box
    float min_x = 1.0;
    float min_y = 1.0;
    float max_x = 0.0;
    float max_y = 0.0;
    
    // Get stroke width (only used for bezier curves and lines)
    float stroke_width = stroke_widths[prim_idx];
    
    // Compute bounding box based on primitive type
    if (primitive_type == 0) {  // Bezier curve
        // Get control points
        float2 p0 = float2(control_points[prim_idx, 0], control_points[prim_idx, 1]);
        float2 p1 = float2(control_points[prim_idx, 2], control_points[prim_idx, 3]);
        float2 p2 = float2(control_points[prim_idx, 4], control_points[prim_idx, 5]);
        
        // Compute bounding box
        min_x = min(min(p0.x, p1.x), p2.x) - stroke_width;
        min_y = min(min(p0.y, p1.y), p2.y) - stroke_width;
        max_x = max(max(p0.x, p1.x), p2.x) + stroke_width;
        max_y = max(max(p0.y, p1.y), p2.y) + stroke_width;
    }
    else if (primitive_type == 1) {  // Triangle
        // Get vertices
        float2 p0 = float2(control_points[prim_idx, 0], control_points[prim_idx, 1]);
        float2 p1 = float2(control_points[prim_idx, 2], control_points[prim_idx, 3]);
        float2 p2 = float2(control_points[prim_idx, 4], control_points[prim_idx, 5]);
        
        // Compute bounding box
        min_x = min(min(p0.x, p1.x), p2.x);
        min_y = min(min(p0.y, p1.y), p2.y);
        max_x = max(max(p0.x, p1.x), p2.x);
        max_y = max(max(p0.y, p1.y), p2.y);
    }
    else if (primitive_type == 2) {  // Ellipse
        // Get center and radii
        float2 center = float2(control_points[prim_idx, 0], control_points[prim_idx, 1]);
        float2 radii = float2(control_points[prim_idx, 2], control_points[prim_idx, 3]);
        
        // Compute bounding box
        min_x = center.x - radii.x;
        min_y = center.y - radii.y;
        max_x = center.x + radii.x;
        max_y = center.y + radii.y;
    }
    else if (primitive_type == 3) {  // Circle
        // Get center and radius
        float2 center = float2(control_points[prim_idx, 0], control_points[prim_idx, 1]);
        float radius = control_points[prim_idx, 2];
        
        // Compute bounding box
        min_x = center.x - radius;
        min_y = center.y - radius;
        max_x = center.x + radius;
        max_y = center.y + radius;
    }
    else if (primitive_type == 4) {  // Line
        // Get endpoints
        float2 p0 = float2(control_points[prim_idx, 0], control_points[prim_idx, 1]);
        float2 p1 = float2(control_points[prim_idx, 2], control_points[prim_idx, 3]);
        
        // Compute bounding box
        min_x = min(p0.x, p1.x) - stroke_width;
        min_y = min(p0.y, p1.y) - stroke_width;
        max_x = max(p0.x, p1.x) + stroke_width;
        max_y = max(p0.y, p1.y) + stroke_width;
    } else if (primitive_type == 5) {  // Rotated Ellipse
        // Get center and radii
        float2 center = float2(control_points[prim_idx, 0], control_points[prim_idx, 1]);
        float3 cov = float3(control_points[prim_idx, 2], control_points[prim_idx, 3], control_points[prim_idx, 4]);
        
        // Compute eigenvalues and eigenvectors of the covariance matrix
        float trace = cov.x + cov.z;
        float det = cov.x * cov.z - cov.y * cov.y;
        float temp = max(trace * trace - 4.0 * det, 1e-10);
        
        // Compute eigenvalues
        float lambda1 = (trace + sqrt(temp)) / 2.0;
        float lambda2 = (trace - sqrt(temp)) / 2.0;
        
        // prevent negative sqrt input
        lambda1 = max(lambda1, 1e-10);
        lambda2 = max(lambda2, 1e-10);
        
        // Compute major and minor axis lengths (2*sqrt(eigenvalue))
        float major_axis = 2.0 * sqrt(lambda1);
        float minor_axis = 2.0 * sqrt(lambda2);
        
        // Use the larger of major and minor axis for both dimensions
        float box_size = max(major_axis, minor_axis);
        
        // Compute bounding box
        min_x = center.x - box_size;
        min_y = center.y - box_size;
        max_x = center.x + box_size;
        max_y = center.y + box_size;
    }
    
    // Convert to grid cell coordinates
    int min_cell_x = int(min_x * grid_size);
    int min_cell_y = int(min_y * grid_size);
    int max_cell_x = int(max_x * grid_size);
    int max_cell_y = int(max_y * grid_size);
    
    // For each cell this primitive intersects
    for (int cell_x = min_cell_x; cell_x <= max_cell_x; cell_x++) {
        for (int cell_y = min_cell_y; cell_y <= max_cell_y; cell_y++) {
            if (cell_x < 0 || cell_x >= grid_size || cell_y < 0 || cell_y >= grid_size) {
                continue;
            }
            // Calculate 1D index for the cell count
            int cell_idx = (cell_x * grid_size + cell_y) * (max_elements_per_cell + 1);
            
            // Atomically increment the count and get the previous value
            int old_count = 0;
            grid_tensor.InterlockedAdd(cell_idx, 1, old_count);
            
            // If there's room in this cell
            if (old_count < max_elements_per_cell) {
                // Add primitive index to cell at the appropriate position
                grid_tensor[cell_idx + old_count + 1] = primitive_idx;
            }
        }
    }
}
