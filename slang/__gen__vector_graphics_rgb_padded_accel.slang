import utils;

import utils;

// Signed distance function for a circle
[Differentiable]
float sdCircle(float2 p, float2 center, float radius) {
    return length(p - center) - radius;
}

// Implicit function for an ellipse
[Differentiable]
float implicitEllipse(float2 p, float2 center, float2 ab) {
    p = p - center;
    
    // The implicit equation for an ellipse is (x/a)^2 + (y/b)^2 = 1
    float value = (p.x * p.x) / (ab.x * ab.x) + (p.y * p.y) / (ab.y * ab.y) - 1.0;
    return value;
}

// Implicit function for a rotated ellipse using covariance matrix
[Differentiable]
float implicitRotatedEllipse(float2 p, float2 center, float3 cov) {
    p = p - center;
    
    // The 4-vector cov represents a 2x2 covariance matrix:
    // [cov.x cov.y]
    // [cov.y cov.z]
    float a = cov.x;  // top-left element
    float b = cov.y;  // off-diagonal element
    float c = cov.z;  // bottom-right element
    
    // Compute the determinant of the covariance matrix
    float det = a*c - b*b;
    
    // Compute the inverse of the covariance matrix
    float inv_a = c/det;
    float inv_b = -b/det;
    float inv_c = a/det;
    
    // Compute (x-μ)ᵀΣ⁻¹(x-μ) - 1
    float value = p.x*(inv_a*p.x + inv_b*p.y) + p.y*(inv_b*p.x + inv_c*p.y) - 1.0;
    return value;
}

// Signed distance function for a line segment with width
[Differentiable]
float sdLine(float2 p, float2 a, float2 b, float width) {
    float2 pa = p - a;
    float2 ba = b - a;
    float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
    return length(pa - ba * h) - width;
}

[Differentiable]
float aboveLine(float2 p, float2 a, float2 b) {
    float2 pa = p - a;
    float2 ba = b - a;
    return -cross(pa, ba);
}

// Signed distance function for a triangle
[Differentiable]
float sdTriangle(float2 p, float2 p0, float2 p1, float2 p2) {
    // Calculate distance to each edge using sdLine
    float d_e0 = aboveLine(p, p0, p1);
    float d_e1 = aboveLine(p, p1, p2);
    float d_e2 = aboveLine(p, p2, p0);

    float dist = min(min(d_e0, d_e1), d_e2);

    return -dist;
}

[Differentiable]
float sdBezier(float2 pos, float2 A, float2 B, float2 C)
{    
    float2 a = B - A;
    float2 b = A - 2.0*B + C;
    float2 c = a * 2.0;
    float2 d = A - pos;
    float kk = 1.0/dot(b,b);
    float kx = kk * dot(a,b);
    float ky = kk * (2.0*dot(a,a)+dot(d,b)) / 3.0;
    float kz = kk * dot(d,a);      
    float res = 0.0;
    float p = ky - kx*kx;
    float p3 = p*p*p;
    float q = kx*(2.0*kx*kx-3.0*ky) + kz;
    float h = q*q + 4.0*p3;
    if(h >= 0.0) 
    { 
        h = sqrt(h);
        float2 x = (float2(h,-h)-q)/2.0;
        float2 uv = sign(x)*pow(abs(x), float2(1.0/3.0));
        float t = clamp(uv.x+uv.y-kx, 0.0, 1.0);
        res = dot2(d + (c + b*t)*t);
    }
    else
    {
        float z = sqrt(-p);
        float v = acos(q/(p*z*2.0)) / 3.0;
        float m = cos(v);
        float n = sin(v)*1.732050808;
        float3 t = clamp(float3(m+m,-n-m,n-m)*z-kx, 0.0, 1.0);
        res = min(dot2(d+(c+b*t.x)*t.x),
                 dot2(d+(c+b*t.y)*t.y));
    }
    return sqrt(res);
}

[Differentiable]
float sdQuadraticBezier(float2 p, float2 p0, float2 p1, float2 p2, float width) {
    float dist = sdBezier(p, p0, p1, p2);
    return dist - width;
}


[Differentiable]
[PreferRecompute]
float load_eval_bezier(float2 x, DiffTensorView<float> p, int idx) {
    float2 p0 = float2(load(p, idx), load(p, idx + 1));
    float2 p1 = float2(load(p, idx + 2), load(p, idx + 3));
    float2 p2 = float2(load(p, idx + 4), load(p, idx + 5));
    float width = load(p, idx + 6);
    return -sdQuadraticBezier(x, p0, p1, p2, width);
}

[Differentiable]
[PreferRecompute]
float load_eval_triangle(float2 x, DiffTensorView<float> p, int idx) {
    float2 p0 = float2(load(p, idx), load(p, idx + 1));
    float2 p1 = float2(load(p, idx + 2), load(p, idx + 3));
    float2 p2 = float2(load(p, idx + 4), load(p, idx + 5));
    return -sdTriangle(x, p0, p1, p2);
}

[Differentiable]
[PreferRecompute]
float load_eval_ellipse(float2 x, DiffTensorView<float> p, int idx) {
    float2 center = float2(load(p, idx), load(p, idx + 1));
    float2 radii = float2(load(p, idx + 2), load(p, idx + 3));
    return -implicitEllipse(x, center, radii);
}

[Differentiable]
[PreferRecompute]
float load_eval_rotated_ellipse(float2 x, DiffTensorView<float> p, int idx) {
    float2 center = float2(load(p, idx), load(p, idx + 1));
    float3 cov = float3(load(p, idx + 2), load(p, idx + 3), load(p, idx + 4));
    return -implicitRotatedEllipse(x, center, cov);
}

[Differentiable]
[PreferRecompute]
float load_eval_circle(float2 x, DiffTensorView<float> p, int idx) {
    float2 center = float2(load(p, idx), load(p, idx + 1));
    float radius = load(p, idx + 2);
    return -sdCircle(x, center, radius);
}

[Differentiable]
[PreferRecompute]
float load_eval_line(float2 x, DiffTensorView<float> p, int idx) {
    float2 a = float2(load(p, idx), load(p, idx + 1));
    float2 b = float2(load(p, idx + 2), load(p, idx + 3));
    float width = load(p, idx + 4);
    return -sdLine(x, a, b, width);
}



[Differentiable]
void start(float2 x, DiffTensorView<float> p, DiffTensorView<float> res, int thread_idx, int if_off, int out_off, bool ret_const, bool ret_impl, DiffTensorView<float> impl_fn, TensorView<int> impl_idx, TensorView<int> out_idx, bool should_force, bool force_value) {
    // Read header information
    int num_primitives = int(load(p, 0));
    int grid_size = int(load(p, 1));
    int max_elements_per_cell = int(load(p, 2));
    
    // Read background color
    float3 bg_color = float3(load(p, 3), load(p, 4), load(p, 5));
    res[thread_idx, 0] = 0.0;
    res[thread_idx, 1] = 0.0;
    res[thread_idx, 2] = 0.0;
    
    
    // Calculate grid cell for this point
    int cell_x = int(x.x * grid_size);
    int cell_y = int(x.y * grid_size);
    
    // Clamp to valid grid range
    cell_x = clamp(cell_x, 0, grid_size - 1);
    cell_y = clamp(cell_y, 0, grid_size - 1);
    
    // Calculate offset to grid data in parameter buffer (now includes 3 more values for bg color)
    int grid_offset = 6 + num_primitives * 20;
    int cell_offset = grid_offset + (cell_x * grid_size + cell_y) * (max_elements_per_cell + 1);
    
    // Get number of primitives in this cell
    int cell_count = int(load(p, cell_offset));

    float transmittance = 1.0;
    
    // Loop through primitives in this cell
    [MaxIters(400)] // Assuming max_elements_per_cell <= 20
    for (int i = 0; i < max_elements_per_cell; i++) {
        // if (i >= cell_count) break;
        if (i >= cell_count || i >= 400) break;
        
        // Get primitive index
        int prim_idx = int(load(p, cell_offset + i + 1));
        int idx = 6 + prim_idx * 20; // 6 for header + bg color, 20 params per primitive
        
        int primitive_type = int(load(p, idx));
        int off = 1;
        
        float test = 0.0;

        if (primitive_type == 0) {  // Bezier curve
            test = load_eval_bezier(x, p, idx+off);
        }
        if (primitive_type == 1) {  // Triangle
            test = load_eval_triangle(x, p, idx+off);
        }
        if (primitive_type == 2) {  // Ellipse
            test = load_eval_ellipse(x, p, idx+off);
        }
        if (primitive_type == 3) {  // Circle
            test = load_eval_circle(x, p, idx+off);
        }
        if (primitive_type == 4) {  // Line
            test = load_eval_line(x, p, idx+off);
        }
        
        off += 7;
        int fillType = int(load(p, idx+off));
        off += 1;

        float3 color = -1.0;
        float2 gradStart, gradEnd;
        float3 startColor, endColor;
        float opacity;
        if (fillType == 0) {
            color = float3(load(p, idx+off), load(p, idx+off+1), load(p, idx+off+2));
            opacity = load(p, idx+off+3);
        } else if (fillType == 1) {
            gradStart = float2(load(p, idx+off), load(p, idx+off+1));
            gradEnd = float2(load(p, idx+off+2), load(p, idx+off+3));
            startColor = float3(load(p, idx+off+4), load(p, idx+off+5), load(p, idx+off+6));  
            endColor = float3(load(p, idx+off+7), load(p, idx+off+8), load(p, idx+off+9));
        }
        
        
        
        // ---- Start: implicit function test ----
        if (ret_impl && abs(test) < abs(impl_fn[thread_idx])) {
            impl_idx[thread_idx] = if_off + i*1 + 0;
            impl_fn[thread_idx] = test;
        }
        // ---- End: implicit function test ----

        if (should_force && load(impl_idx, thread_idx) == (if_off + i*1 + 0) ? force_value : test > 0.0) {
                            if(ret_const) {
                                out_idx[thread_idx] = hash(out_idx[thread_idx], prim_idx);
                            }

            res[thread_idx, 0] += transmittance * opacity * color.x;
            res[thread_idx, 1] += transmittance * opacity * color.y;
            res[thread_idx, 2] += transmittance * opacity * color.z;
            transmittance *= (1.0 - opacity);
            if (transmittance < 0.01) {
                return;
            }
        }
    }

    res[thread_idx, 0] += transmittance * bg_color.x;
    res[thread_idx, 1] += transmittance * bg_color.y;
    res[thread_idx, 2] += transmittance * bg_color.z;

}

[Differentiable]
[AutoPyBindCUDA]
[CUDAKernel]
void run(
    DiffTensorView<float> x, // Evaluation points
    DiffTensorView<float> p, // Parameters buffer with primitive type identifiers and acceleration structure
    DiffTensorView<float> r  // Return value
, DiffTensorView<float> impl_fn, TensorView<int> impl_idx, TensorView<int> out_idx, int force_sign, bool ret_const, bool ret_impl)
{
    bool should_force = force_sign != -1;
    bool force_value = force_sign == 0;
    int if_off = 0; 
    int out_off = 0; 


    uint3 dispatch_id = cudaBlockIdx() * cudaBlockDim() + cudaThreadIdx();
    int X_NUM = x.size(0);
    int i = dispatch_id.x;
    if (i >= X_NUM) return;
    r[i] = 0.0;  // Initialize to 0
    float2 x_ = float2(x[i,0], x[i,1]);
    start(x_, p, r, i, if_off + 0, out_off + 0, ret_const, ret_impl, impl_fn, impl_idx, out_idx, should_force, force_value);
} 