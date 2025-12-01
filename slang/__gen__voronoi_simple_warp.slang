import utils;

import utils;

// Hash function for Perlin noise
float hash(float2 p) {
    p = 50.0 * frac(p * 0.3183099);
    return frac(p.x * p.y * (p.x + p.y));
}

// Return random vector 
float2 getGradient(float2 cell) {
    float angle = hash(cell) * 6.28318530718; // 2*PI
    return float2(cos(angle), sin(angle));
}

// Compute Perlin noise at position p
float perlin(float2 p) {
    float2 cell = floor(p);
    float2 frac = frac(p);
    
    // Compute dot products with gradients
    float2 g00 = getGradient(cell);
    float2 g10 = getGradient(cell + float2(1.0, 0.0));
    float2 g01 = getGradient(cell + float2(0.0, 1.0));
    float2 g11 = getGradient(cell + float2(1.0, 1.0));
    
    float d00 = dot(g00, frac);
    float d10 = dot(g10, frac - float2(1.0, 0.0));
    float d01 = dot(g01, frac - float2(0.0, 1.0));
    float d11 = dot(g11, frac - float2(1.0, 1.0));
    
    // Smooth interpolation (using Hermite cubic)
    float2 w = frac * frac * (3.0 - 2.0 * frac);
    
    float x1 = mix(d00, d10, w.x);
    float x2 = mix(d01, d11, w.x);
    
    return mix(x1, x2, w.y) * 0.5 + 0.5; // Map from [-0.5, 0.5] to [0, 1]
}

// Apply domain warping using multiple octaves of Perlin noise
float2 warpDomain(float2 p, float strength, int octaves) {
    float2 offset = float2(0.0, 0.0);
    float amplitude = strength;
    float frequency = 4.0;
    
    [MaxIters(8)]
    for (int i = 0; i < octaves; i++) {
        float nx = perlin(p * frequency + offset);
        float ny = perlin(p * frequency + offset + float2(5.2, 1.3));
        
        offset = float2(nx, ny) * amplitude;
        p += offset;
        
        amplitude *= 0.5;
        frequency *= 2.0;
    }
    
    return p;
}

// Modified from the second shader at: https://thebookofshaders.com/12/
[Differentiable]
void start(float2 x, DiffTensorView<float> p, DiffTensorView<float> res, int thread_idx, int if_off, int out_off, bool ret_const, bool ret_impl, DiffTensorView<float> impl_fn, TensorView<int> impl_idx, TensorView<int> out_idx, bool should_force, bool force_value) {
    // Warp the input coordinate using Perlin noise
    float warp_strength = 0.15; // Adjust this to control warping intensity
    int octaves = 2;           // Number of noise octaves
    float2 warped_x = no_diff warpDomain(x, warp_strength, octaves);
    
    // Standard Voronoi computation, but using warped coordinates
    float m_dist = 10.;  // minimum distance
    
    [MaxIters(1000)]
    for (int i = 0; i < p.size(0) / 5; i++) {
        float2 point = float2(load(p, i*5 + 0), load(p, i*5 + 1));
        
        // Vector between the warped pixel and the point
        float2 diff = warped_x - point;
        
        // Distance to the point
        float dist = length(diff);
        float test = m_dist - dist;
        
        
        
        // ---- Start: implicit function test ----
        if (ret_impl && abs(test) < abs(impl_fn[thread_idx])) {
            impl_idx[thread_idx] = if_off + i*1 + 0;
            impl_fn[thread_idx] = test;
        }
        // ---- End: implicit function test ----

        if (should_force && load(impl_idx, thread_idx) == (if_off + i*1 + 0) ? force_value : test > 0.0) {
                if(ret_const) { 
                out_idx[thread_idx] = hash(out_idx[thread_idx], (if_off + i*1 + 0)*2);
            }
m_dist = dist;
            res[thread_idx, 0] = load(p, i*5 + 2);
            res[thread_idx, 1] = load(p, i*5 + 3);
            res[thread_idx, 2] = load(p, i*5 + 4);
        }
    }
}

[Differentiable]
[AutoPyBindCUDA]
[CUDAKernel]
void run(
    DiffTensorView<float> x,
    DiffTensorView<float> p,
    DiffTensorView<float> r
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
    float2 x_ = float2(x[i,0], x[i,1]);
    start(x_, p, r, i, if_off + 0, out_off + 0, ret_const, ret_impl, impl_fn, impl_idx, out_idx, should_force, force_value);
}
