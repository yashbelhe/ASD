import utils;

// Hash function for Perlin noise
[NoDiscontinuity]
float hash(float2 p) {
    p = 50.0 * frac(p * 0.3183099);
    return frac(p.x * p.y * (p.x + p.y));
}

// Return random vector 
[NoDiscontinuity]
float2 getGradient(float2 cell) {
    float angle = hash(cell) * 6.28318530718; // 2*PI
    return float2(cos(angle), sin(angle));
}

// Compute Perlin noise at position p
[NoDiscontinuity]
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
[NoDiscontinuity]
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
void start(float2 x, DiffTensorView<float> p, DiffTensorView<float> res, int thread_idx) {
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
        
        [Disc]
        if (test > 0.0) {
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
)
{
    uint3 dispatch_id = cudaBlockIdx() * cudaBlockDim() + cudaThreadIdx();
    int X_NUM = x.size(0);
    int i = dispatch_id.x;
    if (i >= X_NUM) return;
    float2 x_ = float2(x[i,0], x[i,1]);
    start(x_, p, r, i);
}
