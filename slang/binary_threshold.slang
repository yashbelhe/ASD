import utils;

[Differentiable]
float sample_grid(float2 uv, DiffTensorView<float> p, int grid_size) {
    float2 coord = clamp(uv, float2(0.0), float2(1.0)) * float(grid_size - 1);
    int2 lower = int2(int(floor(coord.x)), int(floor(coord.y)));
    int2 upper = int2(lower + 1);
    lower = clamp(lower, int2(0, 0), int2(grid_size - 1, grid_size - 1));
    upper = clamp(upper, int2(0, 0), int2(grid_size - 1, grid_size - 1));

    float2 frac = coord - float2(lower);

    int base = 1; // first entry stores grid size
    float v00 = p[lower.y * grid_size + lower.x + base];
    float v10 = p[lower.y * grid_size + upper.x + base];
    float v01 = p[upper.y * grid_size + lower.x + base];
    float v11 = p[upper.y * grid_size + upper.x + base];

    float vx0 = mix(v00, v10, frac.x);
    float vx1 = mix(v01, v11, frac.x);
    return mix(vx0, vx1, frac.y);
}

[Differentiable]
void start(float2 x, DiffTensorView<float> p, DiffTensorView<float> res, int thread_idx) {
    int grid_size = int(p[0]);
    float test = sample_grid(x, p, grid_size);
    res[thread_idx] = 0.0;

    [Disc]
    if (test > 0.0) {
        res[thread_idx] = 1.0;
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
    r[i] = 0.0;
    float2 x_ = float2(x[i,0], x[i,1]);
    start(x_, p, r, i);
}
