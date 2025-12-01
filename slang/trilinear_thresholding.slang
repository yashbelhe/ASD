import utils;

[Differentiable]
[NoDiscontinuity]
float trilinearInterpolation(float3 uvw, DiffTensorView<float> grid, int gridSize) {
    float3 gridCoord = uvw * float(gridSize - 1);
    int3 gridLower = int3(int(floor(gridCoord.x)), int(floor(gridCoord.y)), int(floor(gridCoord.z)));
    int3 gridUpper = int3(int(ceil(gridCoord.x)), int(ceil(gridCoord.y)), int(ceil(gridCoord.z)));
    gridLower = clamp(gridLower, int3(0, 0, 0), int3(gridSize - 1, gridSize - 1, gridSize - 1));
    gridUpper = clamp(gridUpper, int3(0, 0, 0), int3(gridSize - 1, gridSize - 1, gridSize - 1));
    float3 frac = gridCoord - float3(gridLower);

    int base = 1;
    int stride = gridSize * gridSize;
    float v000 = grid[gridLower.z * stride + gridLower.y * gridSize + gridLower.x + base];
    float v100 = grid[gridLower.z * stride + gridLower.y * gridSize + gridUpper.x + base];
    float v010 = grid[gridLower.z * stride + gridUpper.y * gridSize + gridLower.x + base];
    float v110 = grid[gridLower.z * stride + gridUpper.y * gridSize + gridUpper.x + base];
    float v001 = grid[gridUpper.z * stride + gridLower.y * gridSize + gridLower.x + base];
    float v101 = grid[gridUpper.z * stride + gridLower.y * gridSize + gridUpper.x + base];
    float v011 = grid[gridUpper.z * stride + gridUpper.y * gridSize + gridLower.x + base];
    float v111 = grid[gridUpper.z * stride + gridUpper.y * gridSize + gridUpper.x + base];

    float v00 = mix(v000, v100, frac.x);
    float v01 = mix(v010, v110, frac.x);
    float v0 = mix(v00, v01, frac.y);

    float v10 = mix(v001, v101, frac.x);
    float v11 = mix(v011, v111, frac.x);
    float v1 = mix(v10, v11, frac.y);

    return mix(v0, v1, frac.z);
}

[Differentiable]
void start(float3 x, DiffTensorView<float> p, DiffTensorView<float> res, int thread_idx) {
    int gridSize = int(p[0]);
    int gridSizeCube = gridSize * gridSize * gridSize;
    float test = trilinearInterpolation(x, p, gridSize);
    float outside = p[gridSizeCube + 2];
    float inside = p[gridSizeCube + 1];
    res[thread_idx] = outside;

    [Disc]
    if (test > 0.0) {
        res[thread_idx] = inside;
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
    float3 x_ = float3(x[i,0], x[i,1], x[i,2]);
    start(x_, p, r, i);
}
