import utils;

[Differentiable]
[NoDiscontinuity]
float bilinearInterpolation(float2 uv, DiffTensorView<float> grid, int gridSize) {
    float2 gridCoord = uv * float(gridSize - 1);
    int2 gridLower = int2(int(floor(gridCoord.x)), int(floor(gridCoord.y)));
    int2 gridUpper = int2(int(ceil(gridCoord.x)), int(ceil(gridCoord.y)));
    gridLower = clamp(gridLower, int2(0, 0), int2(gridSize - 1, gridSize - 1));
    gridUpper = clamp(gridUpper, int2(0, 0), int2(gridSize - 1, gridSize - 1));
    float2 frac = gridCoord - float2(gridLower);

    float v00 = grid[gridLower.y * gridSize + gridLower.x + 3];
    float v10 = grid[gridLower.y * gridSize + gridUpper.x + 3];
    float v01 = grid[gridUpper.y * gridSize + gridLower.x + 3];
    float v11 = grid[gridUpper.y * gridSize + gridUpper.x + 3];

    float v0 = mix(v00, v10, frac.x);
    float v1 = mix(v01, v11, frac.x);
    return mix(v0, v1, frac.y);
}

static const float WINDOW_SIZE = 0.1;
static const float M_PI = 3.14159265358979323846;

[Differentiable]
void start(float2 x, DiffTensorView<float> p, DiffTensorView<float> res, int thread_idx) {
    int gridSize = int(p[0]);
    int start_step = int(p[1]);
    int n_steps = int(p[2]);

    float t_interval = 0.9 / float(100 - 1);
    float t_start = 0.08;

#ifdef SINGLE_STEP
    int i = 0;
#else
    [MaxIters(100)]
    for (int i = 0; i < n_steps; i++) {
#endif
        float2 uv;
        if (start_step < 0) {
            uv = x;
        } else {
            float t = t_start + float(start_step + i) * t_interval;
            float angle = 2.0 * M_PI * t;
            float a = atan2(
                4 * M_PI * sin(angle) - angle * sin(angle) + cos(angle),
                4 * M_PI * cos(angle) - angle * cos(angle) - sin(angle)
            );

            float2x2 rot = float2x2(cos(a), -sin(a),
                                    sin(a), cos(a));
            float2x2 rot_inv = transpose(rot);
            float2 transl = float2(
                0.5 + 0.35 * (1 - angle / (4 * M_PI)) * sin(angle),
                0.5 - 0.35 * (1 - angle / (4 * M_PI)) * cos(angle)
            );
            float2 x_inv_trans = mul(rot_inv, x - transl);
            uv = 0.5 * (x_inv_trans / WINDOW_SIZE) + 0.5;
        }

        float test = bilinearInterpolation(uv, p, gridSize);
        if (any(uv < 0.0) || any(uv > 1.0)) {
            test = 0.0;
        }

        [Disc]
        if (test > 0.0) {
            res[thread_idx] = 1.0;
            return;
        }
#ifndef SINGLE_STEP
    }
#endif
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
