[Differentiable]
void start(float2 x, DiffTensorView<float> p, DiffTensorView<float> res, int thread_idx) {
    float2 v0 = float2(p[0], p[1]);
    float2 v1 = float2(p[2], p[3]);
    float2 v2 = float2(p[4], p[5]);

    float2 e0 = v1 - v0;
    float2 e1 = v2 - v1;
    float2 e2 = v0 - v2;
    float2 p0 = x - v0;
    float2 p1 = x - v1;
    float2 p2 = x - v2;

    float c0 = e0.x * p0.y - e0.y * p0.x;
    float c1 = e1.x * p1.y - e1.y * p1.x;
    float c2 = e2.x * p2.y - e2.y * p2.x;

    [Disc]
    if (c0 >= 0.0) {
        [Disc]
        if (c1 >= 0.0) {
            [Disc]
            if (c2 >= 0.0) {
                res[thread_idx] = 1.0;
            } else {
                res[thread_idx] = 0.0;
            }
        } else {
            res[thread_idx] = 0.0;
        }
    } else {
        [Disc]
        if (c1 <= 0.0) {
            [Disc]
            if (c2 <= 0.0) {
                res[thread_idx] = 1.0;
            } else {
                res[thread_idx] = 0.0;
            }
        } else {
            res[thread_idx] = 0.0;
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
    r[i] = 0.0;
    float2 x_ = float2(x[i,0], x[i,1]);
    start(x_, p, r, i);
}
