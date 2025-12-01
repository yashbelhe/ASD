[Differentiable]
void start(float2 x, DiffTensorView<float> p, DiffTensorView<float> res, int thread_idx) {
    float center_x = p[0];
    float center_y = p[1];
    float radius = p[2];
    float test = (x.x - center_x)*(x.x - center_x) + (x.y - center_y)*(x.y - center_y) - radius*radius;
    [Disc]
    if (test < 0.0) {
        res[thread_idx] = 1.0;
    } else {
        res[thread_idx] = 0.0;
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
