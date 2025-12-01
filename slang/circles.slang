[Differentiable]
void start(float2 x, DiffTensorView<float> p, DiffTensorView<float> res, int thread_idx) {
    int num_p = p.size(0) / 4;
    float trans = 0.0;
    
    [MaxIters(100)]
    for (int i = 0; i < num_p; i++) {
        float center_x = p[4*i];
        float center_y = p[4*i + 1];
        float radius = p[4*i + 2];
        float opacity = p[4*i + 3];
        float test = (x.x - center_x)*(x.x - center_x) + (x.y - center_y)*(x.y - center_y) - radius*radius;
        // float test = pow(x.x - center_x, 2) + pow(x.y - center_y, 2) - radius*radius;
        [Disc]
        if (test > 0.0) {
            res[thread_idx] += (1.0 - trans) * opacity;
            trans *= (1.0 - opacity);
            trans = min(trans, 1.0);
        }
    }

}

[Differentiable]
[AutoPyBindCUDA]
[CUDAKernel]
void run(
    DiffTensorView<float> x, // Evaluation points,
    DiffTensorView<float> p, // Integrand parameters [num_circles, center_x, center_y, radius, opacity, ...],
    DiffTensorView<float> r // Return value,
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
