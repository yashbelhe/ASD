import utils;

// modified from the second shader at: https://thebookofshaders.com/12/
[Differentiable]
void start(float2 x, DiffTensorView<float> p, DiffTensorView<float> res, int thread_idx) {
    float m_dist = 10.;  // minimum distance

    [MaxIters(1000)]
    for (int i = 0; i < p.size(0) / 5; i++) {
        float2 point = float2(load(p, i*5 + 0), load(p, i*5 + 1));

        // Vector between the pixel and the point
        float2 diff = x - point;

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
