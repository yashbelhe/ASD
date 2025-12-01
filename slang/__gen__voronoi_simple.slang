import utils;

import utils;

// modified from the second shader at: https://thebookofshaders.com/12/
[Differentiable]
void start(float2 x, DiffTensorView<float> p, DiffTensorView<float> res, int thread_idx, int if_off, int out_off, bool ret_const, bool ret_impl, DiffTensorView<float> impl_fn, TensorView<int> impl_idx, TensorView<int> out_idx, bool should_force, bool force_value) {
    float m_dist = 10.;  // minimum distance

    [MaxIters(1000)]
    for (int i = 0; i < p.size(0) / 5; i++) {
        float2 point = float2(load(p, i*5 + 0), load(p, i*5 + 1));

        // Vector between the pixel and the point
        float2 diff = x - point;

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
