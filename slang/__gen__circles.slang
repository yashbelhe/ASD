import utils;

[Differentiable]
void start(float2 x, DiffTensorView<float> p, DiffTensorView<float> res, int thread_idx, int if_off, int out_off, bool ret_const, bool ret_impl, DiffTensorView<float> impl_fn, TensorView<int> impl_idx, TensorView<int> out_idx, bool should_force, bool force_value) {
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
    r[i] = 0.0;
    float2 x_ = float2(x[i,0], x[i,1]);
    start(x_, p, r, i, if_off + 0, out_off + 0, ret_const, ret_impl, impl_fn, impl_idx, out_idx, should_force, force_value);
}
