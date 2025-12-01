import utils;

[Differentiable]
void start(float2 x, DiffTensorView<float> p, DiffTensorView<float> res, int thread_idx, int if_off, int out_off, bool ret_const, bool ret_impl, DiffTensorView<float> impl_fn, TensorView<int> impl_idx, TensorView<int> out_idx, bool should_force, bool force_value) {
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

    
    
    // ---- Start: implicit function test ----
    if (ret_impl && abs(c0) < abs(impl_fn[thread_idx])) {
        impl_idx[thread_idx] = if_off + 0;
        impl_fn[thread_idx] = c0;
    }
    // ---- End: implicit function test ----

    if (should_force && load(impl_idx, thread_idx) == (if_off + 0) ? force_value : c0 > 0.0) {
                       if(ret_const) {
                           out_idx[thread_idx] = hash(out_idx[thread_idx], (if_off + 0)*2);
                       }

        
        
        // ---- Start: implicit function test ----
        if (ret_impl && abs(c1) < abs(impl_fn[thread_idx])) {
            impl_idx[thread_idx] = if_off + 1;
            impl_fn[thread_idx] = c1;
        }
        // ---- End: implicit function test ----

        if (should_force && load(impl_idx, thread_idx) == (if_off + 1) ? force_value : c1 > 0.0) {
                           if(ret_const) {
                               out_idx[thread_idx] = hash(out_idx[thread_idx], (if_off + 1)*2);
                           }

            
            
            // ---- Start: implicit function test ----
            if (ret_impl && abs(c2) < abs(impl_fn[thread_idx])) {
                impl_idx[thread_idx] = if_off + 2;
                impl_fn[thread_idx] = c2;
            }
            // ---- End: implicit function test ----

            if (should_force && load(impl_idx, thread_idx) == (if_off + 2) ? force_value : c2 > 0.0) {
                               if(ret_const) {
                                   out_idx[thread_idx] = hash(out_idx[thread_idx], (if_off + 2)*2);
                               }

                res[thread_idx] = 1.0;
            } else {
                       if(ret_const) {
                           out_idx[thread_idx] = hash(out_idx[thread_idx], (if_off + 2)*2 + 1);
                       }

                res[thread_idx] = 0.0;
            }
        } else {
                   if(ret_const) {
                       out_idx[thread_idx] = hash(out_idx[thread_idx], (if_off + 2)*2 + 1);
                   }

            res[thread_idx] = 0.0;
        }
    } else {
               if(ret_const) {
                   out_idx[thread_idx] = hash(out_idx[thread_idx], (if_off + 2)*2 + 1);
               }

        
        
        // ---- Start: implicit function test ----
        if (ret_impl && abs(c1) < abs(impl_fn[thread_idx])) {
            impl_idx[thread_idx] = if_off + 3;
            impl_fn[thread_idx] = c1;
        }
        // ---- End: implicit function test ----

        if (should_force && load(impl_idx, thread_idx) == (if_off + 3) ? force_value : c1 > 0.0) {
                           if(ret_const) {
                               out_idx[thread_idx] = hash(out_idx[thread_idx], (if_off + 3)*2);
                           }

            
            
            // ---- Start: implicit function test ----
            if (ret_impl && abs(c2) < abs(impl_fn[thread_idx])) {
                impl_idx[thread_idx] = if_off + 4;
                impl_fn[thread_idx] = c2;
            }
            // ---- End: implicit function test ----

            if (should_force && load(impl_idx, thread_idx) == (if_off + 4) ? force_value : c2 > 0.0) {
                               if(ret_const) {
                                   out_idx[thread_idx] = hash(out_idx[thread_idx], (if_off + 4)*2);
                               }

                res[thread_idx] = 1.0;
            } else {
                       if(ret_const) {
                           out_idx[thread_idx] = hash(out_idx[thread_idx], (if_off + 4)*2 + 1);
                       }

                res[thread_idx] = 0.0;
            }
        } else {
                   if(ret_const) {
                       out_idx[thread_idx] = hash(out_idx[thread_idx], (if_off + 4)*2 + 1);
                   }

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
