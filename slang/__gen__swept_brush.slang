import utils;

import utils;

static const float WINDOW_SIZE = 0.1;
static const float M_PI = 3.14159265358979323846;

[Differentiable]
void start(float2 x, DiffTensorView<float> p, DiffTensorView<float> res, int thread_idx, int if_off, int out_off, bool ret_const, bool ret_impl, DiffTensorView<float> impl_fn, TensorView<int> impl_idx, TensorView<int> out_idx, bool should_force, bool force_value) {
    // First parameter is the number of bristles
    int n_bristles = int(p[0]);
    // Second parameter is start step
    int start_step = int(p[1]);
    // Third parameter is the number of steps
    int n_steps = int(p[2]);

    float t_start = 0.0;
#ifdef DRAW_PATH
    float t_interval = 2 * 1.0 / float(100 - 1) / 10;
#else
    float t_interval = 2 * 1.0 / float(100 - 1);
#endif

    [MaxIters(100)]
    for (int i = 0; i < n_steps; i++) {
        float2 x_inv_trans;
        if (start_step < 0) {
            // Map from [0, 1] to [-WINDOW_SIZE, WINDOW_SIZE]
            x_inv_trans = WINDOW_SIZE * (2.0 * x - 1);
        } else {
            float t = t_start + float(start_step + i) * t_interval;

            float angle = 2.0 * M_PI * t;
            float a = atan2(
                4 * M_PI * sin(angle) - angle * sin(angle) + cos(angle),
                4 * M_PI * cos(angle) - angle * cos(angle) - sin(angle)
            );

            // Transform point by inverse of bristle transform matrix
            float2x2 rot = float2x2(cos(a), -sin(a),
                                    sin(a), cos(a));
            float2x2 rot_inv = transpose(rot);
            float2 transl = float2(
                0.5 + 0.35 * (1 - angle / (4 * M_PI)) * sin(angle),
                0.5 - 0.35 * (1 - angle / (4 * M_PI)) * cos(angle)
            );
            x_inv_trans = mul(rot_inv, x - transl);
        }

        // Iterate over bristles
        [MaxIters(16)]
        for (int j = 0; j < n_bristles; j++) {
            // Bristle parameters
            float x_offset = p[3 + j * 3];
            float rad_offset = p[3 + j * 3 + 1];
            float b_thickness = p[3 + j * 3 + 2];

#ifdef DRAW_PATH
            float b_length = 2e-3;
#else
            float b_length = 2e-2;
#endif

            float2 b_center = float2(1.5 * x_offset, rad_offset);

            // Ellipse test
            float2 d = (x_inv_trans - b_center) / float2(b_length, b_thickness);
            float test = 1.0 - dot(d, d);

            
            
            // ---- Start: implicit function test ----
            if (ret_impl && abs(test) < abs(impl_fn[thread_idx])) {
                impl_idx[thread_idx] = if_off + i*16 + j*1 + 0;
                impl_fn[thread_idx] = test;
            }
            // ---- End: implicit function test ----

            if (should_force && load(impl_idx, thread_idx) == (if_off + i*16 + j*1 + 0) ? force_value : test > 0.0) {
                                if(ret_const) {
                                    out_idx[thread_idx] = hash(out_idx[thread_idx], (if_off + i*16 + j*1 + 0)*2);
                                }

                // Inside the bristle
                res[thread_idx] = 1.0;
                return;
            }
        }
    }
}

[Differentiable]
[AutoPyBindCUDA]
[CUDAKernel]
void run(
    DiffTensorView<float> x, // Evaluation points
    DiffTensorView<float> p, // Parameters buffer with grid values and threshold outputs
    DiffTensorView<float> r  // Return value
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
    r[i] = 0.0;  // Initialize to 0
    float2 x_ = float2(x[i,0], x[i,1]);
    start(x_, p, r, i, if_off + 0, out_off + 0, ret_const, ret_impl, impl_fn, impl_idx, out_idx, should_force, force_value);
}
