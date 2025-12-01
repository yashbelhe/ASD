import utils;

[Differentiable]
float sdSphere(float3 x, float radius) {
    return length(x) - radius;
}

[Differentiable]
float sdCube(float3 x, float size) {
    float3 d = abs(x) - size;
    return length(max(d, 0.0)) + min(max(d.x, max(d.y, d.z)), 0.0);
}

[Differentiable]
float sdCylinderZ(float3 x, float radius, float height) {
    float2 d = abs(float2(length(x.xy), x.z)) - float2(radius,height);
    return min(max(d.x, d.y), 0.0) + length(max(d, 0.0));
}

[Differentiable]
float sdCylinderX(float3 x, float radius, float height) {
    float2 d = abs(float2(length(x.yz), x.x)) - float2(radius,height);
    return min(max(d.x, d.y), 0.0) + length(max(d, 0.0));
}

[Differentiable]
float sdCylinderY(float3 x, float radius, float height) {
    float2 d = abs(float2(length(x.xz), x.y)) - float2(radius,height);
    return min(max(d.x, d.y), 0.0) + length(max(d, 0.0));
}

[Differentiable]
void start(float3 x, DiffTensorView<float> p, DiffTensorView<float> res, int thread_idx, int if_off, int out_off, bool ret_const, bool ret_impl, DiffTensorView<float> impl_fn, TensorView<int> impl_idx, TensorView<int> out_idx, bool should_force, bool force_value) {
    float sphere_radius = p[0];
    float3 sphere_position = float3(p[1], p[2], p[3]);
    float cube_size = p[4];
    float3 cube_position = float3(p[5], p[6], p[7]);
    float cylinder_z_radius = p[8];
    float3 cylinder_z_position = float3(p[9], p[10], p[11]);
    float cylinder_x_radius = p[12];
    float3 cylinder_x_position = float3(p[13], p[14], p[15]);
    float cylinder_y_radius = p[16];
    float3 cylinder_y_position = float3(p[17], p[18], p[19]);

    x = 2 * x - 1; // Normalize to [-1, 1]

#ifdef PRIM_VIS
    #if PRIM_TYPE == 0
        float test_sphere = -sdSphere(x - sphere_position, sphere_radius);
        
        
        // ---- Start: implicit function test ----
        if (ret_impl && abs(test_sphere) < abs(impl_fn[thread_idx])) {
            impl_idx[thread_idx] = if_off + 0;
            impl_fn[thread_idx] = test_sphere;
        }
        // ---- End: implicit function test ----

        if (should_force && load(impl_idx, thread_idx) == (if_off + 0) ? force_value : test_sphere > 0.0) {
                out_idx[thread_idx] = hash(out_idx[thread_idx], (if_off + 0)*2);
            res[thread_idx] = 1.0;
            return;
        }
    #elif PRIM_TYPE == 1
        float test_cube = -sdCube(x - cube_position, cube_size);
        
        
        // ---- Start: implicit function test ----
        if (ret_impl && abs(test_cube) < abs(impl_fn[thread_idx])) {
            impl_idx[thread_idx] = if_off + 1;
            impl_fn[thread_idx] = test_cube;
        }
        // ---- End: implicit function test ----

        if (should_force && load(impl_idx, thread_idx) == (if_off + 1) ? force_value : test_cube > 0.0) {
                out_idx[thread_idx] = hash(out_idx[thread_idx], (if_off + 1)*2);
            res[thread_idx] = 1.0;
            return;
        }
    #elif PRIM_TYPE == 2
        float test_cylinder_z = -sdCylinderZ(x - cylinder_z_position, cylinder_z_radius, 0.99);
        
        
        // ---- Start: implicit function test ----
        if (ret_impl && abs(test_cylinder_z) < abs(impl_fn[thread_idx])) {
            impl_idx[thread_idx] = if_off + 2;
            impl_fn[thread_idx] = test_cylinder_z;
        }
        // ---- End: implicit function test ----

        if (should_force && load(impl_idx, thread_idx) == (if_off + 2) ? force_value : test_cylinder_z > 0.0) {
                out_idx[thread_idx] = hash(out_idx[thread_idx], (if_off + 2)*2);
            res[thread_idx] = 1.0;
            return;
        }
    #elif PRIM_TYPE == 3
        float test_cylinder_x = -sdCylinderX(x - cylinder_x_position, cylinder_x_radius, 0.99);
        
        
        // ---- Start: implicit function test ----
        if (ret_impl && abs(test_cylinder_x) < abs(impl_fn[thread_idx])) {
            impl_idx[thread_idx] = if_off + 3;
            impl_fn[thread_idx] = test_cylinder_x;
        }
        // ---- End: implicit function test ----

        if (should_force && load(impl_idx, thread_idx) == (if_off + 3) ? force_value : test_cylinder_x > 0.0) {
                out_idx[thread_idx] = hash(out_idx[thread_idx], (if_off + 3)*2);
            res[thread_idx] = 1.0;
            return;
        }
    #elif PRIM_TYPE == 4
        float test_cylinder_y = -sdCylinderY(x - cylinder_y_position, cylinder_y_radius, 0.99);
        
        
        // ---- Start: implicit function test ----
        if (ret_impl && abs(test_cylinder_y) < abs(impl_fn[thread_idx])) {
            impl_idx[thread_idx] = if_off + 4;
            impl_fn[thread_idx] = test_cylinder_y;
        }
        // ---- End: implicit function test ----

        if (should_force && load(impl_idx, thread_idx) == (if_off + 4) ? force_value : test_cylinder_y > 0.0) {
                out_idx[thread_idx] = hash(out_idx[thread_idx], (if_off + 4)*2);
            res[thread_idx] = 1.0;
            return;
        }
    #elif PRIM_TYPE == 5
        float test_cylinder_z = -sdCylinderZ(x - cylinder_z_position, cylinder_z_radius, 0.99);
        float test_cylinder_x = -sdCylinderX(x - cylinder_x_position, cylinder_x_radius, 0.99);
        
        
        // ---- Start: implicit function test ----
        if (ret_impl && abs(test_cylinder_z) < abs(impl_fn[thread_idx])) {
            impl_idx[thread_idx] = if_off + 5;
            impl_fn[thread_idx] = test_cylinder_z;
        }
        // ---- End: implicit function test ----

        if (should_force && load(impl_idx, thread_idx) == (if_off + 5) ? force_value : test_cylinder_z > 0.0) {
                out_idx[thread_idx] = hash(out_idx[thread_idx], (if_off + 5)*2);
            res[thread_idx] = 1.0;
            return;
        }
        
        
        // ---- Start: implicit function test ----
        if (ret_impl && abs(test_cylinder_x) < abs(impl_fn[thread_idx])) {
            impl_idx[thread_idx] = if_off + 6;
            impl_fn[thread_idx] = test_cylinder_x;
        }
        // ---- End: implicit function test ----

        if (should_force && load(impl_idx, thread_idx) == (if_off + 6) ? force_value : test_cylinder_x > 0.0) {
                out_idx[thread_idx] = hash(out_idx[thread_idx], (if_off + 6)*2);
            res[thread_idx] = 1.0;
            return;
        }
    #elif PRIM_TYPE == 6
        float test_cylinder_z = -sdCylinderZ(x - cylinder_z_position, cylinder_z_radius, 0.99);
        float test_cylinder_y = -sdCylinderY(x - cylinder_y_position, cylinder_y_radius, 0.99);
        float test_cylinder_x = -sdCylinderX(x - cylinder_x_position, cylinder_x_radius, 0.99);
        
        
        // ---- Start: implicit function test ----
        if (ret_impl && abs(test_cylinder_z) < abs(impl_fn[thread_idx])) {
            impl_idx[thread_idx] = if_off + 7;
            impl_fn[thread_idx] = test_cylinder_z;
        }
        // ---- End: implicit function test ----

        if (should_force && load(impl_idx, thread_idx) == (if_off + 7) ? force_value : test_cylinder_z > 0.0) {
                out_idx[thread_idx] = hash(out_idx[thread_idx], (if_off + 7)*2);
            res[thread_idx] = 1.0;
            return;
        }
        
        
        // ---- Start: implicit function test ----
        if (ret_impl && abs(test_cylinder_y) < abs(impl_fn[thread_idx])) {
            impl_idx[thread_idx] = if_off + 8;
            impl_fn[thread_idx] = test_cylinder_y;
        }
        // ---- End: implicit function test ----

        if (should_force && load(impl_idx, thread_idx) == (if_off + 8) ? force_value : test_cylinder_y > 0.0) {
                out_idx[thread_idx] = hash(out_idx[thread_idx], (if_off + 8)*2);
            res[thread_idx] = 1.0;
            return;
        }
        
        
        // ---- Start: implicit function test ----
        if (ret_impl && abs(test_cylinder_x) < abs(impl_fn[thread_idx])) {
            impl_idx[thread_idx] = if_off + 9;
            impl_fn[thread_idx] = test_cylinder_x;
        }
        // ---- End: implicit function test ----

        if (should_force && load(impl_idx, thread_idx) == (if_off + 9) ? force_value : test_cylinder_x > 0.0) {
                out_idx[thread_idx] = hash(out_idx[thread_idx], (if_off + 9)*2);
            res[thread_idx] = 1.0;
            return;
        }
    #elif PRIM_TYPE == 7
        float test_sphere = -sdSphere(x - sphere_position, sphere_radius);
        float test_cube = -sdCube(x - cube_position, cube_size);
        
        
        // ---- Start: implicit function test ----
        if (ret_impl && abs(test_sphere) < abs(impl_fn[thread_idx])) {
            impl_idx[thread_idx] = if_off + 10;
            impl_fn[thread_idx] = test_sphere;
        }
        // ---- End: implicit function test ----

        if (should_force && load(impl_idx, thread_idx) == (if_off + 10) ? force_value : test_sphere > 0.0) {
            out_idx[thread_idx] = hash(out_idx[thread_idx], (if_off + 10)*2);
            
            
            // ---- Start: implicit function test ----
            if (ret_impl && abs(test_cube) < abs(impl_fn[thread_idx])) {
                impl_idx[thread_idx] = if_off + 11;
                impl_fn[thread_idx] = test_cube;
            }
            // ---- End: implicit function test ----

            if (should_force && load(impl_idx, thread_idx) == (if_off + 11) ? force_value : test_cube > 0.0) {
                    out_idx[thread_idx] = hash(out_idx[thread_idx], (if_off + 11)*2);
                res[thread_idx] = 1.0;
                return;
            }
        }
    #endif
#else
    float test_sphere = -sdSphere(x - sphere_position, sphere_radius);
    float test_cube = -sdCube(x - cube_position, cube_size);
    float test_cylinder_z = sdCylinderZ(x - cylinder_z_position, cylinder_z_radius, 10.0);
    float test_cylinder_x = sdCylinderX(x - cylinder_x_position, cylinder_x_radius, 10.0);
    float test_cylinder_y = sdCylinderY(x - cylinder_y_position, cylinder_y_radius, 10.0);

    
    
    // ---- Start: implicit function test ----
    if (ret_impl && abs(test_sphere) < abs(impl_fn[thread_idx])) {
        impl_idx[thread_idx] = if_off + 12;
        impl_fn[thread_idx] = test_sphere;
    }
    // ---- End: implicit function test ----

    if (should_force && load(impl_idx, thread_idx) == (if_off + 12) ? force_value : test_sphere > 0.0) {
        out_idx[thread_idx] = hash(out_idx[thread_idx], (if_off + 12)*2);
        
        
        // ---- Start: implicit function test ----
        if (ret_impl && abs(test_cube) < abs(impl_fn[thread_idx])) {
            impl_idx[thread_idx] = if_off + 13;
            impl_fn[thread_idx] = test_cube;
        }
        // ---- End: implicit function test ----

        if (should_force && load(impl_idx, thread_idx) == (if_off + 13) ? force_value : test_cube > 0.0) {
            out_idx[thread_idx] = hash(out_idx[thread_idx], (if_off + 13)*2);
            
            
            // ---- Start: implicit function test ----
            if (ret_impl && abs(test_cylinder_z) < abs(impl_fn[thread_idx])) {
                impl_idx[thread_idx] = if_off + 14;
                impl_fn[thread_idx] = test_cylinder_z;
            }
            // ---- End: implicit function test ----

            if (should_force && load(impl_idx, thread_idx) == (if_off + 14) ? force_value : test_cylinder_z > 0.0) {
                out_idx[thread_idx] = hash(out_idx[thread_idx], (if_off + 14)*2);
                
                
                // ---- Start: implicit function test ----
                if (ret_impl && abs(test_cylinder_x) < abs(impl_fn[thread_idx])) {
                    impl_idx[thread_idx] = if_off + 15;
                    impl_fn[thread_idx] = test_cylinder_x;
                }
                // ---- End: implicit function test ----

                if (should_force && load(impl_idx, thread_idx) == (if_off + 15) ? force_value : test_cylinder_x > 0.0) {
                    out_idx[thread_idx] = hash(out_idx[thread_idx], (if_off + 15)*2);
                    
                    
                    // ---- Start: implicit function test ----
                    if (ret_impl && abs(test_cylinder_y) < abs(impl_fn[thread_idx])) {
                        impl_idx[thread_idx] = if_off + 16;
                        impl_fn[thread_idx] = test_cylinder_y;
                    }
                    // ---- End: implicit function test ----

                    if (should_force && load(impl_idx, thread_idx) == (if_off + 16) ? force_value : test_cylinder_y > 0.0) {
                            out_idx[thread_idx] = hash(out_idx[thread_idx], (if_off + 16)*2);
                        res[thread_idx] = 1.0;
                        return;
                    }
                }
            }
        }
    }
#endif
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
    float3 x_ = float3(x[i,0], x[i,1], x[i,2]);
    start(x_, p, r, i, if_off + 0, out_off + 0, ret_const, ret_impl, impl_fn, impl_idx, out_idx, should_force, force_value);
} 