import utils;

import utils;

static const int MAX_STEPS = 500;

[PreferRecompute]
[Differentiable]
float3 mix(float3 a, float3 b, float t) {
    return a + t * (b - a);
}

[PreferRecompute]
[Differentiable]
float mix(float a, float b, float t) {
    return a + t * (b - a);
}

[PreferRecompute]
[Differentiable]
float sdSphere(float3 p, float3 center, float radius) {
    return length(p - center) - radius;
}

[PreferRecompute]
[Differentiable]
bool shade_spheres(float step_size, float3 light_pos, float3 point, float light_intensity, DiffTensorView<float> p, DiffTensorView<float> res, int thread_idx, int if_off, int out_off, bool ret_const, bool ret_impl, DiffTensorView<float> impl_fn, TensorView<int> impl_idx, TensorView<int> out_idx, bool should_force, bool force_value, int thread_idx) {
    int idx = 1;
    float3 center = float3(load(p, idx + 0), load(p, idx + 1), load(p, idx + 2));
    float radius = load(p, idx + 3);
    float3 sphere_color = float3(load(p, idx + 4), load(p, idx + 5), load(p, idx + 6));

    float dist = sdSphere(point, center, radius);
    float test = step_size - dist;
    
    
    // ---- Start: implicit function test ----
    if (ret_impl && abs(test) < abs(impl_fn[thread_idx])) {
        impl_idx[thread_idx] = if_off + 0;
        impl_fn[thread_idx] = test;
    }
    // ---- End: implicit function test ----

    if (should_force && load(impl_idx, thread_idx) == (if_off + 0) ? force_value : test > 0.0) {
                        if(ret_const) {
                            out_idx[thread_idx] = hash(out_idx[thread_idx], (if_off + 0)*2);
                        }

        float3 to_light = normalize(light_pos - point);
        float3 normal = normalize(point - center);
        float diffuse = max(dot(normal, to_light), 0.0);
        float3 color = sphere_color * light_intensity * diffuse;
        res[thread_idx, 0] = color.x;
        res[thread_idx, 1] = color.y;
        res[thread_idx, 2] = color.z;
        return true;
    }
    return false;
}

[Differentiable]
void start(float2 uv, DiffTensorView<float> p, DiffTensorView<float> res, int thread_idx, int if_off, int out_off, bool ret_const, bool ret_impl, DiffTensorView<float> impl_fn, TensorView<int> impl_idx, TensorView<int> out_idx, bool should_force, bool force_value) {
    res[thread_idx, 0] = 0.0;
    res[thread_idx, 1] = 0.0;
    res[thread_idx, 2] = 0.0;

    uv = float2(uv.x * 2.0 - 1.0, uv.y * 2.0 - 1.0);

    int num_spheres = int(load(p, 0));
    int cam_idx = 12 + num_spheres * 7;
    float3 ray_origin = float3(load(p, cam_idx + 0), load(p, cam_idx + 1), load(p, cam_idx + 2));
    float3 look_at = float3(load(p, cam_idx + 3), load(p, cam_idx + 4), load(p, cam_idx + 5));
    float fov = load(p, cam_idx + 6);
    float step_size = load(p, cam_idx + 7);

    int light_idx = 8 + num_spheres * 7;
    float3 light_pos = float3(load(p, light_idx + 0), load(p, light_idx + 1), load(p, light_idx + 2));
    float light_intensity = load(p, light_idx + 3);

    float3 forward = normalize(look_at - ray_origin);
    float3 right = normalize(cross(float3(0.0, 1.0, 0.0), forward));
    float3 up = cross(forward, right);
    float3 ray_dir = normalize(forward + right * uv.x * fov + up * uv.y * fov);

    [MaxIters(500)]
    for (int i = 0; i < MAX_STEPS; ++i) {
        float3 point = ray_origin + ray_dir * (i * step_size);
        if (shade_spheres(step_size, light_pos, point, light_intensity, p, res, thread_idx, if_off + i*1 + 0, out_off + i*3 + 3, ret_const, ret_impl, impl_fn, impl_idx, out_idx, should_force, force_value, thread_idx)) {
            return;
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
    float2 uv = float2(x[i,0], x[i,1]);
    start(uv, p, r, i, if_off + 0, out_off + 0, ret_const, ret_impl, impl_fn, impl_idx, out_idx, should_force, force_value);
}
