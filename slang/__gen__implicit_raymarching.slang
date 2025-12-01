import utils;

import utils;

static const int MAX_STEPS = 500;
static const float MAX_DIST = 100.0;
static const float EPSILON = 0.001;

[Differentiable]
float trilinear(float3 p, float3 gridSize, DiffTensorView<float> values) {
    float3 pos = p;
    if (any(pos < float3(0.0)) || any(pos >= float3(1.0))) {
        return -10.0;
    }
    pos = clamp(pos, float3(0.0), float3(1.0));

    float3 indices = pos * (gridSize - float3(1.0));
    int3 i0 = int3(floor(indices));
    float3 f = indices - float3(i0);
    int n = int(gridSize.x);

    int i000 = i0.x + i0.y * n + i0.z * n * n;
    int i001 = i0.x + i0.y * n + min(i0.z + 1, n - 1) * n * n;
    int i010 = i0.x + min(i0.y + 1, n - 1) * n + i0.z * n * n;
    int i011 = i0.x + min(i0.y + 1, n - 1) * n + min(i0.z + 1, n - 1) * n * n;
    int i100 = min(i0.x + 1, n - 1) + i0.y * n + i0.z * n * n;
    int i101 = min(i0.x + 1, n - 1) + i0.y * n + min(i0.z + 1, n - 1) * n * n;
    int i110 = min(i0.x + 1, n - 1) + min(i0.y + 1, n - 1) * n + i0.z * n * n;
    int i111 = min(i0.x + 1, n - 1) + min(i0.y + 1, n - 1) * n + min(i0.z + 1, n - 1) * n * n;

    float v000 = load(values, i000 + 2);
    float v001 = load(values, i001 + 2);
    float v010 = load(values, i010 + 2);
    float v011 = load(values, i011 + 2);
    float v100 = load(values, i100 + 2);
    float v101 = load(values, i101 + 2);
    float v110 = load(values, i110 + 2);
    float v111 = load(values, i111 + 2);

    float v00 = mix(v000, v100, f.x);
    float v01 = mix(v001, v101, f.x);
    float v10 = mix(v010, v110, f.x);
    float v11 = mix(v011, v111, f.x);
    float v0 = mix(v00, v10, f.y);
    float v1 = mix(v01, v11, f.y);
    return mix(v0, v1, f.z);
}

[Differentiable]
[PreferRecompute]
float3 calculateNormal(float3 p, float3 gridSize, DiffTensorView<float> values, float h) {
    float3 normal;
    normal.x = trilinear(p + float3(h, 0, 0), gridSize, values) - trilinear(p - float3(h, 0, 0), gridSize, values);
    normal.y = trilinear(p + float3(0, h, 0), gridSize, values) - trilinear(p - float3(0, h, 0), gridSize, values);
    normal.z = trilinear(p + float3(0, 0, h), gridSize, values) - trilinear(p - float3(0, 0, h), gridSize, values);
    return normalize(normal);
}

[Differentiable]
[PreferRecompute]
float implicitIndicator(float3 p, float3 gridSize, DiffTensorView<float> values) {
    return trilinear(p, gridSize, values);
}

[Differentiable]
void start(float2 uv, DiffTensorView<float> p, DiffTensorView<float> res, int thread_idx, int if_off, int out_off, bool ret_const, bool ret_impl, DiffTensorView<float> impl_fn, TensorView<int> impl_idx, TensorView<int> out_idx, bool should_force, bool force_value) {
    res[thread_idx, 0] = 0.0;
    res[thread_idx, 1] = 0.0;
    res[thread_idx, 2] = 0.0;

    uv = float2(uv.x * 2.0 - 1.0, uv.y * 2.0 - 1.0);
    int n = int(load(p, 0));
    float threshold = load(p, 1);
    float3 gridSize = float3(float(n), float(n), float(n));

    int camIdx = 2 + n * n * n;
    float3 rayOrigin = float3(load(p, camIdx + 0), load(p, camIdx + 1), load(p, camIdx + 2));
    float3 lookAt = float3(load(p, camIdx + 3), load(p, camIdx + 4), load(p, camIdx + 5));
    float fov = load(p, camIdx + 6);
    float stepSize = load(p, camIdx + 7);

    int lightIdx = camIdx + 8;
    float3 lightDir = normalize(float3(load(p, lightIdx + 0), load(p, lightIdx + 1), load(p, lightIdx + 2)));
    float3 lightColor = float3(load(p, lightIdx + 3), load(p, lightIdx + 4), load(p, lightIdx + 5));

    int matIdx = lightIdx + 6;
    float3 diffuseColor = float3(load(p, matIdx + 0), load(p, matIdx + 1), load(p, matIdx + 2));

    float3 forward = normalize(lookAt - rayOrigin);
    float3 right = normalize(cross(float3(0.0, 1.0, 0.0), forward));
    float3 up = cross(forward, right);
    float3 rayDir = normalize(forward + right * uv.x * fov + up * uv.y * fov);

    float dist = 0.0;
    for (int step = 0; step < MAX_STEPS; ++step) {
        float3 point = rayOrigin + rayDir * dist;
        float val = implicitIndicator(point, gridSize, p)
            - threshold;
        
        
        // ---- Start: implicit function test ----
        if (ret_impl && abs(val) < abs(impl_fn[thread_idx])) {
            impl_idx[thread_idx] = if_off + step*1 + 0;
            impl_fn[thread_idx] = val;
        }
        // ---- End: implicit function test ----

        if (should_force && load(impl_idx, thread_idx) == (if_off + step*1 + 0) ? force_value : val > 0.0) {
                           if(ret_const) {
                               out_idx[thread_idx] = hash(out_idx[thread_idx], (if_off + step*1 + 0)*2);
                           }

            float3 normal = calculateNormal(point, gridSize, p, EPSILON);
            float diffuse = max(dot(normal, lightDir), 0.0);
            float3 color = diffuseColor * diffuse * lightColor;
            res[thread_idx, 0] = color.x;
            res[thread_idx, 1] = color.y;
            res[thread_idx, 2] = color.z;
            return;
        }
        dist += stepSize;
        if (dist > MAX_DIST) {
            break;
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
    r[i, 0] = 0.0;
    r[i, 1] = 0.0;
    r[i, 2] = 0.0;
    float2 uv = float2(x[i,0], x[i,1]);
    start(uv, p, r, i, if_off + 0, out_off + 0, ret_const, ret_impl, impl_fn, impl_idx, out_idx, should_force, force_value);
}
