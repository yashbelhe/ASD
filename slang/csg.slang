[Differentiable]
[NoDiscontinuity]
float sdSphere(float3 x, float radius) {
    return length(x) - radius;
}

[Differentiable]
[NoDiscontinuity]
float sdCube(float3 x, float size) {
    float3 d = abs(x) - size;
    return length(max(d, 0.0)) + min(max(d.x, max(d.y, d.z)), 0.0);
}

[Differentiable]
[NoDiscontinuity]
float sdCylinderZ(float3 x, float radius, float height) {
    float2 d = abs(float2(length(x.xy), x.z)) - float2(radius,height);
    return min(max(d.x, d.y), 0.0) + length(max(d, 0.0));
}

[Differentiable]
[NoDiscontinuity]
float sdCylinderX(float3 x, float radius, float height) {
    float2 d = abs(float2(length(x.yz), x.x)) - float2(radius,height);
    return min(max(d.x, d.y), 0.0) + length(max(d, 0.0));
}

[Differentiable]
[NoDiscontinuity]
float sdCylinderY(float3 x, float radius, float height) {
    float2 d = abs(float2(length(x.xz), x.y)) - float2(radius,height);
    return min(max(d.x, d.y), 0.0) + length(max(d, 0.0));
}

[Differentiable]
void start(float3 x, DiffTensorView<float> p, DiffTensorView<float> res, int thread_idx) {
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

    float test_sphere = -sdSphere(x - sphere_position, sphere_radius);
    float test_cube = -sdCube(x - cube_position, cube_size);
    float test_cylinder_z = sdCylinderZ(x - cylinder_z_position, cylinder_z_radius, 10.0);
    float test_cylinder_x = sdCylinderX(x - cylinder_x_position, cylinder_x_radius, 10.0);
    float test_cylinder_y = sdCylinderY(x - cylinder_y_position, cylinder_y_radius, 10.0);

    [Disc]
    if (test_sphere > 0.0) {
        [Disc]
        if (test_cube > 0.0) {
            [Disc]
            if (test_cylinder_z > 0.0) {
                [Disc]
                if (test_cylinder_x > 0.0) {
                    [Disc]
                    if (test_cylinder_y > 0.0) {
                        res[thread_idx] = 1.0;
                        return;
                    }
                }
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
)
{
    uint3 dispatch_id = cudaBlockIdx() * cudaBlockDim() + cudaThreadIdx();
    int X_NUM = x.size(0);
    int i = dispatch_id.x;
    if (i >= X_NUM) return;
    r[i] = 0.0;  // Initialize to 0
    float3 x_ = float3(x[i,0], x[i,1], x[i,2]);
    start(x_, p, r, i);
} 
