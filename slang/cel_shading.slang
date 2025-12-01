#define EPSILON 0.0001
#define MAX_STEPS 500
#define MIN_DIST 0.0
#define MAX_DIST 25.0

#define AMBIENT 0.1

[Differentiable]
[NoDiscontinuity]
float TorusSDF(float3 samplePoint, float2 dimensions)
{
	return length(float2(length(samplePoint.xz) - dimensions.x, samplePoint.y)) - dimensions.y;
}

[Differentiable]
[NoDiscontinuity]
float SceneSDF(float3 samplePoint)
{
    return TorusSDF(samplePoint, float2(1.3, 0.45));
}

[Differentiable]
float March(float3 origin, float3 direction, float start, float stop, inout float edgeLength, float EDGE_THICKNESS)
{
    float depth = start;

    [MaxIters(500)]
    for	(int i = 0; i < MAX_STEPS; i++)
    {
        float dist = SceneSDF(origin + (depth * direction));
        edgeLength = min(dist, edgeLength);
        float test = EPSILON - dist;

        [Disc]
        if (test >= 0.0) {
            return depth;
        }

        depth += dist;
        if (depth >= stop)
            break;
    }

    return stop;
}

[Differentiable]
[NoDiscontinuity]
float3 RayDirection(float fov, float2 size, float2 fragCoord)
{
    float2 xy = fragCoord - (size / 2.0);
    float z = size.y / tan(radians(fov) / 2.0);
    return normalize(float3(xy, -z));
}

[Differentiable]
[NoDiscontinuity]
float3 EstimateNormal(float3 point)
{
    return normalize(float3(
        SceneSDF(float3(point.x + EPSILON, point.y, point.z)) - SceneSDF(float3(point.x - EPSILON, point.y, point.z)),
        SceneSDF(float3(point.x, point.y + EPSILON, point.z)) - SceneSDF(float3(point.x, point.y - EPSILON, point.z)),
        SceneSDF(float3(point.x, point.y, point.z + EPSILON)) - SceneSDF(float3(point.x, point.y, point.z - EPSILON))
    ));
}

[Differentiable]
[NoDiscontinuity]
float4x4 LookAt(float3 camera, float3 target, float3 up)
{
    float3 f = normalize(target - camera);
    float3 s = cross(f, up);
    float3 u = cross(s, f);

    return float4x4(
        float4(s, 0.0),
        float4(u, 0.0),
        float4(-f, 0.0),
        float4(0.0, 0.0, 0.0, 1.0)
    );
}

[Differentiable]
[PreferRecompute]
float3 ComputeLighting(float3 point, float3 lightDir, float3 lightColor, DiffTensorView<float> p)
{
    float3 color = float3(AMBIENT);
    float intensity = dot(EstimateNormal(point), normalize(lightDir));
    float3 thresholds = float3(load(p, 4), load(p, 5), load(p, 6));

    [Disc]
    if (thresholds.x - intensity > 0.0) {
        intensity = thresholds.x;
    } else {
        [Disc]
        if (thresholds.y - intensity > 0.0) {
            intensity = thresholds.y;
        } else {
            [Disc]
            if (thresholds.z - intensity > 0.0) {
                intensity = thresholds.z;
            } else {
                intensity = 1.0;
            }
        }
    }
    intensity = max(intensity, AMBIENT);
    color = lightColor * intensity;
    return color;
}


[Differentiable]
void start(float2 x, DiffTensorView<float> p, DiffTensorView<float> res, int thread_idx) {
    float3 color = float3(0.0);
    float iTime = load(p, 0);
    float2 fragCoord = x;
    float2 iResolution = 1.0;

    float3 viewDir = RayDirection(45.0, iResolution.xy, fragCoord);
    float3 origin = float3(sin(iTime) * 9.0, (sin(iTime * 2.0) * 4.0) + 6.0, cos(iTime) * 9.0);
    float4x4 viewTransform = transpose(LookAt(origin, float3(0.0), float3(0.0, 1.0, 0.0)));
    viewDir = mul(viewTransform, float4(viewDir, 0.0)).xyz;

    float EDGE_THICKNESS = load(p, 7);

    float edgeLength = MAX_DIST;
    float dist = March(origin, viewDir, MIN_DIST, MAX_DIST, edgeLength, EDGE_THICKNESS);

    if (dist > MAX_DIST - EPSILON) {
        color = float3(0.6);
        res[thread_idx, 0] = color.x;
        res[thread_idx, 1] = color.y;
        res[thread_idx, 2] = color.z;
        return;
    }

    if (dist < EPSILON) {
        color = float3(0.0);
        res[thread_idx, 0] = color.x;
        res[thread_idx, 1] = color.y;
        res[thread_idx, 2] = color.z;
        return;
    }

    float3 hitPoint = origin + (dist * viewDir);
    float3 lightDir = float3(load(p, 1), load(p, 2), load(p, 3));
    float3 baseColor = float3(1.0, 0.5, 0.1);

    color = ComputeLighting(hitPoint, lightDir, baseColor, p);

    res[thread_idx, 0] = color.x;
    res[thread_idx, 1] = color.y;
    res[thread_idx, 2] = color.z;
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

    r[i, 0] = 0.0;
    r[i, 1] = 0.0;
    r[i, 2] = 0.0;

    float2 x_ = float2(x[i,0], x[i,1]);
    start(x_, p, r, i);
}
