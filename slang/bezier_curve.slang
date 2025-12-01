import utils;

[Differentiable]
[NoDiscontinuity]
float sdBezier(float2 pos, float2 A, float2 B, float2 C)
{    
    float2 a = B - A;
    float2 b = A - 2.0*B + C;
    float2 c = a * 2.0;
    float2 d = A - pos;
    float kk = 1.0/dot(b,b);
    float kx = kk * dot(a,b);
    float ky = kk * (2.0*dot(a,a)+dot(d,b)) / 3.0;
    float kz = kk * dot(d,a);      
    float res = 0.0;
    float p = ky - kx*kx;
    float p3 = p*p*p;
    float q = kx*(2.0*kx*kx-3.0*ky) + kz;
    float h = q*q + 4.0*p3;
    if(h >= 0.0) 
    { 
        h = sqrt(h);
        float2 x = (float2(h,-h)-q)/2.0;
        float2 uv = sign(x)*pow(abs(x), float2(1.0/3.0));
        float t = clamp(uv.x+uv.y-kx, 0.0, 1.0);
        res = dot2(d + (c + b*t)*t);
    }
    else
    {
        float z = sqrt(-p);
        float v = acos(q/(p*z*2.0)) / 3.0;
        float m = cos(v);
        float n = sin(v)*1.732050808;
        float3 t = clamp(float3(m+m,-n-m,n-m)*z-kx, 0.0, 1.0);
        res = min(dot2(d+(c+b*t.x)*t.x),
                 dot2(d+(c+b*t.y)*t.y));
    }
    return sqrt(res);
}

[Differentiable]
[NoDiscontinuity]
float sdQuadraticBezier(float2 p, float2 p0, float2 p1, float2 p2, float width) {
    float dist = sdBezier(p, p0, p1, p2);
    return dist - width;
}

[Differentiable]
void start(float2 x, DiffTensorView<float> p, DiffTensorView<float> res, int thread_idx) {
    float2 p0 = float2(p[0], p[1]);
    float2 p1 = float2(p[2], p[3]);
    float2 p2 = float2(p[4], p[5]);
    float width = p[6];
    float sd = sdQuadraticBezier(x, p0, p1, p2, width);
    [Disc]
    if (sd <= 0.0) {
        res[thread_idx] = 1.0;
    } else {
        res[thread_idx] = 0.0;
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
    r[i] = 0.0;
    float2 x_ = float2(x[i,0], x[i,1]);
    start(x_, p, r, i);
}
