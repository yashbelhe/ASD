#define DISC_EPS 1e-6
bool is_zero_level(float a) {
    return abs(a) < DISC_EPS;
}

// CUDA kernel for FNV-1a hash function
// int fnv1a_hash(int a, int b) {
int hash(int a, int b) {
    const uint32_t FNV_prime = 16777619u;
    const uint32_t FNV_offset_basis = 2166136261u;

    uint32_t hash = FNV_offset_basis;

    // Treat the signed integers as raw bits (cast to uint32_t)
    uint32_t a_unsigned = (uint32_t)a;
    uint32_t b_unsigned = (uint32_t)b;

    // Combine the first number
    hash ^= (a_unsigned >> 24) & 0xFF;
    hash *= FNV_prime;
    hash ^= (a_unsigned >> 16) & 0xFF;
    hash *= FNV_prime;
    hash ^= (a_unsigned >> 8) & 0xFF;
    hash *= FNV_prime;
    hash ^= a_unsigned & 0xFF;
    hash *= FNV_prime;

    // Combine the second number
    hash ^= (b_unsigned >> 24) & 0xFF;
    hash *= FNV_prime;
    hash ^= (b_unsigned >> 16) & 0xFF;
    hash *= FNV_prime;
    hash ^= (b_unsigned >> 8) & 0xFF;
    hash *= FNV_prime;
    hash ^= b_unsigned & 0xFF;
    hash *= FNV_prime;

    // Cast the result back to int
    return (int)hash;
}

[PreferRecompute]
[Differentiable]
float load(DiffTensorView<float> p, int idx)
{
  return p[idx];
}

[PreferRecompute]
uint load(TensorView<int> p, int idx)
{
  return p[idx];
}


[Differentiable]
float cross(float2 a, float2 b) {
    return a.x*b.y - a.y*b.x;
}

[Differentiable]
float2 mix(float2 a, float2 b, float t) {
    return a + t * (b - a);
}

[Differentiable]
float mix(float a, float b, float t) {
    return a + t * (b - a);
}

[Differentiable]
float3 mix(float3 a, float3 b, float t) {
    return a + t * (b - a);
}

[Differentiable]
float dot2(float2 v) {
    return dot(v, v);
}
