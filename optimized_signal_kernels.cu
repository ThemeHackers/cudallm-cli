#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math_constants.h>
extern "C" {
__global__ __launch_bounds__(256, 2)
void dft_kernel(const float* __restrict__ real_in,
                const float* __restrict__ imag_in,
                float*       __restrict__ magnitude_out,
                int n)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n) return;
    float sum_r = 0.0f, sum_i = 0.0f;
    float inv_n = 6.28318531f / (float)n;
    for (int t = 0; t < n; t++) {
        float s, c;
        sincosf(-inv_n * (float)k * (float)t, &s, &c); 
        float xr = __ldg(&real_in[t]);
        float xi = __ldg(&imag_in[t]);
        sum_r += xr * c - xi * s;
        sum_i += xr * s + xi * c;
    }
    magnitude_out[k] = sqrtf(sum_r * sum_r + sum_i * sum_i) / (float)n;
}
__device__ __forceinline__ int bit_reverse(int x, int bits)
{
    return (int)(__brev((unsigned)x) >> (32 - bits));
}
__global__ __launch_bounds__(256, 2)
void fft_bitrev_kernel(float* __restrict__ real,
                       float* __restrict__ imag,
                       int n, int bits)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    int rev = bit_reverse(idx, bits);
    if (idx < rev) {                   
        float tr = real[idx], ti = imag[idx];
        real[idx] = real[rev]; imag[idx] = imag[rev];
        real[rev] = tr;        imag[rev] = ti;
    }
}
__global__ __launch_bounds__(256, 2)
void fft_butterfly_kernel(float* __restrict__ real,
                          float* __restrict__ imag,
                          int n, int step)
{
    int tid      = blockIdx.x * blockDim.x + threadIdx.x;
    int half     = step >> 1;
    if (tid >= n / 2) return;
    int pair     = tid / half;
    int k        = tid % half;
    int i        = pair * step + k;
    int j        = i + half;
    float angle  = -CUDART_PI_F * (float)k / (float)half;
    float wr, wi;
    sincosf(angle, &wi, &wr);           
    float ur = real[i], ui = imag[i];
    float vr = real[j] * wr - imag[j] * wi;
    float vi = real[j] * wi + imag[j] * wr;
    real[i] = ur + vr;  imag[i] = ui + vi;
    real[j] = ur - vr;  imag[j] = ui - vi;
}
__global__ __launch_bounds__(512, 1)
void fft_shared_kernel(float* __restrict__ g_real,
                       float* __restrict__ g_imag,
                       int n, int bits)
{
    extern __shared__ float smem[];    
    float* s_r = smem;
    float* s_i = smem + n;
    int tid = threadIdx.x;
    if (tid >= n) return;
    int rev  = bit_reverse(tid, bits);
    s_r[rev] = g_real[tid];
    s_i[rev] = g_imag[tid];
    __syncthreads();
    for (int step = 2; step <= n; step <<= 1) {
        int half  = step >> 1;
        int pair  = tid / half;
        int k     = tid % half;
        int i     = pair * step + k;
        int j     = i + half;
        float angle = -CUDART_PI_F * (float)k / (float)half;
        float wr, wi;
        sincosf(angle, &wi, &wr);
        float ur = s_r[i], ui = s_i[i];
        float vr = s_r[j] * wr - s_i[j] * wi;
        float vi = s_r[j] * wi + s_i[j] * wr;
        __syncthreads();
        s_r[i] = ur + vr;  s_i[i] = ui + vi;
        s_r[j] = ur - vr;  s_i[j] = ui - vi;
        __syncthreads();
    }
    g_real[tid] = s_r[tid];
    g_imag[tid] = s_i[tid];
}
__global__ __launch_bounds__(256, 2)
void vision_filter_kernel(uchar4* __restrict__ pixels,
                          int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    uchar4 p    = __ldg(&pixels[y * width + x]);
    unsigned char g = (unsigned char)(
        (77u * p.x + 150u * p.y + 29u * p.z) >> 8u);
    pixels[y * width + x] = make_uchar4(g, g, g, p.w);
}
__global__ __launch_bounds__(256, 2)
void pattern_match_kernel(const unsigned char* __restrict__ data,
                          int data_len,
                          int pat_len,
                          int* __restrict__ found_idx)
{
    extern __shared__ unsigned char s_pat[];
    for (int i = threadIdx.x; i < pat_len; i += blockDim.x)
        s_pat[i] = data[i];
    __syncthreads();
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int end = data_len - pat_len;
    bool found = false;
    if (idx <= end) {
        found = true;
        for (int i = 0; i < pat_len && found; i++)
            found = (data[idx + i] == s_pat[i]);
    }
    unsigned mask = __ballot_sync(0xFFFFFFFF, found);
    if (mask && (threadIdx.x % 32 == __ffs(mask) - 1)) {
        atomicMin(found_idx, idx);     
    }
}
}
