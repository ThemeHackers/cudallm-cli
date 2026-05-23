#include <iostream>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <cstring>


__device__ void device_md5(const char* password, unsigned int* hash) {

    hash[0] = hash[1] = hash[2] = hash[3] = 0;
}

__global__ void crackPasswordKernel(const char* charset, int charset_len, int pwd_len, const unsigned int* target_hash, char* result_found, bool* flag) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    

    char current_pwd[8]; 
    int temp = idx;
    

    for (int i = 0; i < pwd_len; i++) {
        current_pwd[i] = charset[temp % charset_len];
        temp /= charset_len;
    }
    current_pwd[pwd_len] = '\0';

   
    unsigned int computed_hash[4];
    device_md5(current_pwd, computed_hash);


    if (computed_hash[0] == target_hash[0] &&
        computed_hash[1] == target_hash[1] &&
        computed_hash[2] == target_hash[2] &&
        computed_hash[3] == target_hash[3]) {
    
        *flag = true; 
        
        for(int i = 0; i < pwd_len; i++) {
            result_found[i] = current_pwd[i];
        }
    }
}

int main(int argc, char* argv[]) {
    const char* h_charset = "abcdefghijklmnopqrstuvwxyz0123456789";
    int charset_len = (int)strlen(h_charset);
    int pwd_len = 4;


    unsigned int h_target[4] = {0, 0, 0, 0};

    char* d_charset;
    unsigned int* d_target;
    char* d_result;
    bool* d_flag;

    cudaMalloc(&d_charset, charset_len);
    cudaMalloc(&d_target, 4 * sizeof(unsigned int));
    cudaMalloc(&d_result, pwd_len + 1);
    cudaMalloc(&d_flag, sizeof(bool));

    cudaMemcpy(d_charset, h_charset, charset_len, cudaMemcpyHostToDevice);
    cudaMemcpy(d_target, h_target, 4 * sizeof(unsigned int), cudaMemcpyHostToDevice);
    bool h_flag = false;
    cudaMemcpy(d_flag, &h_flag, sizeof(bool), cudaMemcpyHostToDevice);

    int total_candidates = 1;
    for (int i = 0; i < pwd_len; i++) total_candidates *= charset_len;

    int threads = 256;
    int blocks = (total_candidates + threads - 1) / threads;

    cudaProfilerStart();
    crackPasswordKernel<<<blocks, threads>>>(d_charset, charset_len, pwd_len, d_target, d_result, d_flag);
    cudaDeviceSynchronize();
    cudaProfilerStop();

    cudaMemcpy(&h_flag, d_flag, sizeof(bool), cudaMemcpyDeviceToHost);
    if (h_flag) {
        char h_result[9] = {};
        cudaMemcpy(h_result, d_result, pwd_len, cudaMemcpyDeviceToHost);
        std::cout << "Password found: " << h_result << std::endl;
    } else {
        std::cout << "Password not found (stub MD5 — expected with zero-target)." << std::endl;
    }

    cudaFree(d_charset);
    cudaFree(d_target);
    cudaFree(d_result);
    cudaFree(d_flag);
    return 0;
}
