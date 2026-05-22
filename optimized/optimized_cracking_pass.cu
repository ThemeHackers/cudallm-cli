#include <iostream>
#include <cuda_runtime.h>


__device__ void device_md5(const char* password, unsigned int* hash) {

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