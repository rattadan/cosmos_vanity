#ifndef WALLET_CUDA_H
#define WALLET_CUDA_H

#ifdef __cplusplus
extern "C" {
#endif

// Check if CUDA is available
int cuda_available(void);

// Generate multiple private keys in parallel using CUDA
void generate_private_keys_cuda(
    unsigned char* private_keys,  // Output: 32 bytes per key
    int count                    // Number of keys to generate
);

#ifdef __cplusplus
}
#endif

#endif // WALLET_CUDA_H
