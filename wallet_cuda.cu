#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <unistd.h>
#include <time.h>
#include <sys/stat.h>
#include "wallet_cuda.h"

// Constants for key generation
#define THREADS_PER_BLOCK 512
#define MAX_BLOCKS 65535
#define NUM_STREAMS 4

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Debug print macro
#define DEBUG_PRINT(fmt, ...) \
    do { fprintf(stderr, "[CUDA Debug] " fmt "\n", ##__VA_ARGS__); } while(0)

// Initialize CUDA random number generator states
__global__ void init_rand_kernel(curandState *state, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(seed, idx, 0, &state[idx]);
}

// Generate random private keys
__global__ void generate_key_kernel(curandState *state, unsigned char *private_keys, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    
    curandState localState = state[idx];
    unsigned char* privKey = &private_keys[idx * 32];
    
    // Generate 32 random bytes for private key
    for (int i = 0; i < 32; i++) {
        privKey[i] = (unsigned char)(curand(&localState) % 256);
    }
    
    // Ensure first byte is non-zero for valid private key
    if (privKey[0] == 0) {
        privKey[0] = 1;
    }
    
    state[idx] = localState;
}

// Forward declarations
static void check_device_permissions(void);

// Helper function implementation
static void check_device_permissions(void) {
    const char* device_paths[] = {
        "/dev/nvidia0",
        "/dev/nvidiactl",
        "/dev/nvidia-uvm",
        NULL
    };

    for (const char** path = device_paths; *path != NULL; path++) {
        struct stat st;
        if (stat(*path, &st) == 0) {
            DEBUG_PRINT("%s: exists", *path);
        }
    }
}

// External C functions
extern "C" {

int cuda_available(void) {
    check_device_permissions();
    
    int deviceCount;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess || deviceCount == 0) {
        return 0;
    }
    
    return cudaSetDevice(0) == cudaSuccess ? 1 : 0;
}

void generate_private_keys_cuda(unsigned char* private_keys, int count) {
    if (!private_keys || count <= 0) {
        DEBUG_PRINT("Invalid parameters: private_keys=%p, count=%d", private_keys, count);
        return;
    }
    
    // Configure kernel launch with optimized parameters
    const int batchSize = count * 64;  // Process 64x more keys at once
    int threadsPerBlock = THREADS_PER_BLOCK;
    int blocks = (batchSize + threadsPerBlock - 1) / threadsPerBlock;
    if (blocks > MAX_BLOCKS) blocks = MAX_BLOCKS;
    
    // Create CUDA streams for overlapping operations
    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; i++) {
        CUDA_CHECK(cudaStreamCreate(&streams[i]));
    }
    
    // Allocate device memory for larger batch
    curandState* d_states;
    unsigned char* d_private_keys;
    
    size_t stateSize = batchSize * sizeof(curandState);
    size_t keySize = batchSize * 32;
    
    CUDA_CHECK(cudaMalloc(&d_states, stateSize));
    CUDA_CHECK(cudaMalloc(&d_private_keys, keySize));
    
    // Process in streams
    int keysPerStream = batchSize / NUM_STREAMS;
    int blocksPerStream = blocks / NUM_STREAMS;
    
    for (int i = 0; i < NUM_STREAMS; i++) {
        int offset = i * keysPerStream;
        init_rand_kernel<<<blocksPerStream, threadsPerBlock, 0, streams[i]>>>
            (d_states + offset, time(NULL) + i);
        generate_key_kernel<<<blocksPerStream, threadsPerBlock, 0, streams[i]>>>
            (d_states + offset, d_private_keys + (offset * 32), keysPerStream);
    }
    
    // Copy results back to host
    CUDA_CHECK(cudaMemcpy(private_keys, d_private_keys, count * 32, cudaMemcpyDeviceToHost));
    
    // Cleanup
    cudaFree(d_states);
    cudaFree(d_private_keys);
    
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamDestroy(streams[i]);
    }
}

} // extern "C"
