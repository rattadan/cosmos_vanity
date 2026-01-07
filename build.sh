#!/bin/bash
set -e

# Create lib directory if it doesn't exist
mkdir -p lib

# Set library path
export LD_LIBRARY_PATH=${PWD}/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Compile CUDA code
nvcc -O3 -shared -Xcompiler -fPIC \
    -I./include \
    -L/usr/local/cuda/lib64 \
    wallet_cuda.cu -o lib/libwallet_cuda.so

# Build Go code
export CGO_ENABLED=1
export CGO_CFLAGS="-I${PWD}/include"
export CGO_LDFLAGS="-L${PWD}/lib -L/usr/local/cuda/lib64 -lwallet_cuda -lcudart"
go build -ldflags "-r ${PWD}/lib" .
