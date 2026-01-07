#!/bin/bash

# Set CUDA environment variables
export CUDA_VISIBLE_DEVICES=0
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${PWD}:$LD_LIBRARY_PATH

# Run the program
./cosmosvanity "$@"
