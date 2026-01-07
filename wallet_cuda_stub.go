//go:build !cuda

package main

// CUDAAvailable returns false when CUDA is not available
func CUDAAvailable() bool {
	return false
}

// GenerateWalletsCUDA generates multiple wallets using CPU (stub when CUDA unavailable)
func GenerateWalletsCUDA(count int) []wallet {
	// Return empty slice when CUDA is not available
	return []wallet{}
}
