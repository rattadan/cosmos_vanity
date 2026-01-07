//go:build cuda && cgo

package main

/*
#cgo CFLAGS: -I${SRCDIR}/include
#cgo LDFLAGS: -L${SRCDIR}/lib -lwallet_cuda -lcudart
#include <stdlib.h>
#include <stdint.h>
#include "wallet_cuda.h"
*/
import "C"
import (
	"crypto/sha256"
	"fmt"
	"unsafe"

	"github.com/btcsuite/btcd/btcec/v2"
)

// CUDAAvailable returns true if CUDA is available on the system
func CUDAAvailable() bool {
	available := C.cuda_available() != 0
	if !available {
		fmt.Println("CUDA is not available")
	} else {
		fmt.Println("CUDA is available and initialized")
	}
	return available
}

// GenerateWalletsCUDA generates multiple wallets using CUDA acceleration
func GenerateWalletsCUDA(count int) []wallet {
	fmt.Printf("Generating %d wallets using CUDA\n", count)

	// Allocate memory for private keys
	privateKeyBytes := make([]byte, count*32)

	// Generate private keys in parallel using CUDA
	C.generate_private_keys_cuda(
		(*C.uchar)(unsafe.Pointer(&privateKeyBytes[0])),
		C.int(count),
	)

	// Convert results to wallet structs
	wallets := make([]wallet, count)
	for i := 0; i < count; i++ {
		// Extract private key
		privKeyBytes := make([]byte, 32)
		copy(privKeyBytes, privateKeyBytes[i*32:(i+1)*32])

		// Create private key using btcec
		_, pubKey := btcec.PrivKeyFromBytes(privKeyBytes)

		// Get compressed public key
		pubKeyBytes := pubKey.SerializeCompressed()

		// Generate address using SHA256
		hash := sha256.Sum256(pubKeyBytes)

		// Create wallet struct
		wallets[i] = wallet{
			Address: fmt.Sprintf("cosmos%x", hash[:20]), // Use first 20 bytes of hash
			Pubkey:  pubKeyBytes,
			Privkey: privKeyBytes,
		}
	}

	return wallets
}
