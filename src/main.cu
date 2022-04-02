#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <cuda.h>
#include "sha256.cuh"
#include "uint256.cuh"
#include "main.hh"
#include <dirent.h>
#include <ctype.h>

// Copyright (c) 2009-2010 Satoshi Nakamoto
// Copyright (c) 2009-2019 The Bitcoin Core developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.
__device__ int lt(uint256 a, uint256 b) {
    // Assumes a and b are little-endian
    // This is correct for CUDA
    // https://stackoverflow.com/questions/15356622/anyone-know-whether-nvidias-gpus-are-big-or-little-endian
    
    for (int i = 8 - 1; i >= 0; i--) {
        if (a[i] < b[i]) {
            return -1;
        } else if (a[i] > b[i]) {
            return 1;
        }
    }
    return 0;
}

__device__ void sha256(unsigned char* data, unsigned long size, unsigned char* digest) {
    SHA256_CTX ctx;
    sha256_init(&ctx);
    sha256_update(&ctx, data, size);
    sha256_final(&ctx, digest);
}

__device__ bool seal_meets_difficulty(BYTE* seal, uint256 limit) {
    // Need a 256 bit integer to store the seal number
    uint256 seal_number;
    
    // Seal is big-endian, and we want little-endian
    for (int i = 0; i < 8; i++) {
        seal_number[i] = seal[60 - (i * 4)] | (seal[61 - (i * 4)] << 8) | (seal[62 - (i * 4)] << 16) | (seal[63 - (i * 4)] << 24);
    }

    // Check if the seal number is less than the limit
    return lt(seal_number, limit) == -1;
}

__device__ void create_seal_hash(BYTE* seal, BYTE* block_hash, unsigned long nonce) {
    BYTE pre_seal[40];
    
    // Convert nonce to bytes (little endian) and store at start of pre_seal;
    for (int i = 0; i < 8; i++) {
        pre_seal[i] = (nonce >> (i * 8)) & 0xFF;
    }

    // Store the block bytes at the end of pre_seal;
    for (int i = 0; i < 32; i++) {
        pre_seal[i + 8] = block_hash[i];
    }
    
    // Hash the pre_seal and store in seal;
    sha256(pre_seal, sizeof(BYTE) * 40, seal);     
}

__global__ void solve(BYTE* seal, unsigned long* solution, unsigned long* nonce_start, unsigned long update_interval, unsigned int n_nonces, uint256 limit, BYTE* block_bytes) {
        BYTE seal_[64];
        
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; 
                i < n_nonces; 
                i += blockDim.x * gridDim.x) 
            {
                unsigned long nonce = nonce_start[i];
                for (unsigned long j = nonce; j < nonce + update_interval; j++) {
                    create_seal_hash(seal_, block_bytes, j);
                    if (seal_meets_difficulty(seal_, limit)) {
                        solution[i] = j;
                        // Copy seal to shared memory
                        for (int k = 0; k < 64; k++) {
                            seal[k] = seal_[k];
                        }
                        break;
                    }            
                }
            }            
}

__global__ void test_sha256(BYTE* data, int size, BYTE* digest) {
    sha256(data, size, digest);
}

__global__ void test_seal_hash(BYTE* seal, BYTE* block_hash, unsigned long nonce) {
    create_seal_hash(seal, block_hash, nonce);
}

__global__ void test_preseal_hash(BYTE* seal, BYTE* preseal_bytes) {
    sha256(preseal_bytes, sizeof(BYTE) * 40, seal);
}

void pre_sha256() {
	// copy symbols
	checkCudaErrors(cudaMemcpyToSymbol(dev_k, host_k, sizeof(host_k), 0, cudaMemcpyHostToDevice));
}

void runSolve(int blockSize, BYTE* seal, unsigned long* solution, unsigned long* nonce_start, unsigned long update_interval, unsigned int n_nonces, uint256 limit, BYTE* block_bytes) {
	int numBlocks = (n_nonces + blockSize - 1) / blockSize;
	solve <<< numBlocks, blockSize >>> (seal, solution, nonce_start, update_interval, n_nonces, limit, block_bytes);
}

void runTest(BYTE* data, unsigned long size, BYTE* digest) {
    // Test sha256
    BYTE* dev_data;
    BYTE* dev_digest;
    checkCudaErrors(cudaMallocManaged(&dev_data, size));
    checkCudaErrors(cudaMallocManaged(&dev_digest, sizeof(BYTE) * 64));
    // Copy data to device
    checkCudaErrors(cudaMemcpy(dev_data, data, size, cudaMemcpyHostToDevice));

    pre_sha256();
    test_sha256<<<1, 1>>>(dev_data, size, dev_digest);
    cudaDeviceSynchronize();
    checkCudaErrors(cudaMemcpy(digest, dev_digest, 64, cudaMemcpyDeviceToHost));
    cudaDeviceReset();
}

void runTestSealHash(BYTE* seal, BYTE* block_hash, unsigned long nonce) {
    BYTE* dev_seal;
    BYTE* dev_block_hash;
    checkCudaErrors(cudaMallocManaged(&dev_seal, 64));
    checkCudaErrors(cudaMallocManaged(&dev_block_hash, 32));
    // Copy data to device
    checkCudaErrors(cudaMemcpy(dev_block_hash, block_hash, 32, cudaMemcpyHostToDevice));

    pre_sha256();

    test_seal_hash<<<1, 1>>>(dev_seal, dev_block_hash, nonce);
    cudaDeviceSynchronize();
    checkCudaErrors(cudaMemcpy(seal, dev_seal, 64, cudaMemcpyDeviceToHost));
    cudaDeviceReset();
}

void runTestPreSealHash(unsigned char* seal, unsigned char* preseal_bytes) {
    BYTE* dev_seal;
    BYTE* dev_preseal_bytes;
    checkCudaErrors(cudaMallocManaged(&dev_seal, 64));
    checkCudaErrors(cudaMallocManaged(&dev_preseal_bytes, 40));
    // Copy data to device
    checkCudaErrors(cudaMemcpy(dev_preseal_bytes, preseal_bytes, 40, cudaMemcpyHostToDevice));

    pre_sha256();

    test_preseal_hash<<<1, 1>>>(dev_seal, dev_preseal_bytes);
    cudaDeviceSynchronize();
    checkCudaErrors(cudaMemcpy(seal, dev_seal, 64, cudaMemcpyDeviceToHost));
    cudaDeviceReset();
}

unsigned long solve_cuda_c(int blockSize, BYTE* seal, unsigned long* nonce_start, unsigned long update_interval, unsigned int n_nonces, uint256 limit, BYTE* block_bytes) {
	unsigned long* nonce_start_d;
	unsigned char* block_bytes_d;
    BYTE* seal_d;
    unsigned long* solution_d;
    unsigned long solution[1] = {0};
    unsigned long* limit_d;

    // Allocate memory on device
    
    // Malloc space for solution in device memory. Should be a single unsigned long.
    printf("Allocating memory on device\n");
    checkCudaErrors(cudaMallocManaged(&solution_d, sizeof(unsigned long)));
    // Malloc space for seal in device memory. Should be one seal.
    checkCudaErrors(cudaMallocManaged(&seal_d, 64 * sizeof(BYTE)));
    // Malloc space for nonce_start in device memory.
    checkCudaErrors(cudaMallocManaged(&nonce_start_d, n_nonces * sizeof(unsigned long)));
    // Malloc space for block_bytes in device memory. Should be 32 bytes.
    checkCudaErrors(cudaMallocManaged(&block_bytes_d, 32 * sizeof(BYTE)));
    // Malloc space for limit in device memory.
    checkCudaErrors(cudaMallocManaged(&limit_d, 8 * sizeof(unsigned long)));

	// Copy data to device memory
    printf("Copying memory to device\n");
	// Put block bytes in device memory. Should be 32 bytes.
	checkCudaErrors(cudaMemcpy(block_bytes_d, block_bytes, 32 * sizeof(BYTE), cudaMemcpyHostToDevice));
	// Put nonce_start in device memory. Should be a single int for each thread.
	checkCudaErrors(cudaMemcpy(nonce_start_d, nonce_start, n_nonces * sizeof(unsigned long), cudaMemcpyHostToDevice));
    // Put limit in device memory.
    checkCudaErrors(cudaMemcpy(limit_d, limit, 8 * sizeof(unsigned long), cudaMemcpyHostToDevice));

    // Set seal to 0xff
    checkCudaErrors(cudaMemset(seal_d, 0xff, 64 * sizeof(unsigned char)));
    // Zero solution
    solution_d[0] = 0;

	pre_sha256();

    // Running Solve on GPU
    printf("Running solve on GPU\n");
	runSolve(blockSize, seal_d, solution_d, nonce_start_d, update_interval, n_nonces, limit_d, block_bytes_d);

	cudaDeviceSynchronize();
    
    // Copy data back to host memory
    printf("Copying memory to host\n");
    checkCudaErrors(cudaMemcpy(solution, solution_d, 1 * sizeof(unsigned long), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(seal, seal_d, 64 * sizeof(BYTE), cudaMemcpyDeviceToHost));
    
	cudaDeviceReset();
	return solution[0];
}
