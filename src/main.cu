/*
 * The MIT License (MIT)
 * Copyright © 2022 Cameron Fairchild

 * Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated 
 * documentation files (the “Software”), to deal in the Software without restriction, including without limitation 
 * the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, 
 * and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

 * The above copyright notice and this permission notice shall be included in all copies or substantial portions of 
 * the Software.

 * THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
 * THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION 
 * OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
 * DEALINGS IN THE SOFTWARE.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <cuda.h>
#include "sha256/sha256.cu"
#include "keccak/keccak.cu"
#include "types.hh"
#include "uint256.cuh"
#include "uint64.cuh"
#include "main.hh"
#include <dirent.h>
#include <ctype.h>


__device__ int lt(uint256 a, uint256 b) {
    // Check if a is less than b
    // Assumes a and b are little-endian
    // This is correct for CUDA
    // https://stackoverflow.com/questions/15356622/anyone-know-whether-nvidias-gpus-are-big-or-little-endian
    BYTE* a_ = (BYTE*)a;
    BYTE* b_ = (BYTE*)b;

    // Checks in reverse order of the bytes
    for (int i = 32 - 1; i >= 0; i--) {
        if (a_[i] < b_[i]) {
            return -1;
        } else if (a_[i] > b_[i]) {
            return 1;
        }
    }
    return 0;
}

__device__ void sha256(unsigned char* data, unsigned long size, unsigned char* digest) {
    CUDA_SHA256_CTX ctx;
    cuda_sha256_init(&ctx);
    cuda_sha256_update(&ctx, data, size);
    cuda_sha256_final(&ctx, digest);
}

__device__ void keccak256(unsigned char* data, unsigned long size, unsigned char* digest) {
    const WORD n_outbit = 256; // 256-bit

    CUDA_KECCAK_CTX ctx;

    cuda_keccak_init(&ctx, n_outbit);
    cuda_keccak_update(&ctx, data, size);
    cuda_keccak_final(&ctx, digest);
}

__device__ bool seal_meets_difficulty(BYTE* seal, uint256 limit) {
    // Need a 256 bit integer to store the seal number
    BYTE seal_[32];

    // Reverse 32 byte array to get little-endian
    for (int i = 0; i < 32; i++) {
        seal_[i] = seal[31-i];
    }

    // Check if the seal number is less than the limit
    int under_limit = lt((unsigned long*)seal_, limit);
    return under_limit == -1;
}

__device__ void create_nonce_bytes(uint64 nonce, BYTE* nonce_bytes) {
    // Convert nonce to bytes (little endian) and store at start of pre_seal;
    for (int i = 0; i < 5; i++) {
        nonce_bytes[i] = (nonce >> (i * 8)) & 0xFF;
    }
}

__device__ int convert_from_ascii_to_int(BYTE ascii_bytes) {
    // Convert the ascii bytes to an integer
    // The ascii bytes are in the form of a hexadecimal number
    int result = 0;
    if (ascii_bytes >= '0' && ascii_bytes <= '9') {
        result = ascii_bytes - '0';
    } else if (ascii_bytes >= 'a' && ascii_bytes <= 'f') {
        result = ascii_bytes - 'a' + 10;
    }
    return result;
}

__device__ void create_pre_seal(BYTE* pre_seal, BYTE* block_hash_bytes, uint64 nonce) {
    BYTE pre_pre_seal[40];
    create_nonce_bytes(nonce, pre_pre_seal);

    for (int i = 0; i < 32; i += 1) {
        // Convert each into ascii and then hex
        unsigned char high_bits = block_hash_bytes[2*i];
        unsigned char low_bits = block_hash_bytes[2*i+1];
        pre_pre_seal[i + 8] = convert_from_ascii_to_int(high_bits) * 16 + convert_from_ascii_to_int(low_bits);
    }

    for (int i = 0; i < 40; i++) {
        pre_seal[i] = pre_pre_seal[i];
    }
}

__device__ void create_seal_hash_from_pre_seal(BYTE* pre_seal, BYTE* seal_hash) {
    BYTE seal_sha256[64];

    // Hash the pre_seal and store in seal;
    sha256(pre_seal, sizeof(BYTE) * 40, seal_sha256);

    // Copy the first 32 bytes of the hash into the seal
    for (int i = 0; i < 32; i++) {
        seal_hash[i] = seal_sha256[i];
    }

    // Hash the seal in keccak
    keccak256(seal_hash, sizeof(BYTE) * 32, seal_hash); 
}

__device__ void create_seal_hash(BYTE* seal, BYTE* block_hash, uint64 nonce) {
    BYTE pre_seal[40];  
    create_pre_seal(pre_seal, block_hash, nonce);
    // Hash the pre_seal and store in seal;
    create_seal_hash_from_pre_seal(pre_seal, seal); 
}

// Flag to indicate if a solution has been found. 
// All threads should stop searching once a solution has been found.
__device__ bool found = false;

// TODO: Use CUDA streams and events to dispatch new blocks and recieve solutions
__global__ void solve(uint64* solution, uint64 nonce_start, uint64 update_interval, uint256 limit, BYTE* block_bytes) {
        for (uint64 i = blockIdx.x * blockDim.x + threadIdx.x; 
                i < update_interval; 
                i += blockDim.x * gridDim.x) 
            {
                if (found) {
                    break;
                }
                BYTE seal[64];

                // Make the seal all 0xff
                for (int j = 0; j < 64; j++) {
                    seal[j] = 0xff;
                }

                uint64 nonce = nonce_start + i;
                create_seal_hash(seal, block_bytes, nonce);
                
                if (seal_meets_difficulty(seal, limit)) {
                    *solution = nonce + 1;
                    found = true;

                    // TODO: Find why these lines make it work
                    // IT'S MAGIC                        
                    BYTE fake_pre_seal[104];  
                    BYTE* fake_block_bytes = fake_pre_seal + 40;
                    create_pre_seal(fake_pre_seal, fake_block_bytes, 10);
                    while (false);

                    return;
                }
                
            }            
}

__global__ void test_lt(uint256 a, uint256 b, int* result) {
    result[0] = lt(a, b);;
}

__global__ void test_create_nonce_bytes(uint64 nonce, BYTE* nonce_bytes) {
    create_nonce_bytes(nonce, nonce_bytes);
}

__global__ void test_create_preseal(BYTE* pre_seal, uint64 nonce, BYTE* block_bytes) {
    create_pre_seal(pre_seal, block_bytes, nonce);
}

__global__ void test_sha256(BYTE* data, int size, BYTE* digest) {
    sha256(data, size, digest);
}

__global__ void test_keccak256(BYTE* data, int size, BYTE* digest) {
    keccak256(data, size, digest);
}

__global__ void test_seal_hash(BYTE* seal, BYTE* block_hash, uint64 nonce) {
    create_seal_hash(seal, block_hash, nonce);
}

__global__ void test_preseal_hash(BYTE* seal, BYTE* preseal_bytes) {
    create_seal_hash_from_pre_seal(preseal_bytes, seal);
}

__global__ void test_seal_meets_difficulty(BYTE* seal, uint256 limit, bool* result) {
    *result = seal_meets_difficulty(seal, limit);
}

void runSolve(int blockSize, uint64* solution, uint64 nonce_start, uint64 update_interval, uint256 limit, BYTE* block_bytes) {
	int numBlocks = (blockSize + blockSize - 1) / blockSize;

	solve <<< numBlocks, blockSize >>> (solution, nonce_start, update_interval, limit, block_bytes);
}

bool runTestSealMeetsDifficulty(BYTE* seal, uint256 limit) {
    BYTE* dev_seal = NULL;
    unsigned long* dev_limit = NULL;
    bool* dev_result = NULL;

    bool result = false;

    checkCudaErrors(cudaMallocManaged(&dev_seal, sizeof(BYTE) * 32));
    checkCudaErrors(cudaMallocManaged(&dev_limit, 8 * sizeof(unsigned long)));
    checkCudaErrors(cudaMallocManaged(&dev_result, sizeof(bool)));

    checkCudaErrors(cudaMemcpy(dev_seal, seal, sizeof(BYTE) * 32, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dev_limit, limit, 8 * sizeof(unsigned long), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaDeviceSynchronize());
    test_seal_meets_difficulty <<< 1, 1 >>> (dev_seal, dev_limit, dev_result);

    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaMemcpy(&result, dev_result, sizeof(bool), cudaMemcpyDeviceToHost));

    return result;
}

void runTestCreateNonceBytes(uint64 nonce, BYTE* nonce_bytes) {
    BYTE* dev_nonce_bytes = NULL;
    checkCudaErrors(cudaMallocManaged(&dev_nonce_bytes, sizeof(BYTE) * 8));

    checkCudaErrors(cudaDeviceSynchronize());
    test_create_nonce_bytes<<<1, 1>>>(nonce, dev_nonce_bytes);

    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaMemcpy(nonce_bytes, dev_nonce_bytes, sizeof(BYTE) * 8, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaDeviceSynchronize());
}

void runTestCreatePreSeal(unsigned char* pre_seal, uint64 nonce, unsigned char* block_bytes) {
    // Test sha256
    BYTE* dev_pre_seal = NULL;
    checkCudaErrors(cudaMallocManaged(&dev_pre_seal, sizeof(BYTE) * 40));

    // malloc block_bytes
    BYTE* dev_block_bytes = NULL;
    checkCudaErrors(cudaMallocManaged(&dev_block_bytes, 64 * sizeof(BYTE)));
    checkCudaErrors(cudaMemcpy(dev_block_bytes, block_bytes, 64 *  sizeof(BYTE), cudaMemcpyHostToDevice));

    
    checkCudaErrors(cudaDeviceSynchronize());
    test_create_preseal<<<1, 1>>>(dev_pre_seal, nonce, dev_block_bytes);

    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaMemcpy(pre_seal, dev_pre_seal, sizeof(BYTE) * 40, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaDeviceSynchronize());
}

void runTest(BYTE* data, unsigned long size, BYTE* digest) {
    // Test sha256
    BYTE* dev_data = NULL;
    BYTE* dev_digest = NULL;
    checkCudaErrors(cudaMallocManaged(&dev_data, size));
    checkCudaErrors(cudaMallocManaged(&dev_digest, sizeof(BYTE) * 64));
    // Copy data to device
    checkCudaErrors(cudaMemcpy(dev_data, data, size, cudaMemcpyHostToDevice));

    checkCudaErrors(cudaDeviceSynchronize());
    test_sha256<<<1, 1>>>(dev_data, size, dev_digest);

    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaMemcpy(digest, dev_digest, 64, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaDeviceSynchronize());
}

void runTestKeccak(BYTE* data, unsigned long size, BYTE* digest) {
    // Test sha256
    BYTE* dev_data = NULL;
    BYTE* dev_digest = NULL;
    checkCudaErrors(cudaMallocManaged(&dev_data, size));
    checkCudaErrors(cudaMallocManaged(&dev_digest, sizeof(BYTE) * 64));
    // Copy data to device
    checkCudaErrors(cudaMemcpy(dev_data, data, size, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaDeviceSynchronize());
    test_keccak256<<<1, 1>>>(dev_data, size, dev_digest);

    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaMemcpy(digest, dev_digest, 64, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaDeviceSynchronize());
}

void runTestSealHash(BYTE* seal, BYTE* block_hash, uint64 nonce) {
    BYTE* dev_seal = NULL;
    BYTE* dev_block_hash = NULL;
    checkCudaErrors(cudaMallocManaged(&dev_seal, 64));
    checkCudaErrors(cudaMallocManaged(&dev_block_hash, 64));
    // Copy data to device
    checkCudaErrors(cudaMemcpy(dev_block_hash, block_hash, 64, cudaMemcpyHostToDevice));

    

    checkCudaErrors(cudaDeviceSynchronize());
    test_seal_hash<<<1, 1>>>(dev_seal, dev_block_hash, nonce);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaMemcpy(seal, dev_seal, 64, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaDeviceSynchronize());
}

void runTestPreSealHash(unsigned char* seal, unsigned char* preseal_bytes) {
    BYTE* dev_seal = NULL;
    BYTE* dev_preseal_bytes = NULL;
    checkCudaErrors(cudaMallocManaged(&dev_seal, 64));
    checkCudaErrors(cudaMallocManaged(&dev_preseal_bytes, 40));
    // Copy data to device
    checkCudaErrors(cudaMemcpy(dev_preseal_bytes, preseal_bytes, 40, cudaMemcpyHostToDevice));

    

    checkCudaErrors(cudaDeviceSynchronize());
    test_preseal_hash<<<1, 1>>>(dev_seal, dev_preseal_bytes);

    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaMemcpy(seal, dev_seal, 64, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaDeviceSynchronize());
}

int runTestLessThan(uint256 a, uint256 b) {
    unsigned long* dev_a;
    unsigned long* dev_b;
    int* dev_result;
    int result[1];
    checkCudaErrors(cudaMallocManaged(&dev_a, 8 * sizeof(unsigned long)));
    checkCudaErrors(cudaMallocManaged(&dev_b, 8 * sizeof(unsigned long)));
    checkCudaErrors(cudaMallocManaged(&dev_result, 1 * sizeof(int)));
    // Copy data to device
    checkCudaErrors(cudaMemcpy(dev_a, a, 8 * sizeof(unsigned long), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dev_b, b, 8 * sizeof(unsigned long), cudaMemcpyHostToDevice));

    checkCudaErrors(cudaDeviceSynchronize());
    test_lt<<<1, 1>>>(dev_a, dev_b, dev_result);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaMemcpy(result, dev_result, sizeof(int), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaDeviceSynchronize());

    return result[0];
}

uint64 solve_cuda_c(int blockSize, BYTE* seal, uint64 nonce_start, uint64 update_interval, uint256 limit, BYTE* block_bytes, int dev_id) {
	unsigned char* block_bytes_d;
    unsigned char* block_bytes_h;
    uint64* solution_d;
    uint64* solution_;
    uint64 solution = 0;
    unsigned long* limit_d;
    unsigned long* limit_h;

    checkCudaErrors(cudaSetDevice(dev_id));

    // Allocate pinned memory on host. This should speed up the data transfer back.
    checkCudaErrors(cudaMallocHost((void**)&solution_, sizeof(uint64)));
    checkCudaErrors(cudaMallocHost((void**)&block_bytes_h, 64 * sizeof(BYTE)));
    checkCudaErrors(cudaMallocHost((void**)&limit_h, 8 * sizeof(unsigned long)));
    // Copy into pinned memory
    memcpy(block_bytes_h, block_bytes, 64 * sizeof(BYTE));
    memcpy(limit_h, limit, 8 * sizeof(unsigned long));
    // Allocate memory on device
    
    // Malloc space for solution in device memory. Should be a single unsigned long.
    checkCudaErrors(cudaMalloc(&solution_d, sizeof(uint64)));
    // Malloc space for block_bytes in device memory. Should be 32 bytes.
    checkCudaErrors(cudaMalloc(&block_bytes_d, 64 * sizeof(BYTE)));
    // Malloc space for limit in device memory.
    checkCudaErrors(cudaMalloc(&limit_d, 8 * sizeof(unsigned long)));

	// Copy data to device memory
	// Put block bytes in device memory. Should be 32 bytes.
	checkCudaErrors(cudaMemcpy(block_bytes_d, block_bytes_h, 64 * sizeof(BYTE), cudaMemcpyHostToDevice));
    // Put limit in device memory.
    checkCudaErrors(cudaMemcpy(limit_d, limit_h, 8 * sizeof(unsigned long), cudaMemcpyHostToDevice));

    // Zero out solution
    checkCudaErrors(cudaMemset(solution_d, 0, sizeof(uint64)));

    checkCudaErrors(cudaDeviceSynchronize());

    // Running Solve on GPU
	runSolve(blockSize, solution_d, nonce_start, update_interval, limit_d, block_bytes_d);

	checkCudaErrors(cudaDeviceSynchronize());
    
    // Copy data back to host memory
    checkCudaErrors(cudaMemcpy(solution_, solution_d, sizeof(uint64), cudaMemcpyDeviceToHost));
    // Check if solution is valid
    solution = *solution_;

    // Free memory
    checkCudaErrors(cudaFree(solution_d));
    checkCudaErrors(cudaFree(block_bytes_d));
    checkCudaErrors(cudaFree(limit_d));

    checkCudaErrors(cudaFreeHost(solution_));
    checkCudaErrors(cudaFreeHost(block_bytes_h));
    checkCudaErrors(cudaFreeHost(limit_h));	

    checkCudaErrors(cudaDeviceSynchronize());

    return solution;
}

void reset_cuda_c() {
    checkCudaErrors(cudaDeviceReset());
}