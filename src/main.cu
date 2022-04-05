#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <cuda.h>
#include "sha256.cuh"
#include "uint256.cuh"
#include "uint64.cuh"
#include "main.hh"
#include <dirent.h>
#include <ctype.h>

// Copyright (c) 2009-2010 Satoshi Nakamoto
// Copyright (c) 2009-2019 The Bitcoin Core developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.
__device__ int lt(uint256 a, uint256 b) {
    // Check if a is less than b
    // Assumes a and b are little-endian
    // This is correct for CUDA
    // https://stackoverflow.com/questions/15356622/anyone-know-whether-nvidias-gpus-are-big-or-little-endian
    BYTE* a_ = (BYTE*)a;
    BYTE* b_ = (BYTE*)b;

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
    SHA256_CTX ctx;
    sha256_init(&ctx);
    sha256_update(&ctx, data, size);
    sha256_final(&ctx, digest);
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
    for (int i = 0; i < 4; i++) {
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

__device__ void create_seal_hash(BYTE* seal, BYTE* block_hash, uint64 nonce) {
    BYTE pre_seal[40];    

    create_pre_seal(pre_seal, block_hash, nonce);
    
    // Hash the pre_seal and store in seal;
    sha256(pre_seal, sizeof(BYTE) * 40, seal);     
}

__global__ void solve(BYTE** seals, uint64* solution, uint64 nonce_start, uint64 update_interval, unsigned int n_nonces, uint256 limit, BYTE* block_bytes) {
        __shared__ bool found;
        found = false;
        
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; 
                i < n_nonces; 
                i += blockDim.x * gridDim.x) 
            {
                if (found) {
                    break;
                }
                BYTE seal[64];

                // Make the seal all 0xff
                for (int j = 0; j < 64; j++) {
                    seal[i] = 0xff;
                }

                uint64 nonce = nonce_start + i * update_interval;
                for (
                    uint64 j = nonce; j < nonce + update_interval; j++) {
                    create_seal_hash(seal, block_bytes, j);
                    
                    if (seal_meets_difficulty(seal, limit)) {
                        solution[i] = j + 1;

                        // Copy seal to shared memory
                       // for (int k = 0; k < 64; k++) {
                         //   seal[i][k] = seal_[k];
                            // print the seal
                            //if (k == 32) {
                            //    printf("i = 32;\n");
                            //}
                            //printf("%02x ", seal_[k]);
                        
                        //}
                        //printf("\n");
                        found = true;
                        break;
                    }
                }
            }            
}

__global__ void test_lt(uint256 a, uint256 b, int* result) {
    result[0] = lt(a, b);
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

__global__ void test_seal_hash(BYTE* seal, BYTE* block_hash, uint64 nonce) {
    create_seal_hash(seal, block_hash, nonce);
}

__global__ void test_preseal_hash(BYTE* seal, BYTE* preseal_bytes) {
    sha256(preseal_bytes, sizeof(BYTE) * 40, seal);
}

__global__ void test_seal_meets_difficulty(BYTE* seal, uint256 limit, bool* result) {
    seal_meets_difficulty(seal, limit);
}

void pre_sha256() {
	// copy symbols
	checkCudaErrors(cudaMemcpyToSymbol(dev_k, host_k, sizeof(host_k), 0, cudaMemcpyHostToDevice));
}

void runSolve(int blockSize, BYTE** seals, uint64* solution, uint64 nonce_start, uint64 update_interval, uint256 limit, BYTE* block_bytes) {
	int numBlocks = (blockSize + blockSize - 1) / blockSize;
	solve <<< numBlocks, blockSize >>> (seals, solution, nonce_start, update_interval, blockSize, limit, block_bytes);
}

bool runTestSealMeetsDifficulty(BYTE* seal, uint256 limit) {
    BYTE* dev_seal;
    unsigned long* dev_limit;
    bool* dev_result;

    bool result;

    checkCudaErrors(cudaMallocManaged(&dev_seal, sizeof(BYTE) * 32));
    checkCudaErrors(cudaMallocManaged(&dev_limit, 8 * sizeof(unsigned long)));
    checkCudaErrors(cudaMallocManaged(&dev_result, sizeof(bool)));

    checkCudaErrors(cudaMemcpy(dev_seal, seal, sizeof(BYTE) * 32, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dev_limit, limit, 8 * sizeof(unsigned long), cudaMemcpyHostToDevice));
    
    test_seal_meets_difficulty <<< 1, 1 >>> (seal, limit, dev_result);

    cudaDeviceSynchronize();
    checkCudaErrors(cudaMemcpy(&result, dev_result, sizeof(bool), cudaMemcpyDeviceToHost));
    cudaDeviceReset();

    return result;
}

void runTestCreateNonceBytes(uint64 nonce, BYTE* nonce_bytes) {
    BYTE* dev_nonce_bytes;
    checkCudaErrors(cudaMallocManaged(&dev_nonce_bytes, sizeof(BYTE) * 8));

    pre_sha256();

    test_create_nonce_bytes<<<1, 1>>>(nonce, dev_nonce_bytes);

    cudaDeviceSynchronize();
    checkCudaErrors(cudaMemcpy(nonce_bytes, dev_nonce_bytes, sizeof(BYTE) * 8, cudaMemcpyDeviceToHost));
    cudaDeviceReset();
}

void runTestCreatePreSeal(unsigned char* pre_seal, uint64 nonce, unsigned char* block_bytes) {
    // Test sha256
    BYTE* dev_pre_seal;
    checkCudaErrors(cudaMallocManaged(&dev_pre_seal, sizeof(BYTE) * 40));

    // malloc block_bytes
    BYTE* dev_block_bytes;
    checkCudaErrors(cudaMallocManaged(&dev_block_bytes, 64 * sizeof(BYTE)));
    checkCudaErrors(cudaMemcpy(dev_block_bytes, block_bytes, 64 *  sizeof(BYTE), cudaMemcpyHostToDevice));

    pre_sha256();

    test_create_preseal<<<1, 1>>>(dev_pre_seal, nonce, dev_block_bytes);

    cudaDeviceSynchronize();
    checkCudaErrors(cudaMemcpy(pre_seal, dev_pre_seal, sizeof(BYTE) * 40, cudaMemcpyDeviceToHost));
    cudaDeviceReset();
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

void runTestSealHash(BYTE* seal, BYTE* block_hash, uint64 nonce) {
    BYTE* dev_seal;
    BYTE* dev_block_hash;
    checkCudaErrors(cudaMallocManaged(&dev_seal, 64));
    checkCudaErrors(cudaMallocManaged(&dev_block_hash, 64));
    // Copy data to device
    checkCudaErrors(cudaMemcpy(dev_block_hash, block_hash, 64, cudaMemcpyHostToDevice));

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
    
    test_lt<<<1, 1>>>(dev_a, dev_b, dev_result);
    cudaDeviceSynchronize();
    checkCudaErrors(cudaMemcpy(result, dev_result, sizeof(int), cudaMemcpyDeviceToHost));
    cudaDeviceReset();

    return result[0];
}

uint64 solve_cuda_c(int blockSize, BYTE* seal, uint64 nonce_start, uint64 update_interval, uint256 limit, BYTE* block_bytes) {
	unsigned char* block_bytes_d;
    unsigned char* block_bytes_h;
    uint64* solution_d;
    uint64* solutions;
    uint64 solution = 0;
    unsigned long* limit_d;
    unsigned long* limit_h;

    // Allocate pinned memory on host. This should speed up the data transfer back.
    checkCudaErrors(cudaMallocHost((void**)&solutions, blockSize * 8 * sizeof(unsigned long)));
    checkCudaErrors(cudaMallocHost((void**)&block_bytes_h, 64 * sizeof(BYTE)));
    checkCudaErrors(cudaMallocHost((void**)&limit_h, 8 * sizeof(unsigned long)));
    // Copy into pinned memory
    memcpy(block_bytes_h, block_bytes, 64 * sizeof(BYTE));
    memcpy(limit_h, limit, 8 * sizeof(unsigned long));
    // Allocate memory on device
    
    // Malloc space for solution in device memory. Should be a single unsigned long.
    checkCudaErrors(cudaMallocManaged(&solution_d, blockSize * sizeof(uint64)));
    // Malloc space for block_bytes in device memory. Should be 32 bytes.
    checkCudaErrors(cudaMallocManaged(&block_bytes_d, 64 * sizeof(BYTE)));
    // Malloc space for limit in device memory.
    checkCudaErrors(cudaMallocManaged(&limit_d, 8 * sizeof(unsigned long)));

	// Copy data to device memory
	// Put block bytes in device memory. Should be 32 bytes.
	checkCudaErrors(cudaMemcpy(block_bytes_d, block_bytes_h, 64 * sizeof(BYTE), cudaMemcpyHostToDevice));
    // Put limit in device memory.
    checkCudaErrors(cudaMemcpy(limit_d, limit_h, 8 * sizeof(unsigned long), cudaMemcpyHostToDevice));

	pre_sha256();

    // Zero out solution
    checkCudaErrors(cudaMemset(solution_d, 0, sizeof(uint64)));

    // Running Solve on GPU
	runSolve(blockSize, NULL, solution_d, nonce_start, update_interval, limit_d, block_bytes_d);

	cudaDeviceSynchronize();
    
    // Copy data back to host memory
    checkCudaErrors(cudaMemcpy(solutions, solution_d, blockSize * sizeof(uint64), cudaMemcpyDeviceToHost));
    // Check if solution is valid
    for (int i = 0; i < blockSize; i++) {
        if (solutions[i] != 0) {
            solution = solutions[i];
            break;
        }
    }

    
    // Free memory
    checkCudaErrors(cudaFree(solution_d));
    checkCudaErrors(cudaFree(block_bytes_d));
    checkCudaErrors(cudaFree(limit_d));

    checkCudaErrors(cudaFreeHost(solutions));
    checkCudaErrors(cudaFreeHost(block_bytes_h));
    checkCudaErrors(cudaFreeHost(limit_h));	
    return solution;
}


void reset_cuda_c() {
    cudaDeviceReset();
}