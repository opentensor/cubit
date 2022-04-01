#ifndef MAIN_H
#define MAIN_H

unsigned long solve_cuda_c(int blockSize, unsigned char* seal, unsigned int* nonce_start, unsigned int update_interval, unsigned int n_nonces, uint256 limit, unsigned char* block_bytes);

#endif // MAIN_H