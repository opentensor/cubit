#ifndef MAIN_H
#define MAIN_H

extern unsigned long solve_cuda_c(int blockSize, unsigned char* seal, unsigned long* nonce_start, unsigned long update_interval, unsigned int n_nonces, unsigned long* limit, unsigned char* block_bytes);

#endif // MAIN_H