#ifndef MAIN_H
#define MAIN_H

extern int runTestLessThan(unsigned long* a, unsigned long* b);
extern void runTestCreatePreSeal(unsigned char* pre_seal, uint64 nonce, unsigned char* block_bytes);
extern void runTestCreateNonceBytes(unsigned long long nonce, unsigned char* nonce_bytes);
extern void runTestPreSealHash(unsigned char* seal, unsigned char* preseal_bytes);
extern void runTestSealHash(unsigned char* seal, unsigned char* block_hash, unsigned long long nonce);
extern void runTest(unsigned char* data, unsigned long size, unsigned char* digest);
extern unsigned long long solve_cuda_c(int blockSize, unsigned char* seal, unsigned long long* nonce_start, unsigned long long update_interval, unsigned int n_nonces, unsigned long* limit, unsigned char* block_bytes);

#endif // MAIN_H