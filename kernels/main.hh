/*
 * The MIT License (MIT)
 * Copyright (c) 2022 Cameron Fairchild
 * Copyright (c) 2022 Opentensor Foundation

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

#ifndef MAIN_H
#define MAIN_H

extern void reset_cuda_c();
extern int runTestLessThan(unsigned long* a, unsigned long* b);
extern bool runTestSealMeetsDifficulty(unsigned char* seal, unsigned long* limit);
extern void runTestCreatePreSeal(unsigned char* pre_seal, uint64 nonce, unsigned char* block_bytes, int dev_id);
extern void runTestCreateNonceBytes(unsigned long long nonce, unsigned char* nonce_bytes, int dev_id);
extern void runTestPreSealHash(unsigned char* seal, unsigned char* preseal_bytes);
extern void runTestSealHash(unsigned char* seal, unsigned char* block_hash, uint64 nonce, int dev_id);
extern void runTest(unsigned char* data, unsigned long size, unsigned char* digest);
extern void runTestKeccak(unsigned char* data, unsigned long size, unsigned char* digest);
extern uint64 solve_cuda_c(int blockSize, unsigned char* seal, uint64 nonce_start, uint64 update_interval, unsigned long* limit, unsigned char* block_bytes, int dev_id);
extern void expose_cuda_errors();

#endif // MAIN_H