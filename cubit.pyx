#cython: language_level=3

# The MIT License (MIT)
# Copyright (c) 2022 Cameron Fairchild
# Copyright (c) 2022 Opentensor Foundation

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated 
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation 
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, 
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of 
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION 
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
# DEALINGS IN THE SOFTWARE.

from cpython cimport array
from cpython.mem cimport PyMem_Malloc, PyMem_Free
from libc.string cimport memcpy
import array

cdef extern from "kernels/int128.hh":
    ctypedef int int128

cdef extern from "kernels/uint64.cuh":
    ctypedef unsigned long long uint64;

cdef extern from "kernels/uint256.cuh":    
    ctypedef unsigned long uint256[8]

cdef extern from "kernels/main.hh":
    void reset_cuda_c();
    int runTestSealMeetsDifficulty(unsigned char* seal, unsigned long* limit);
    int runTestLessThan(uint256 a, uint256 b);
    void runTestCreatePreSeal(unsigned char* pre_seal, uint64 nonce, unsigned char* block_bytes, int dev_id);
    void runTestCreateNonceBytes(unsigned long long nonce, unsigned char* nonce_bytes, int dev_id);
    void runTestSealHash(unsigned char* seal, unsigned char* block_hash, uint64 nonce, int dev_id);
    void runTestPreSealHash(unsigned char* seal, unsigned char* preseal_bytes);
    void runTest(unsigned char* data, unsigned long size, unsigned char* digest);
    void runTestKeccak(unsigned char* data, unsigned long size, unsigned char* digest);
    uint64 solve_cuda_c(int blockSize, unsigned char* seal, uint64 nonce_start, uint64 update_interval, uint256 limit, unsigned char* block_bytes, int dev_id);
    void expose_cuda_errors();

cpdef void log_cuda_errors():
    expose_cuda_errors()

cpdef bytes run_test(unsigned char* data, unsigned long length): 
    cdef unsigned char* digest_ = <unsigned char*> PyMem_Malloc(
        64 * sizeof(unsigned char))

    cdef unsigned long size = sizeof(unsigned char) * length

    try:
        runTest(data, size, digest_)

        return digest_[:32]
    finally:
        PyMem_Free(digest_)

cpdef bytes run_test_keccak(unsigned char* data, unsigned int length): 
    cdef unsigned char* digest_ = <unsigned char*> PyMem_Malloc(
        64 * sizeof(unsigned char))

    cdef unsigned long size = sizeof(unsigned char) * length

    try:
        runTestKeccak(data, size, digest_)

        return digest_[:32]
    finally:
        PyMem_Free(digest_)

cpdef bytes run_test_seal_hash(unsigned char* block_bytes, uint64 nonce, int dev_id):
    cdef unsigned char* digest_ = <unsigned char*> PyMem_Malloc(
        64 * sizeof(unsigned char))

    try:
        runTestSealHash(digest_, block_bytes, nonce, dev_id)

        return digest_[:32]
    finally:
        PyMem_Free(digest_)

cpdef bytes run_test_preseal_hash(unsigned char* preseal_bytes):
    cdef unsigned char* digest_ = <unsigned char*> PyMem_Malloc(
        64 * sizeof(unsigned char))

    try:
        runTestPreSealHash(digest_, preseal_bytes)

        return digest_[:32]
    finally:
        PyMem_Free(digest_)

cpdef bytes run_test_create_nonce_bytes(uint64 nonce, int dev_id):
    cdef unsigned char* nonce_bytes = <unsigned char*> PyMem_Malloc(
        8 * sizeof(unsigned char))
    cdef int i

    try:
        runTestCreateNonceBytes(nonce, nonce_bytes, dev_id)
        
        # Convert digest to python string
        nonce_bytes_str = nonce_bytes

        return nonce_bytes[:8]
    finally:
        PyMem_Free(nonce_bytes)

cpdef int run_test_less_than(const unsigned char[:] a, const unsigned char[:] b):
    cdef unsigned char* a_char = <unsigned char*> PyMem_Malloc(
        32 * sizeof(unsigned char))
        
    cdef unsigned char* b_char = <unsigned char*> PyMem_Malloc(
        32 * sizeof(unsigned char))

    cdef int result

    for i in range(32):
        a_char[i] = a[i]

    for i in range(32):
        b_char[i] = b[i]

    try:
        result = runTestLessThan(<unsigned long*>a_char, <unsigned long*>b_char)
        return result
    finally:
        PyMem_Free(a_char)
        PyMem_Free(b_char)

cpdef int run_test_seal_meets_difficulty(const unsigned char[:] seal, const unsigned char[:] upper):
    cdef unsigned char* limit_char = <unsigned char*> PyMem_Malloc(
        32 * sizeof(unsigned char))

    cdef unsigned char* seal_ = <unsigned char*> PyMem_Malloc(
        32 * sizeof(unsigned char))

    cdef int i
    for i in range(32):
        limit_char[i] = upper[i]

    for i in range(32):
        seal_[i] = seal[i]

    try:
        return runTestSealMeetsDifficulty(seal_, <unsigned long*>limit_char)
    finally:
        PyMem_Free(seal_)
        PyMem_Free(limit_char)

cpdef bytearray run_test_create_pre_seal(uint64 nonce, unsigned char* block_bytes, int dev_id):
    cdef unsigned char* preseal_bytes = <unsigned char*> PyMem_Malloc(
        40 * sizeof(unsigned char))
    cdef int i

    try:
        runTestCreatePreSeal(preseal_bytes, nonce, block_bytes, dev_id)

        return bytearray(preseal_bytes[:40])
    finally:
        PyMem_Free(preseal_bytes)

cpdef int128 solve_cuda(int blockSize, uint64 nonce_start, uint64 update_interval, const unsigned char[:] limit, const unsigned char[:] block_bytes, const int dev_id):
    cdef uint64 solution
    cdef int128 solution_128

    cdef unsigned char* block_bytes_c = <unsigned char*> PyMem_Malloc(
        64 * sizeof(unsigned char))

    cdef unsigned char* limit_char = <unsigned char*> PyMem_Malloc(
        32 * sizeof(unsigned char))

    cdef unsigned int i

    for i in range(64):
        block_bytes_c[i] = block_bytes[i]
    
    for i in range(32):
        limit_char[i] = limit[i]

    try:
        solution = solve_cuda_c(blockSize, NULL, nonce_start, update_interval, <unsigned long*>limit_char, block_bytes_c, dev_id);
        solution_128 = solution
        return solution_128 - 1
    finally:  
        PyMem_Free(block_bytes_c)
        PyMem_Free(limit_char)

cpdef void reset_cuda():
    reset_cuda_c()