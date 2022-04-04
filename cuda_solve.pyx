#!python
#cython: language_level=3

from cpython cimport array
from cpython.mem cimport PyMem_Malloc, PyMem_Free
from libc.string cimport memcpy
from libc.stdio cimport printf
import array

cdef extern from "src/uint128.hh":
    ctypedef int int128

cdef extern from "src/uint64.cuh":
    ctypedef unsigned long long uint64;

cdef extern from "src/uint256.cuh":    
    ctypedef unsigned long uint256[8]

cdef extern from "src/main.hh":
    int runTestLessThan(uint256 a, uint256 b);
    void runTestCreatePreSeal(unsigned char* pre_seal, uint64 nonce, unsigned char* block_bytes);
    void runTestCreateNonceBytes(uint64 nonce, unsigned char* nonce_bytes);
    void runTestSealHash(unsigned char* seal, unsigned char* block_hash, uint64 nonce);
    void runTestPreSealHash(unsigned char* seal, unsigned char* preseal_bytes);
    void runTest(unsigned char* data, unsigned long size, unsigned char* digest);
    uint64 solve_cuda_c(int blockSize, unsigned char* seal, uint64* nonce_start, uint64 update_interval, unsigned int n_nonces, uint256 limit, unsigned char* block_bytes);

cpdef bytes run_test(unsigned char* data, unsigned long length): 
    cdef unsigned char* digest_ = <unsigned char*> PyMem_Malloc(
        64 * sizeof(unsigned char))

    cdef unsigned long size = sizeof(unsigned char) * length

    try:
        runTest(data, size, digest_)

        return digest_[:32]
    finally:
        PyMem_Free(digest_)

cpdef bytes run_test_seal_hash(unsigned char* block_bytes, uint64 nonce):
    cdef unsigned char* digest_ = <unsigned char*> PyMem_Malloc(
        64 * sizeof(unsigned char))

    try:
        runTestSealHash(digest_, block_bytes, nonce)

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

cpdef bytes run_test_create_nonce_bytes(uint64 nonce):
    cdef unsigned char* nonce_bytes = <unsigned char*> PyMem_Malloc(
        8 * sizeof(unsigned char))
    cdef int i

    try:
        runTestCreateNonceBytes(nonce, nonce_bytes)
        
        # Convert digest to python string
        nonce_bytes_str = nonce_bytes

        return nonce_bytes[:8]
    finally:
        PyMem_Free(nonce_bytes)

cpdef int run_test_less_than(const unsigned char[:] a, const unsigned char[:] b):
    cdef unsigned long* a_ = <unsigned long*> PyMem_Malloc(
        8 * sizeof(unsigned long))

    cdef unsigned char* a_char = <unsigned char*> PyMem_Malloc(
        32 * sizeof(unsigned char))

    cdef unsigned long* b_ = <unsigned long*> PyMem_Malloc(
        8 * sizeof(unsigned long))

    cdef unsigned char* b_char = <unsigned char*> PyMem_Malloc(
        32 * sizeof(unsigned char))

    cdef int result

    for i in range(32):
        a_char[i] = a[i]

    # Note sure if this will work
    memcpy(a_, a_char , 8 * sizeof(unsigned long))

    for i in range(32):
        b_char[i] = b[i]

    # Note sure if this will work
    memcpy(b_, b_char , 8 * sizeof(unsigned long))

    try:
        result = runTestLessThan(a_, b_)
        return result
    finally:
        PyMem_Free(a_)
        PyMem_Free(b_)
        PyMem_Free(a_char)
        PyMem_Free(b_char)

cpdef bytearray run_test_create_pre_seal(uint64 nonce, unsigned char* block_bytes):
    cdef unsigned char* preseal_bytes = <unsigned char*> PyMem_Malloc(
        40 * sizeof(unsigned char))
    cdef int i

    try:
        runTestCreatePreSeal(preseal_bytes, nonce, block_bytes)

        return bytearray(preseal_bytes[:40])
    finally:
        PyMem_Free(preseal_bytes)

cpdef tuple solve_cuda(int blockSize, list nonce_start, uint64 update_interval, unsigned int n_nonces, const unsigned char[:] limit, const unsigned char[:] block_bytes):
    cdef uint64 solution
    cdef int128 solution_128

    cdef uint64* nonce_start_c = <uint64*> PyMem_Malloc(
        blockSize * sizeof(uint64))

    cdef unsigned char* block_bytes_c = <unsigned char*> PyMem_Malloc(
        64 * sizeof(unsigned char))
    
    cdef unsigned char* seal_ = <unsigned char*> PyMem_Malloc(
        64 * sizeof(unsigned char))

    cdef unsigned long* limit_ = <unsigned long*> PyMem_Malloc(
        8 * sizeof(unsigned long))

    cdef unsigned char* limit_char = <unsigned char*> PyMem_Malloc(
        32 * sizeof(unsigned char))

    cdef unsigned int i

    for i in range(n_nonces):
        nonce_start_c[i] = nonce_start[i]

    for i in range(64):
        block_bytes_c[i] = block_bytes[i]
    
    for i in range(32):
        limit_char[i] = limit[i]

    # Note sure if this will work
    memcpy(limit_, limit_char , 8 * sizeof(unsigned long))

    try:
        solution = solve_cuda_c(blockSize, seal_, nonce_start_c, update_interval, n_nonces, limit_, block_bytes_c);
        solution_128 = solution
        return (solution_128 - 1, bytearray(seal_[:32]))
    finally:
        PyMem_Free(nonce_start_c)    
        PyMem_Free(block_bytes_c)
        PyMem_Free(seal_)
        PyMem_Free(limit_)
        PyMem_Free(limit_char)