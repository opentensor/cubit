from cpython cimport array
from cpython.mem cimport PyMem_Malloc, PyMem_Free
from libc.string cimport memcpy
from libc.stdio cimport printf
import array


cdef extern from "src/uint256.cuh":    
    ctypedef unsigned long uint256[8]

cdef extern from "src/main.hh":
    void runTestSealHash(unsigned char* seal, unsigned char* block_hash, unsigned long nonce);
    void runTestPreSealHash(unsigned char* seal, unsigned char* preseal_bytes);
    void runTest(unsigned char* data, unsigned long size, unsigned char* digest);
    unsigned long solve_cuda_c(int blockSize, unsigned char* seal, unsigned long* nonce_start, unsigned long update_interval, unsigned int n_nonces, uint256 limit, unsigned char* block_bytes);

cpdef bytes run_test(unsigned char* data, unsigned long length): 
    cdef unsigned char* digest_ = <unsigned char*> PyMem_Malloc(
        64 * sizeof(unsigned char))

    cdef unsigned long size = sizeof(unsigned char) * length
    cdef bytes digest_str

    try:
        runTest(data, size, digest_)
        # Convert digest to python string
        digest_str = digest_

        return digest_str
    finally:
        PyMem_Free(digest_)

cpdef bytes run_test_seal_hash(unsigned char* block_bytes, unsigned long nonce):
    cdef unsigned char* digest_ = <unsigned char*> PyMem_Malloc(
        64 * sizeof(unsigned char))

    cdef bytes digest_str

    try:
        runTestSealHash(digest_, block_bytes, nonce)
        # Convert digest to python string
        digest_str = digest_

        return digest_str
    finally:
        PyMem_Free(digest_)

cpdef bytes run_test_preseal_hash(unsigned char* preseal_bytes):
    cdef unsigned char* digest_ = <unsigned char*> PyMem_Malloc(
        64 * sizeof(unsigned char))

    cdef bytes digest_str

    try:
        runTestPreSealHash(digest_, preseal_bytes)
        # Convert digest to python string
        digest_str = digest_

        return digest_str
    finally:
        PyMem_Free(digest_)

cpdef tuple solve_cuda(int blockSize, list nonce_start, unsigned int update_interval, unsigned int n_nonces, unsigned long long difficulty, const unsigned char[:] limit, const unsigned char[:] block_bytes):
    cdef bytes seal
    cdef unsigned long solution

    cdef unsigned long* nonce_start_c = <unsigned long*> PyMem_Malloc(
        blockSize * sizeof(unsigned long))

    cdef unsigned char* block_bytes_c = <unsigned char*> PyMem_Malloc(
        32 * sizeof(unsigned char))
    
    cdef unsigned char* seal_ = <unsigned char*> PyMem_Malloc(
        64 * sizeof(unsigned char))

    cdef unsigned long* limit_ = <unsigned long*> PyMem_Malloc(
        8 * sizeof(unsigned long))

    cdef unsigned char* limit_char = <unsigned char*> PyMem_Malloc(
        32 * sizeof(unsigned char))

    cdef unsigned int i

    for i in range(n_nonces):
        nonce_start_c[i] = nonce_start[i]

    for i in range(32):
        block_bytes_c[i] = block_bytes[i]
    
    for i in range(32):
        limit_char[i] = limit[i]

    # Note sure if this will work
    memcpy(limit_, limit_char , 8 * sizeof(unsigned long))

    try:
        solution = solve_cuda_c(blockSize, seal_, nonce_start_c, update_interval, n_nonces, limit_, block_bytes_c);
        seal = seal_
        return (solution, seal)
    finally:
        PyMem_Free(nonce_start_c)
        PyMem_Free(block_bytes_c)
        PyMem_Free(seal_)
        PyMem_Free(limit_)
        PyMem_Free(limit_char)