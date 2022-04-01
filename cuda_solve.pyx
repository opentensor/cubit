from cpython cimport array
from cpython.mem cimport PyMem_Malloc, PyMem_Free
import array


cdef extern from "src/uint256.cuh":    
    ctypedef unsigned long uint256[8]

cdef extern from "src/main.h":
    cdef unsigned long solve_cuda_c(int blockSize, unsigned char* seal, unsigned int* nonce_start, unsigned int update_interval, unsigned int n_nonces, uint256 limit, unsigned char* block_bytes)

cpdef tuple solve_cuda(int blockSize, list nonce_start, unsigned int update_interval, unsigned int n_nonces, unsigned long long difficulty, unsigned long[:] limit, unsigned char[:] block_bytes):
    cdef unsigned char seal[64]
    
    cdef unsigned long solution

    cdef unsigned int* nonce_start_c = <unsigned int*> PyMem_Malloc(
        blockSize * sizeof(unsigned int))

    cdef unsigned char* block_bytes_c = <unsigned char*> PyMem_Malloc(
        64 * sizeof(unsigned char))
    
    cdef unsigned char* seal_ = <unsigned char*> PyMem_Malloc(
        64 * sizeof(unsigned char))

    cdef uint256 limit_;

    limit_[0] = limit[0]
    limit_[1] = limit[1]
    limit_[2] = limit[2]
    limit_[3] = limit[3]
    limit_[4] = limit[4]
    limit_[5] = limit[5]
    limit_[6] = limit[6]
    limit_[7] = limit[7]

    cdef int i
    for i in range(n_nonces):
        nonce_start_c[i] = nonce_start[i]

    for i in range(64):
        block_bytes_c[i] = block_bytes[i]
    try:
        solution = solve_cuda_c(blockSize, seal_, nonce_start_c, update_interval, n_nonces, limit_, block_bytes_c);
        seal = array.array('B', seal_)
        
        return (solution, seal)
    finally:
        PyMem_Free(nonce_start_c)
        PyMem_Free(block_bytes_c)
        PyMem_Free(seal_)