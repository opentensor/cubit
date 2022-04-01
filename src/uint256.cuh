// Copyright (c) 2009-2010 Satoshi Nakamoto
// Copyright (c) 2009-2019 The Bitcoin Core developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.
#ifndef UINT256_CUH
#define UINT256_CUH

typedef unsigned long uint256[8];
__device__ int lt(uint256 a, uint256 b);

#endif // UINT256_CUH