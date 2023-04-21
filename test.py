# The MIT License (MIT)
#
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

import binascii
import datetime
import hashlib
import math
import random
import unittest
from typing import List

import bittensor as bt
import torch
from Crypto.Hash import keccak

from cubit import (reset_cuda, run_test,
                                     run_test_create_nonce_bytes,
                                     run_test_create_pre_seal, run_test_keccak,
                                     run_test_less_than, run_test_preseal_hash,
                                     run_test_seal_hash,
                                     run_test_seal_meets_difficulty,
                                     solve_cuda)


class TestCli( unittest.TestCase ):
    st: bt.Subtensor
    bn: int
    bh: str
    difficulty: int
    limit: int
    upper: int
    upper_bytes: bytes
    block_bytes: bytes
    dev_id: int

    def setUp( self ) -> None:
        if not torch.cuda.is_available():
            print("No GPU available")
            self.fail("No GPU available")

        self.dev_id = 5 # By default, use the first GPU
        self.st = bt.subtensor(network="finney")
        self.bn = self.st.get_current_block()
        self.bh = self.st.substrate.get_block_hash(self.bn)
        self.difficulty = 1_000_000_000 #st.difficulty
        self.limit = int(math.pow(2,256)) - 1
        self.upper = int(self.limit // self.difficulty) - 1
        self.upper_bytes = self.upper.to_bytes(32, byteorder='little', signed=False)
        self.block_bytes = self.bh.encode('utf-8')[2:]

    def tearDown(self) -> None:
        reset_cuda() # Reset the CUDA device

    @staticmethod
    def hex_bytes_to_u8_list( hex_bytes:bytes ) -> List[int]:
        hex_chunks = [int(hex_bytes[i:i+2], 16) for i in range(0, len(hex_bytes), 2)]
        return hex_chunks

    @staticmethod
    def seal_meets_difficulty( seal:bytes, difficulty:int ) -> bool:
        seal_number = int.from_bytes(seal, "big")
        product = seal_number * difficulty
        limit = int(math.pow(2,256))- 1
        return product <= limit

    @staticmethod
    def get_nonce_bytes( nonce:int ) -> bytes:
        nonce_bytes = binascii.hexlify(nonce.to_bytes(8, 'little'))
        return nonce_bytes

    # Test sha256 implementation vs hashlib
    def test_sha_implementation( self ) -> None:
        print(self._testMethodName)
        test_input = bytes("test", 'utf-8')
        test_hash = run_test(test_input, len(test_input))
        compare_hash = hashlib.sha256( test_input ).digest()

        self.assertEqual(test_hash, compare_hash)
    
    def test_keccak_implementation ( self ) -> None:
        print(self._testMethodName)
        test_input = bytes("test", 'utf-8')
        test_hash = run_test_keccak(test_input, len(test_input))
        kec = keccak.new(digest_bits=256)
        compare_hash = kec.update( test_input ).digest()

        self.assertEqual(test_hash, compare_hash)

    # Test hash of formed preseal
    def test_preseal_hash( self ) -> None:
        print(self._testMethodName)
        nonce = 0
        nonce_bytes = self.get_nonce_bytes(nonce)
        pre_seal = nonce_bytes + self.block_bytes

        seal = run_test_preseal_hash(bytearray(self.hex_bytes_to_u8_list(pre_seal)))
        
        seal_sh256 = hashlib.sha256( bytearray(self.hex_bytes_to_u8_list(pre_seal)) ).digest()
        kec = keccak.new(digest_bits=256)
        seal_2 = kec.update( seal_sh256 ).digest()

        self.assertEqual(seal, seal_2)
    
    # Test create nonce bytes from nonce
    def test_create_nonce_bytes( self ) -> bytes:
        print(self._testMethodName)
        nonce = random.randint(int(math.pow(2, 45)), int(math.pow(2, 63)-1))
        nonce_bytes: bytes = run_test_create_nonce_bytes(nonce, self.dev_id)
        nonce_bytes_2 = self.get_nonce_bytes(nonce)
        # Unhexlify to compare
        nonce_bytes_2 = binascii.unhexlify(nonce_bytes_2)

        self.assertEqual(nonce_bytes, nonce_bytes_2)
    
    # Test create pre seal
    def test_create_pre_seal( self ) -> bytes:
        print(self._testMethodName)
        nonce = random.randint(int(math.pow(2, 45)), int(math.pow(2, 63)-1))
        pre_seal = run_test_create_pre_seal(nonce, self.block_bytes, self.dev_id)

        nonce_bytes = self.get_nonce_bytes(nonce)
        pre_seal_2 = bytearray(self.hex_bytes_to_u8_list(nonce_bytes + self.block_bytes))

        self.assertEqual(pre_seal, pre_seal_2)

    # Test create pre seal
    def test_create_new_pre_seal( self ) -> bytes:
        print(self._testMethodName)
        nonce = random.randint(int(math.pow(2, 45)), int(math.pow(2, 63)-1))
        kec = keccak.new(digest_bits=512)
        block_and_hotkey_hash_bytes = kec.update( self.block_bytes ).digest()
        block_and_hotkey_hash_hex = binascii.hexlify(block_and_hotkey_hash_bytes)[:64]

        pre_seal = run_test_create_pre_seal(nonce, block_and_hotkey_hash_hex, self.dev_id)

        nonce_bytes = self.get_nonce_bytes(nonce)
        pre_seal_2 = bytearray(self.hex_bytes_to_u8_list(nonce_bytes + block_and_hotkey_hash_hex))

        self.assertEqual(pre_seal, pre_seal_2)

    def test_new_seal_hash( self ) -> None:
        nonce = random.randint(int(math.pow(2, 45)), int(math.pow(2, 63)-1))
        kec = keccak.new(digest_bits=512)
        block_and_hotkey_hash_bytes = kec.update( self.block_bytes ).digest()
        block_and_hotkey_hash_hex = binascii.hexlify(block_and_hotkey_hash_bytes)[:64]

        seal = run_test_seal_hash(block_and_hotkey_hash_hex, nonce, self.dev_id)

        nonce_bytes = binascii.hexlify(nonce.to_bytes(8, 'little'))
        pre_seal = nonce_bytes + block_and_hotkey_hash_hex
        seal_sh256 = hashlib.sha256( bytearray(self.hex_bytes_to_u8_list(pre_seal)) ).digest()
        kec = keccak.new(digest_bits=256)
        seal_2 = kec.update( seal_sh256 ).digest()

        self.assertEqual(seal, seal_2)
    
    # Test block and nonce hash
    def test_seal_hash( self ) -> None:
        print(self._testMethodName)
        nonce = random.randint(int(math.pow(2, 45)), int(math.pow(2, 63)-1))
        seal = run_test_seal_hash(self.block_bytes, nonce, self.dev_id)
        nonce_bytes = self.get_nonce_bytes(nonce)
        pre_seal = nonce_bytes + self.block_bytes

        seal_sh256 = hashlib.sha256( bytearray(self.hex_bytes_to_u8_list(pre_seal)) ).digest()
        kec = keccak.new(digest_bits=256)
        seal_2 = kec.update( seal_sh256 ).digest()

        self.assertEqual(seal, seal_2)
    
    # Test less than
    def test_less_than( self ) -> None:
        print(self._testMethodName)
        for _ in range(0, 100):
            a = random.randint(0, 20000000000000000000000)
            b = random.randint(0, 20000000000000000000000)
            a_ = a.to_bytes(32, byteorder='little', signed=False)
            b_ = b.to_bytes(32, byteorder='little', signed=False)
            result = run_test_less_than(a_, b_)
            self.assertEqual(a < b, result == -1, f"{a} is{' not' if (a >= b) else ''} less than {b}") # 0 means a < b

    # Test seal meets difficulty
    def test_seal_meets_difficulty( self ) -> None:
        print(self._testMethodName)
        difficulty = 48 * 10**9
        upper = int(self.limit // difficulty)
        upper_bytes = upper.to_bytes(32, byteorder='little', signed=False)

        nonce = 0
        nonce_bytes = self.get_nonce_bytes(nonce)
        pre_seal = nonce_bytes + self.block_bytes

        seal_sh256 = hashlib.sha256( bytearray(self.hex_bytes_to_u8_list(pre_seal)) ).digest()
        kec = keccak.new(digest_bits=256)
        seal = kec.update( seal_sh256 ).digest()

        result = run_test_seal_meets_difficulty(seal, upper_bytes)
        result_2 = self.seal_meets_difficulty(seal, difficulty)

        self.assertEqual(result, result_2)
    
    # Test a solve
    def test_solve( self ) -> None:
        print(self._testMethodName)
        solution = -1
        interval = 30_000
        start_nonce = 0
        time_start = datetime.datetime.now()
        TPB = 256
        while solution == -1:
            # int blockSize, uint64 nonce_start, uint64 update_interval, const unsigned char[:] limit,
            # const unsigned char[:] block_bytes
            solution = solve_cuda(TPB, start_nonce, interval, self.upper_bytes, self.block_bytes, self.dev_id)
            start_nonce += interval * TPB

        self.assertNotEqual(solution, -1)
        time_end = datetime.datetime.now()

        time_diff = time_end - time_start
        print(f"Solve took: {time_diff.total_seconds()} seconds")
        
        seal_sh256 = hashlib.sha256( bytearray(self.hex_bytes_to_u8_list(self.get_nonce_bytes(solution) + self.block_bytes)) ).digest()
        kec = keccak.new(digest_bits=256)
        seal = kec.update( seal_sh256 ).digest()

        seal_number = int.from_bytes(seal, "big")
        limit = int(math.pow(2,256)) - 1
        product = seal_number * self.difficulty

        self.assertTrue(
            product < limit, # self.seal_meets_difficulty(seal, self.difficulty)
            f"solution: {solution} with seal: 0x{seal.hex()} for block_num: {self.bn} \ndoes not meet difficulty {self.difficulty} for block hash: {self.bh}"
        )
