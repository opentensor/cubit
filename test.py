from typing import List
import unittest
import random
import math
import hashlib
import binascii

import bittensor as bt
from numpy import byte

from bittensor_register_cuda import solve_cuda, run_test, run_test_seal_hash, \
    run_test_preseal_hash, run_test_create_nonce_bytes, run_test_create_pre_seal, \
        run_test_less_than, run_test_seal_meets_difficulty, reset_cuda

class TestCli( unittest.TestCase ):
    st: bt.Subtensor
    bn: int
    bh: str
    difficulty: int
    limit: int
    upper: int
    upper_bytes: bytes
    block_bytes: bytes

    def setUp( self ) -> None:
        self.st = bt.subtensor(network="endpoint",chain_endpoint="subtensor.fairchild.dev:9944")
        self.bn = self.st.get_current_block()
        self.bh = self.st.substrate.get_block_hash(self.bn)
        self.difficulty = 100000 #st.difficulty
        self.limit = int(math.pow(2,256)) - 1
        self.upper = int(self.limit // self.difficulty)
        self.upper_bytes = self.upper.to_bytes(32, byteorder='little', signed=False)
        self.block_bytes = self.bh.encode('utf-8')[2:]

    def tearDown(self) -> None:
        reset_cuda() # Reset the CUDA device
        return super().tearDown()

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
        test_input = bytes("test", 'utf-8')
        test_hash = run_test(test_input, len(test_input))
        compare_hash = hashlib.sha256( test_input ).digest()
        assert test_hash == compare_hash
        #print(test_hash, "\n", compare_hash, "\n", test_hash == compare_hash)

    # Test hash of formed preseal
    def test_preseal_hash( self ) -> None:
        nonce = 0
        nonce_bytes = self.get_nonce_bytes(nonce)
        pre_seal = nonce_bytes + self.block_bytes

        seal = run_test_preseal_hash(bytearray(self.hex_bytes_to_u8_list(pre_seal)))
        seal_2 = hashlib.sha256( bytearray(self.hex_bytes_to_u8_list(pre_seal)) ).digest()
        assert seal == seal_2
        #print(seal, "\n", seal_2, "\n", seal == seal_2)

    # Test create nonce bytes from nonce
    def test_create_nonce_bytes( self ) -> bytes:
        nonce = 100000000
        nonce_bytes: bytes = run_test_create_nonce_bytes(nonce)
        nonce_bytes_2 = self.get_nonce_bytes(nonce)
        # Unhexlify to compare
        nonce_bytes_2 = binascii.unhexlify(nonce_bytes_2)
        assert nonce_bytes == nonce_bytes_2

    # Test create pre seal
    def test_create_pre_seal( self ) -> bytes:
        nonce = 1304006780
        pre_seal = run_test_create_pre_seal(nonce, self.block_bytes)

        nonce_bytes = self.get_nonce_bytes(nonce)
        pre_seal_2 = bytearray(self.hex_bytes_to_u8_list(nonce_bytes + self.block_bytes))
        assert pre_seal == pre_seal_2
        #print(pre_seal, "\n", pre_seal_2, "\n", pre_seal == pre_seal_2)

    # Test block and nonce hash
    def test_seal_hash( self ) -> None:
        nonce = 0
        seal = run_test_seal_hash(self.block_bytes, nonce)
        nonce_bytes = self.get_nonce_bytes(nonce)
        pre_seal = nonce_bytes + self.block_bytes
        seal_2 = hashlib.sha256( bytearray(self.hex_bytes_to_u8_list(pre_seal)) ).digest()
        assert seal == seal_2
        #print(seal, "\n", seal_2, "\n", seal == seal_2)

    # Test less than
    def test_less_than( self ) -> None:
        for _ in range(0, 100):
            a = random.randint(0, 20000000000000000000000)
            b = random.randint(0, 20000000000000000000000)
            a_ = a.to_bytes(32, byteorder='little', signed=False)
            b_ = b.to_bytes(32, byteorder='little', signed=False)
            result = run_test_less_than(a_, b_)
            assert (a < b) is (result == -1) # -1 means a < b

    # Test seal meets difficulty
    def test_seal_meets_difficulty( self ) -> None:
        difficulty = 48 * 10**9
        upper = int(self.limit // difficulty)
        upper_bytes = upper.to_bytes(32, byteorder='little', signed=False)

        nonce = 0
        nonce_bytes = self.get_nonce_bytes(nonce)
        pre_seal = nonce_bytes + self.block_bytes
        seal = hashlib.sha256( bytearray(self.hex_bytes_to_u8_list(pre_seal)) ).digest()
        result = run_test_seal_meets_difficulty(seal, upper_bytes)
        result_2 = self.seal_meets_difficulty(seal, difficulty)
        assert result == result_2

    # Test a solve
    def test_solve( self ) -> None:
        solution = -1
        interval = 50000
        start_nonce = 0
        while solution == -1:
            start_nonce += interval*4
            # int blockSize, uint64 nonce_start, uint64 update_interval, const unsigned char[:] limit,
            # const unsigned char[:] block_bytes
            solution = solve_cuda(4, start_nonce, interval, self.upper_bytes, self.block_bytes)
        assert solution != -1
        seal = hashlib.sha256( bytearray(self.hex_bytes_to_u8_list(self.get_nonce_bytes(solution) + self.block_bytes)) ).digest()
        assert self.seal_meets_difficulty(seal, self.difficulty)