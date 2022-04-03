import struct
from typing import List
from numpy import block, byte, diff
from bittensor_register_cuda import solve_cuda, run_test, run_test_seal_hash, run_test_preseal_hash, run_test_create_nonce_bytes, run_test_create_pre_seal
import math
import hashlib
import binascii
import bittensor as bt

def hex_bytes_to_u8_list( hex_bytes: bytes ):
    hex_chunks = [int(hex_bytes[i:i+2], 16) for i in range(0, len(hex_bytes), 2)]
    return hex_chunks

def seal_meets_difficulty( seal:bytes, difficulty:int ):
    seal_number = int.from_bytes(seal, "big")
    product = seal_number * difficulty
    limit = int(math.pow(2,256))- 1    
    if product > limit:
        return False
    else:
        return True

st = bt.subtensor(network="endpoint",chain_endpoint="subtensor.fairchild.dev:9944")
bn = st.get_current_block()
bh = st.substrate.get_block_hash(bn)
difficulty = st.difficulty
limit = int(math.pow(2,256)) - 1
upper = int(limit // difficulty)
print(limit, difficulty, upper)

upper_bytes = upper.to_bytes(32, byteorder='little', signed=False)
block_bytes = bh.encode('utf-8')[2:]

# Test sha256 implementation vs hashlib
test_input = bytes("test", 'utf-8')
test_hash = run_test(test_input, len(test_input))
compare_hash = hashlib.sha256( test_input ).digest()
print(test_hash, "\n", compare_hash, "\n", test_hash == compare_hash)

# Test hash of formed preseal
nonce = 0
nonce_bytes = binascii.hexlify(nonce.to_bytes(8, 'little'))
pre_seal = nonce_bytes + block_bytes

seal = run_test_preseal_hash(bytearray(hex_bytes_to_u8_list(pre_seal)))
seal_2 = hashlib.sha256( bytearray(hex_bytes_to_u8_list(pre_seal)) ).digest()
print(seal, "\n", seal_2, "\n", seal == seal_2)

#solution, seal = solve_cuda(4, [0, 20000, 40000, 60000, 80000], 20000, 5, difficulty, upper_bytes, block_bytes)

# Test create nonce bytes from nonce
nonce = 100000000
nonce_bytes = run_test_create_nonce_bytes(nonce)

nonce_bytes_2 = nonce.to_bytes(8, 'little')
print(nonce_bytes, "\n", nonce_bytes_2, "\n", nonce_bytes == nonce_bytes_2)

# Test create pre seal
nonce = 1304006780
pre_seal = run_test_create_pre_seal(nonce, block_bytes)
nonce_bytes = binascii.hexlify(nonce.to_bytes(8, 'little'))
pre_seal_2 = bytearray(hex_bytes_to_u8_list(nonce_bytes + block_bytes))
print(pre_seal, "\n", pre_seal_2, "\n", pre_seal == pre_seal_2)

# Test block and nonce hash
nonce = 0
seal = run_test_seal_hash(block_bytes, 0)
nonce_bytes = binascii.hexlify(nonce.to_bytes(8, 'little'))
pre_seal = nonce_bytes + block_bytes
seal_2 = hashlib.sha256( bytearray(hex_bytes_to_u8_list(pre_seal)) ).digest()
print(seal, "\n", seal_2, "\n", seal == seal_2)

# Test a solve
solution, seal = solve_cuda(4, [0, 20000, 40000, 60000, 80000], 20000, 5, difficulty, upper_bytes, block_bytes)
print(seal_meets_difficulty(seal, difficulty))
print(solution, seal)


