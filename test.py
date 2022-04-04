import struct
from typing import List
from numpy import block, byte, diff
from bittensor_register_cuda import solve_cuda, run_test, run_test_seal_hash, run_test_preseal_hash, run_test_create_nonce_bytes, run_test_create_pre_seal, run_test_less_than, run_test_seal_meets_difficulty
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
    return product <= limit

st = bt.subtensor(network="endpoint",chain_endpoint="subtensor.fairchild.dev:9944")
bn = st.get_current_block()
bh = st.substrate.get_block_hash(bn)
difficulty = 100000 #st.difficulty
limit = int(math.pow(2,256)) - 1
upper = int(limit // difficulty)
#print(limit, difficulty, upper)

upper_bytes = upper.to_bytes(32, byteorder='little', signed=False)
block_bytes = bh.encode('utf-8')[2:]

# Test sha256 implementation vs hashlib
"""
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


# Test less than
import random
for _ in range(0, 100):
    a = random.randint(0, 20000000000000000000000)
    b = random.randint(0, 20000000000000000000000)
    a_ = a.to_bytes(32, byteorder='little', signed=False)
    b_ = b.to_bytes(32, byteorder='little', signed=False)
    result = run_test_less_than(a_, b_)
    if (a < b) is not (result == -1):
        print("\n", a, "\n", b, "\n", result)
        break
else:
    print("Test lt: PASS")


# Test seal meets difficulty
diff_ = 48 * 10**9
upper_ = int(limit // difficulty)

upper_bytes_ = upper.to_bytes(32, byteorder='little', signed=False)

nonce = 0
nonce_bytes = binascii.hexlify(nonce.to_bytes(8, 'little'))
pre_seal = nonce_bytes + block_bytes
seal = hashlib.sha256( bytearray(hex_bytes_to_u8_list(pre_seal)) ).digest()
result = run_test_seal_meets_difficulty(seal, upper_bytes_)
result_2 = seal_meets_difficulty(seal, diff_)
print("Test seal meets difficulty: ", result == result_2)
print("\n")
"""

# Test a solve

solution = -1
interval = 50000
start_nonce = 0
while solution == -1:
    nonces = [nonce for nonce in range(start_nonce, start_nonce+interval*4, interval)]
    start_nonce += interval*4
    solution, seal = solve_cuda(4, nonces, interval, 4, upper_bytes, block_bytes)
print(seal_meets_difficulty(seal, difficulty))

nonce_bytes = binascii.hexlify(solution.to_bytes(8, 'little'))
seal_1 = hashlib.sha256( bytearray(hex_bytes_to_u8_list(nonce_bytes + block_bytes)) ).digest()
print(solution, seal)
print(solution, seal_1)
print(seal_meets_difficulty(seal_1, difficulty))
check = run_test_seal_meets_difficulty(seal, upper_bytes)
print(check)


