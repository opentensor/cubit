# Bittensor Register CUDA

Allows for CUDA registration for bittensor using python.

## Requirements
`Cython`  
`bittensor@feature/cuda_solver`  
`nvcc`
`gcc`
`sm_61` enabled CUDA GPU (10XX series or higher) 

## Install
`pip install -e ./bittensor_register_cuda`
## Testing 
Testing uses unittest as there is an issue with pytest and Cython

`python3 -m unittest test.py`

## TODO
- Investigate source of SEGFAULT
- Speed-up transfer to host after finding a solution
- Perhaps use events/streams

## Acknowledgments
  
https://github.com/rmcgibbo/npcuda-example/  
https://github.com/mochimodev/cuda-hashing-algos/  
