# Bittensor Register CUDA

Allows for CUDA registration for bittensor using python.

## Requirements
    Cython
    bittensor@feature/cuda_solver  
    nvcc (cuda-11.6)
    gcc (9.4.0)
    sm_86 enabled CUDA GPU (30XX series, Axxxx series or higher)

## Install
`pip install -e .`
## Testing 
Testing uses unittest as there is an issue with pytest and Cython

`python3 -m unittest test.py`

## TODO
- Speed-up transfer to host after finding a solution to POW
- Perhaps use events/streams

## Acknowledgments
  
https://github.com/rmcgibbo/npcuda-example/  
https://github.com/mochimodev/cuda-hashing-algos/  
https://github.com/camfairchild/bittensor_register_cuda/
