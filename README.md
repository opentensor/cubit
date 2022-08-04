# cubit

A python package to register on the bittensor network using a CUDA device.

## Requirements
    Cython
    bittensor@feature/cuda_solver  
    nvcc (cuda-11.3)
    gcc (9.3.1)
    sm_86 enabled CUDA GPU (30XX series, Axxxx series or higher)

## Install
Create install dir  
`mkdir cubit-tmp`  
Enter tmp dir  
`cd ./cubit-tmp`  
Download wheel `*.whl` from [releases](https://github.com/opentensor/cubit/releases/latest)  
`wget https://github.com/opentensor/cubit/releases/download/v1.0.3/cubit-1.0.3-cp310-cp310-linux_x86_64.whl`  
Install the wheel  
`pip install cubit-1.0.3-cp310-cp310-linux_x86_64.whl`  

#### Install testing dependencies
Install `test` extras from wheel  
`pip install cubit-1.0.3-cp310-cp310-linux_x86_64.whl[test]`  

### From source
Clone repo  
`git clone https://github.com/opentensor/cubit.git`  
Enter dir  
`cd cubit/`  
Install dev requirements  
`pip install -r requirements.dev.txt`  
Install as editable  
`pip install -e .`  

#### Install testing dependencies
Install `test` extras as editable   
`pip install -e .[test]`  
## Testing 
Testing uses unittest as there is an issue with pytest and Cython compatability

`python3 -m unittest test.py`

## TODO
- Speed-up transfer to host after finding a solution to POW
- Perhaps use events/streams

## Acknowledgments
  
https://github.com/rmcgibbo/npcuda-example/  
https://github.com/mochimodev/cuda-hashing-algos/  
https://github.com/camfairchild/bittensor_register_cuda/
