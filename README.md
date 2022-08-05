# cubit

A python package to register on the bittensor network using a CUDA device.

## Requirements
- Ubuntu 20.04 or higher  
- bittensor@feature/cuda_solver  
- sm_86 enabled CUDA GPU (30XX series, Axxxx series or higher)

## Install
Using the wheel for your version of python (3.9, 3.10) from [releases](https://github.com/opentensor/cubit/releases/latest)  
For Python 3.9  
```
pip install https://github.com/opentensor/cubit/releases/download/v1.0.5/cubit-1.0.5-cp39-cp39-linux_x86_64.whl
```   
For Python 3.10  
```
pip install https://github.com/opentensor/cubit/releases/download/v1.0.5/cubit-1.0.5-cp310-cp310-linux_x86_64.whl
```    

#### Install testing dependencies
Install `test` extras from wheel
Python 3.9
```
pip install https://github.com/opentensor/cubit/releases/download/v1.0.5/cubit-1.0.5-cp39-cp39-linux_x86_64.whl[test]
``` 

Python 3.10
```
pip install https://github.com/opentensor/cubit/releases/download/v1.0.5/cubit-1.0.5-cp310-cp310-linux_x86_64.whl[test]
``` 

### From source
#### Requirements   
- [cuda-toolkit 11.3 or higher](https://developer.nvidia.com/cuda-downloads)
    - nvcc
- gcc (9.3.1 or higher)
- python 3.9 or higher  
    
You can check if you have cuda-toolkit with 
```
nvcc --version
```  


Clone repo  
```
git clone https://github.com/opentensor/cubit.git
```  
Enter dir  
```
cd cubit/
```  
Install dev requirements  
```
pip install -r requirements.dev.txt
```  
Install as editable  
```
pip install -e .
```  

#### Install testing dependencies
Install `test` extras as editable   
```
pip install -e .[test]
```  
## Unit Testing 
Testing uses unittest as there is an issue with pytest and Cython compatability

```
python3 -m unittest test.py
```  

## Acknowledgments
  
https://github.com/rmcgibbo/npcuda-example/  
https://github.com/mochimodev/cuda-hashing-algos/  
https://github.com/camfairchild/bittensor_register_cuda/
