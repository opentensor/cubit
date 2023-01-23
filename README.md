# cubit

A python package to register on the bittensor network using a CUDA device.

## Requirements
- Ubuntu 20.04 or higher  
- bittensor>=3.0.0  
- sm_61, sm_70, sm_75, sm_80, or sm_86 enabled CUDA GPU (See [here](https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/))

## Building docker image

1. Change the base image to the target bittensor version
1. Build the image:
    1. `sudo docker build -t opentensorfdn/bittensor:VERSION_BT-cubitVERSION_CUBIT -f ./docker/Dockerfile .`
1. Push the image
    1. `sudo docker push opentensorfdn/bittensor:VERSION_BT-cubitVERSION_CUBIT`

## Install
Using the wheel for your version of python (3.7, 3.8, etc.) from [releases](https://github.com/opentensor/cubit/releases/latest)  

For Python 3.7  
```
pip install https://github.com/opentensor/cubit/releases/download/v1.1.2/cubit-1.1.2-cp37-cp37m-linux_x86_64.whl
```   
For Python 3.8  
```
pip install https://github.com/opentensor/cubit/releases/download/v1.1.2/cubit-1.1.2-cp38-cp38-linux_x86_64.whl
```   
For Python 3.9  
```
pip install https://github.com/opentensor/cubit/releases/download/v1.1.2/cubit-1.1.2-cp39-cp39-linux_x86_64.whl
```   
For Python 3.9  
```
pip install https://github.com/opentensor/cubit/releases/download/v1.1.2/cubit-1.1.2-cp310-cp310-linux_x86_64.whl
```   
#### Install testing dependencies
Install `test` extras from wheel
For Python 3.7  
```
pip install https://github.com/opentensor/cubit/releases/download/v1.1.2/cubit-1.1.2-cp37-cp37m-linux_x86_64.whl[test]
```   
For Python 3.8  
```
pip install https://github.com/opentensor/cubit/releases/download/v1.1.2/cubit-1.1.2-cp38-cp38-linux_x86_64.whl[test]
```   
For Python 3.9  
```
pip install https://github.com/opentensor/cubit/releases/download/v1.1.2/cubit-1.1.2-cp39-cp39-linux_x86_64.whl[test]
```   
For Python 3.10  
```
pip install https://github.com/opentensor/cubit/releases/download/v1.1.2/cubit-1.1.2-cp310-cp310-linux_x86_64.whl[test]
```   
### From source
#### Requirements   
- [cuda-toolkit 11.3 or higher](https://developer.nvidia.com/cuda-downloads)
    - nvcc
- gcc (9.3.1 or higher)
- python 3.7 or higher  
    
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
