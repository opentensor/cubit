# syntax=docker/dockerfile:1
FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-devel
#nvidia key migration
RUN apt-key del 7fa2af80
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu2004/x86_64/7fa2af80.pub
# Update the base image
RUN apt update && apt upgrade -y
# Install dependencies
RUN apt install -y curl sudo nano git htop wget unzip python3-dev tmux
# Install bittensor
ADD install.sh /install_bittensor.sh
RUN chmod +x /install_bittensor.sh
RUN bash /install_bittensor.sh
RUN pip3 install Cython>=0.29.14 

COPY . /bittensor_register_cuda/
RUN cd /bittensor_register_cuda && pip3 install -e .