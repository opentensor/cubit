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
# Install nvm via curl/bash
RUN curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.1/install.sh | bash \
    && export NVM_DIR="$HOME/.nvm" \
    && [ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh" \
    && nvm install --lts \
    && npm install -g npm@latest \
    && npm install -g npm@8.5.5 \
    && npm install -g pm2
# Install bittensor
ADD install.sh /install_bittensor.sh
RUN chmod +x /install_bittensor.sh
RUN bash /install_bittensor.sh
RUN pip3 install torch==1.10+cu113 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip3 install Cython==0.29.14 

## NOTE: you need to clone the bittensor_register_cuda repo to /bittensor_register_cuda
COPY . /bittensor_register_cuda/
RUN cd /bittensor_register_cuda && pip3 install -e .