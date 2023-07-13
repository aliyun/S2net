FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04

# update key, as nvidia will change this key non-scheduled
RUN cd ~ && \
    apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub

# use this line for better apt-get network performance; comment out if apt-get is slow
RUN export DEBIAN_FRONTEND=noninteractive && mv /etc/apt/sources.list /etc/apt/sources.list.back && echo "deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ bionic main restricted universe multiverse \n# deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ bionic main restricted universe multiverse \ndeb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ bionic-updates main restricted universe multiverse \n# deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ bionic-updates main restricted universe multiverse \ndeb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ bionic-backports main restricted universe multiverse \n# deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ bionic-backports main restricted universe multiverse \ndeb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ bionic-security main restricted universe multiverse \n# deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ bionic-security main restricted universe multiverse" >> /etc/apt/sources.list

# install some dependency
RUN export DEBIAN_FRONTEND=noninteractive && \
    apt-get update && \
    apt-get upgrade -y && \ 
    apt-get install -y libopencv-dev build-essential curl git vim wget zip bzip2 cmake libopenexr-dev libeigen3-dev apt-utils zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev graphviz

# install anaconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py37_4.12.0-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -tipsy && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

ENV PATH /opt/conda/bin:$PATH
ENV LD_LIBRARY_PATH /usr/local/cuda-10.2/lib64:/usr/local/cuda-10.2/extras/CUPTI/lib64:$LD_LIBRARY_PATH

# install torch
RUN conda update -n base -c defaults conda
RUN python -m pip install --upgrade pip
RUN pip3 install torch==1.8.2 torchvision==0.9.2 torchaudio==0.8.2 --extra-index-url https://download.pytorch.org/whl/lts/1.8/cu102

# install apex
RUN cd ~ && git clone https://github.com/NVIDIA/apex && \
    cd apex && \
    pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./ && \
    cd .. && rm -rf apex

# install some extra dependencies
RUN pip3 install scipy Pillow tensorboardX tqdm matplotlib pyquaternion pyrender opencv-python timm einops visdom plyfile openexr pytest
RUN pip3 install attrdict tensorboard open3d numba pyyaml attrdict termcolor scikit-image healpy yacs h5py
RUN cd ~ && git clone https://github.com/meder411/MappedConvolutions.git && \
    cd MappedConvolutions && cd package && \
    python setup.py install && \
    cd .. && rm -rf MappedConvolutions

