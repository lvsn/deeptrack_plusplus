FROM nvidia/cuda:8.0-cudnn7-devel
MAINTAINER Mathieu Garon <mathieugaron91@gmail.com>

ENV PATH /usr/local/bin:$PATH
ENV LANG C.UTF-8

RUN apt-get update && \
    apt-get install -y \
            curl wget vim \
            software-properties-common ipython3 \
            openexr libopenexr-dev zlib1g-dev \
            libtiff5-dev libjpeg8-dev zlib1g-dev libfreetype6-dev liblcms2-dev libwebp-dev \
            libffi-dev \
            libhdf5-dev \
            pkg-config \
            bzip2 git gfortran sudo && \
    apt-get -y autoremove && \
    rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/xianyi/OpenBLAS && \
    cd OpenBLAS && \
    make FC=gfortran && \
    make PREFIX=/usr install

# Setup python
RUN cd /root/ && wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh

ENV PATH /opt/conda/bin:$PATH

# Setup Numpy with OpenBLAS
RUN cd / && conda install cython
RUN cd ~ && git clone https://github.com/numpy/numpy && cd ~/numpy && git checkout v1.11.2
ADD site.cfg /root/numpy/site.cfg
RUN cd ~/numpy && python3 setup.py build --fcompiler=gnu95 && python3 setup.py install

# Setup Scipy & cie
RUN cd / && conda install pillow scipy
RUN cd / && conda install scikit-learn scikit-image matplotlib
RUN cd / && conda install tqdm sympy requests cffi h5py
RUN cd / && conda install -c menpo opencv
RUN pip install plyfile
RUN cd / && conda install pyyaml

# PyTorch
RUN conda install pytorch=0.4.1 torchvision cuda80 -c pytorch

# pytorch toolkit
RUN git clone https://github.com/MathGaron/pytorch_toolbox && \
    cd pytorch_toolbox && \
    python setup.py install

# Configuration
ADD .bashrc /root/.bashrc
ADD .vimrc /root/.vimrc

RUN apt-get update && \
    apt-get install -y \
            tmux htop strace ltrace \
            gdb python3-dbg && \
    apt-get -y autoremove && \
    rm -rf /var/lib/apt/lists/*

RUN chsh -s /bin/bash
WORKDIR /root
