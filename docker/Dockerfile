FROM nvidia/cuda:9.0-cudnn7-runtime


RUN apt-get update
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends apt-utils

RUN apt-get update && apt-get install -y --no-install-recommends \
wget curl xz-utils build-essential zlib1g-dev zip unzip \
libreadline-gplv2-dev libncursesw5-dev libssl-dev libosmesa6-dev \
libsqlite3-dev tk-dev libgdbm-dev libc6-dev libbz2-dev \
libgl1-mesa-glx libgl1-mesa-dev patchelf libglfw3-dev \
libsqlite3-dev libpng-dev libjpeg-dev git-all && \
rm -rf /var/lib/apt/lists/*

RUN wget https://www.python.org/ftp/python/3.6.10/Python-3.6.10.tar.xz && \
tar -xJf Python-3.6.10.tar.xz && cd Python-3.6.10 && \
./configure && make && make install && \
pip3 install --upgrade pip wheel setuptools \
numpy pandas cffi Cython lockfile glfw imageio

RUN mkdir /root/.mujoco && cd /root/.mujoco && \
wget https://www.roboti.us/download/mjpro150_linux.zip && \
unzip mjpro150_linux.zip && \
wget https://www.roboti.us/file/mjkey.txt

ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/.mujoco/mjpro150/bin

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

RUN which python3
