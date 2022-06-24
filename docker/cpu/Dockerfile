#Download base image ubuntu 20.04
FROM ubuntu:20.04

# LABEL about the custom image for `neuroengineering-tools`
LABEL maintainer="shandilya.shreyas@gmail.com"
LABEL version="0.0"
LABEL description="This is custom Docker Image for `neuroengineering-tools`."

# Disable Prompt During Packages Installation
ARG DEBIAN_FRONTEND=noninteractive

# Update Ubuntu Software repository
RUN apt update
RUN apt install -y --no-install-recommends wget \
        curl \
        xz-utils \
        build-essential \
        zlib1g-dev \
        zip \
        unzip \
        python3-pip python3-dev \
        libreadline-gplv2-dev \
        libncursesw5-dev \
        libssl-dev \
        libosmesa6-dev \
        libsqlite3-dev \
        tk-dev \
        libgdbm-dev \
        libc6-dev \
        libbz2-dev \
        libgl1-mesa-glx \
        libgl1-mesa-dev \
        patchelf \
        libglfw3-dev \
        libglew-dev \
        libsqlite3-dev \
        libpng-dev \
        libjpeg-dev \
        git-all \
        liblzma-dev \
        tzdata \
        software-properties-common \
        net-tools \
        vim \
        xpra \
        xserver-xorg-dev \
        libopengl0 \
        libosmesa6-dev \
        libgl1-mesa-glx \
        libglfw3 \
        libgl1-mesa-dev \
        libglew-dev \
        patchelf && \
    mkdir /root/.mujoco && \
    wget -q https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz -O mujoco.tar.gz && \
    tar -zxf mujoco.tar.gz -C "/root/.mujoco" && \
    rm mujoco.tar.gz && \
    echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/.mujoco/mujoco210/bin' >> ~/.bashrc && \
    echo "/root/.mujoco/mujoco210/bin" > /etc/ld.so.conf.d/mujoco_ld_lib_path.conf && \
    ldconfig && \
    pip3 install --upgrade pip && \
    pip install -U 'mujoco-py<2.2,>=2.1' 

COPY requirements.txt requirements.txt
RUN apt install -y swig cmake && pip install -r requirements.txt

RUN export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/.mujoco/mujoco210/bin && python3 -c "import mujoco_py"
RUN mkdir /neurorobotics
