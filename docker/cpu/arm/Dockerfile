# Download base image ros:noetic (ubuntu 20.04 with ros noetic pre-installed)
FROM ros:noetic-ros-core-focal

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
	vim \
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
	cmake \
	swig \
	libgstreamer1.0-dev \
	libgstreamer-plugins-base1.0-dev \
	libgstreamer-plugins-bad1.0-dev \
	gstreamer1.0-plugins-base \
	gstreamer1.0-plugins-good \
	gstreamer1.0-plugins-bad \
	gstreamer1.0-plugins-ugly \
	gstreamer1.0-libav \
	gstreamer1.0-tools \
	gstreamer1.0-x \
	gstreamer1.0-alsa \
	gstreamer1.0-gl \
	gstreamer1.0-gtk3 \
	gstreamer1.0-qt5 \
	gstreamer1.0-pulseaudio \
        patchelf && \
    mkdir /root/.mujoco && \
    wget -q https://github.com/deepmind/mujoco/releases/download/2.3.3/mujoco-2.3.3-linux-aarch64.tar.gz -O mujoco.tar.gz && \
    tar -zxf mujoco.tar.gz -C "/root/.mujoco" && \
    rm mujoco.tar.gz && \
    echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/.mujoco/mujoco210/bin' >> ~/.bashrc && \
    echo "/root/.mujoco/mujoco210/bin" > /etc/ld.so.conf.d/mujoco_ld_lib_path.conf && \
    ldconfig && \
    pip3 install --upgrade pip && \
    pip install -U mujoco 

RUN export LD_LIBRARY_PATH=/root/.mujoco/mujoco210/bin:$LD_LIBRARY_PATH && python3 -c "import mujoco"

# Need to resolve all dependencies before gymnasium is installed.
RUN pip install -U gymnasium
