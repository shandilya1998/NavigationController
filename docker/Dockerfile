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

# Installs google cloud sdk, this is mostly for using gsutil to export model.
RUN wget -nv \
https://dl.google.com/dl/cloudsdk/release/google-cloud-sdk.tar.gz && \
mkdir /root/tools && \
tar xvzf google-cloud-sdk.tar.gz -C /root/tools && \
rm google-cloud-sdk.tar.gz && \
/root/tools/google-cloud-sdk/install.sh --usage-reporting=false \
--path-update=false --bash-completion=false \
--disable-installation-options && \
rm -rf /root/.config/* && \
ln -s /root/.config /config && \
rm -rf /root/tools/google-cloud-sdk/.install/.backup

# Path configuration
ENV PATH $PATH:/root/tools/google-cloud-sdk/bin
# Make sure gsutil will use the default service account
RUN echo '[GoogleCompute]\nservice_account = default' > /etc/boto.cfg

RUN pip install tensorflow

RUN mkdir /root/trainer
COPY . /root/trainer/

RUN which python3
