FROM nvidia/cuda@sha256:4df157f2afde1cb6077a191104ab134ed4b2fd62927f27b69d788e8e79a45fa1

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update
RUN apt-get install -y --no-install-recommends apt-utils

RUN apt-get update && apt-get install -y --no-install-recommends \
wget curl xz-utils build-essential zlib1g-dev zip unzip \
libreadline-gplv2-dev libncursesw5-dev libssl-dev libosmesa6-dev \
libsqlite3-dev tk-dev libgdbm-dev libc6-dev libbz2-dev \
libgl1-mesa-glx libgl1-mesa-dev patchelf libglfw3-dev libglew-dev \
libsqlite3-dev libpng-dev libjpeg-dev git-all liblzma-dev tzdata \
software-properties-common net-tools vim xpra xserver-xorg-dev  && \
ln -fs /usr/share/zoneinfo/Asia/Kolkata /etc/localtime && \
dpkg-reconfigure --frontend noninteractive tzdata && \
apt-get clean && \
rm -rf /var/lib/apt/lists/*

RUN wget https://www.python.org/ftp/python/3.6.10/Python-3.6.10.tar.xz && \
tar -xJf Python-3.6.10.tar.xz && cd Python-3.6.10 && \
./configure && make && make install && \
pip3 install --upgrade pip wheel setuptools sphinx_rtd_theme numpydoc  \
ipdb scipy sphinx numpy pandas cffi Cython lockfile glfw imageio

ENV LANG C.UTF-8

RUN mkdir /root/.mujoco && cd /root/.mujoco && \
wget https://www.roboti.us/download/mjpro150_linux.zip && \
unzip mjpro150_linux.zip && \
wget https://www.roboti.us/file/mjkey.txt

ENV LD_LIBRARY_PATH /root/.mujoco/mjpro150/bin:${LD_LIBRARY_PATH}
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib64:${LD_LIBRARY_PATH}

COPY vendor/Xdummy /usr/local/bin/Xdummy
RUN chmod +x /usr/local/bin/Xdummy
COPY vendor/10_nvidia.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json


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
WORKDIR /root/trainer

ENTRYPOINT ["python3", "train.py"]
