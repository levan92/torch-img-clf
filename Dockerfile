FROM nvcr.io/nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04

ENV cwd="/home/"
WORKDIR $cwd

RUN apt-get -y update && apt-get install -y \
    software-properties-common \
    build-essential \
    checkinstall \
    cmake \
    pkg-config \
    yasm \
    git \
    vim \
    curl \
    wget \
    gfortran \
    sudo \
    apt-transport-https \
    libcanberra-gtk-module \
    libcanberra-gtk3-module \
    dbus-x11 \
    vlc \
    iputils-ping \
    python3-dev \
    python3-pip

# some image/media dependencies
RUN apt-get -y update && apt-get install -y \
    libjpeg8-dev \
    libpng-dev \
    libtiff5-dev \
    libtiff-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libdc1394-22-dev \
    libxine2-dev

# dependencies for FFMPEG build
RUN apt-get -y update && apt-get install -y libchromaprint1 libchromaprint-dev frei0r-plugins-dev gnutls-bin ladspa-sdk libavc1394-0 libavc1394-dev libiec61883-0 libiec61883-dev libass-dev libbluray-dev libbs2b-dev libcaca-dev libgme-dev libgsm1-dev libmysofa-dev libopenmpt-dev libopus-dev libpulse-dev librsvg2-dev librubberband-dev libshine-dev libsnappy-dev libsoxr-dev libspeex-dev libtwolame-dev libvpx-dev libwavpack-dev libwebp-dev libx265-dev libx264-dev libzmq3-dev libzvbi-dev libopenal-dev libomxil-bellagio-dev libcdio-dev libcdio-paranoia-dev libsdl2-dev libmp3lame-dev libssh-dev libtheora-dev libxvidcore-dev

RUN DEBIAN_FRONTEND=noninteractive apt-get install -y tzdata python3-tk
ENV TZ=Asia/Singapore
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get clean && rm -rf /tmp/* /var/tmp/* /var/lib/apt/lists/* && apt-get -y autoremove

### APT END ###

RUN pip3 install --no-cache-dir --upgrade pip 

RUN pip3 install --no-cache-dir \
    setuptools==41.0.0 \
    protobuf==3.13.0 \
    numpy==1.15.4 \
    cryptography==3.2

RUN pip3 install --no-cache-dir --ignore-installed pyxdg==0.26

RUN pip3 install --no-cache-dir jupyter==1.0.0
RUN echo 'alias jup="jupyter notebook --allow-root --no-browser"' >> ~/.bashrc

RUN pip3 install --no-cache-dir \
    GPUtil==1.4.0 \
    tqdm==4.50.0 \
    requests==2.24.0 \
    python-dotenv==0.14.0 \
    pyyaml==5.4.1

RUN pip3 install --no-cache-dir  \
    scipy==1.0.1 \
    matplotlib==3.3.2 \
    Pillow==7.1.2 \
    scikit-image==0.17.2 \
    scikit-learn==0.23.2 \
    pandas==1.1.2 \
    seaborn==0.11.0 \
    tables==3.6.1 \
    numba==0.51.2

RUN pip3 install --no-cache-dir opencv-python==4.4.0.44

RUN pip3 install --no-cache-dir torch==1.4.0+cu100 torchvision==0.5.0+cu100 -f https://download.pytorch.org/whl/torch_stable.html

RUN pip3 install --no-cache-dir tensorboard==2.4.1

# DETECTRON2 DEPENDENCY: PYCOCOTOOLS 
RUN pip3 install --no-cache-dir cython==0.29.21
RUN git clone https://github.com/pdollar/coco
RUN cd coco/PythonAPI \
    && python3 setup.py build_ext install \
    && cd ../.. \
    && rm -r coco