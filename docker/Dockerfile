FROM ubuntu:18.04

ARG DEBIAN_FRONTEND=noninteractive

# GCC dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libgmp-dev \ 
    libmpfr-dev \
    libmpc-dev \ 
    bash \
    git \
    flex \
    gcc-multilib \
    && rm -rf /var/lib/apt/lists/*

# CMAKE dependencies
RUN apt-get update && apt-get install -y \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# OpenCV Library
RUN apt-get update && apt-get install -y --no-install-recommends\
    libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

# Install CMAKE
RUN git clone https://github.com/Kitware/CMake.git cmake_repo && \
    cd cmake_repo && \
    git checkout v3.17.3 && \
    ./bootstrap && \
    make && \
    make install && \
    cd .. && \
    rm cmake_repo -r

# Install GCC
RUN git clone git://gcc.gnu.org/git/gcc.git gcc_repo && \
    cd gcc_repo && \
    git checkout releases/gcc-10.1.0 && \
    ./configure --enable-languages=c,c++ --disable-multilib && \
    make && \
    make install && \
    cd .. && \
    rm gcc_repo -r

# Set environment variables
ENV CC=/usr/local/bin/gcc
ENV export CXX=/usr/local/bin/g++

CMD cd /home/out && cmake /home/source/. && make    
