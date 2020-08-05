FROM ubuntu:18.04 AS builder

ARG DEBIAN_FRONTEND=noninteractive

# Build image dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libgmp-dev \ 
    libmpfr-dev \
    libmpc-dev \ 
    bash \
    git \
    flex \
    gcc-multilib \
    libssl-dev \
    checkinstall \
    && rm -rf /var/lib/apt/lists/*

# Build CMAKE
RUN git clone https://github.com/Kitware/CMake.git cmake_repo && \
    cd cmake_repo && \
    git checkout v3.17.3 && \
    mkdir out && \
    ./bootstrap && \
    make && \
    checkinstall -D --pkgname=cmake --pkgversion=3.17.3


# Build GCC
RUN git clone git://gcc.gnu.org/git/gcc.git gcc_repo && \
    cd gcc_repo && \
    git checkout releases/gcc-10.1.0 && \
    mkdir out && \
    ./configure --enable-languages=c,c++ --disable-multilib && \
    make && \
    checkinstall -D --pkgname=gcc --pkgversion=10.1.0


# Target image
FROM ubuntu:18.04

ARG DEBIAN_FRONTEND=noninteractive

# OpenCV Library
RUN apt-get update && apt-get install -y --no-install-recommends\
    libopencv-dev \
    libmpc-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy packages from builder image
COPY --from=builder /cmake_repo/*.deb .
COPY --from=builder /gcc_repo/*.deb .

# Install CMAKE and GCC
RUN apt install ./*.deb && rm *.deb

# Set environment variables
ENV CC=/usr/local/bin/gcc
ENV export CXX=/usr/local/bin/g++

CMD cd /home/out && cmake /home/source/. && make
