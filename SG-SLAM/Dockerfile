FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

SHELL ["/bin/bash", "-c"]

ARG CATKIN_JOBS=2
ARG CATKIN_BUILD_PACKAGES=semgraph_slam

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    gnupg2 \
    lsb-release \
    ca-certificates \
    software-properties-common \
    build-essential \
    cmake \
    git \
    python3-pip \
    libeigen3-dev \
    libpcl-dev \
    libgoogle-glog-dev \
    libgflags-dev \
    libatlas-base-dev \
    libsuitesparse-dev \
    libtbb-dev \
    libfmt-dev \
    libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir catkin-tools

RUN curl -sSL "https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc" | apt-key add -
RUN echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -cs) main" > /etc/apt/sources.list.d/ros1-latest.list

RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-noetic-ros-base \
    ros-noetic-roscpp \
    ros-noetic-std-msgs \
    ros-noetic-sensor-msgs \
    ros-noetic-nav-msgs \
    ros-noetic-geometry-msgs \
    ros-noetic-tf \
    ros-noetic-tf2 \
    ros-noetic-tf2-ros \
    ros-noetic-rosbag \
    ros-noetic-pcl-conversions \
    ros-noetic-pcl-ros \
    && rm -rf /var/lib/apt/lists/*

RUN add-apt-repository ppa:borglab/gtsam-release-4.0 && \
    apt-get update && apt-get install -y --no-install-recommends \
    libgtsam-dev \
    libgtsam-unstable-dev \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /opt/catkin_ws/src/SG-SLAM
WORKDIR /opt/catkin_ws/src/SG-SLAM
COPY cpp /opt/catkin_ws/src/SG-SLAM/cpp
COPY ros /opt/catkin_ws/src/SG-SLAM/ros

WORKDIR /opt/catkin_ws
RUN source /opt/ros/noetic/setup.bash && \
    catkin init && \
    catkin config \
      --extend /opt/ros/noetic \
      --merge-devel \
      --cmake-args \
        -DCMAKE_BUILD_TYPE=Release \
        -DUSE_SYSTEM_EIGEN3=ON \
        -DUSE_SYSTEM_TBB=ON \
        -DENABLE_MILESTONE1_BENCHMARKS=OFF && \
    catkin build --no-status -j${CATKIN_JOBS} -p1 ${CATKIN_BUILD_PACKAGES}

RUN echo "source /opt/ros/noetic/setup.bash" >> /root/.bashrc && \
    echo "[ -f /opt/catkin_ws/devel/setup.bash ] && source /opt/catkin_ws/devel/setup.bash" >> /root/.bashrc

CMD ["/bin/bash", "-l"]
