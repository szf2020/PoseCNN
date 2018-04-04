FROM gcr.io/tensorflow/tensorflow:1.4.0-gpu

WORKDIR /notebooks

RUN apt-get update && \
  apt-get install -y \
  git \
  libopencv-dev \
  wget \
  libnlopt-dev \
  cmake \
  libglew-dev \
  libgtest-dev \
  libsuitesparse-dev \
  libflann-dev \
  libboost-all-dev \
  libvtk5-dev \
  libvtk5.10-qt4 \
  python-vtk \
  libvtk-java \
  libassimp-dev \
  vim \
  python-opencv \
  python-yaml \
  python-tk && \
  rm -rf /var/lib/apt/lists/*
  
RUN pip install easydict \
  transforms3d \
  Cython
  
ARG NUM_JOBS=4

# install cuda toolkit
#RUN wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_9.0.176-1_amd64.deb && \
#  dpkg -i cuda-repo-ubuntu1604_9.0.176-1_amd64.deb && \
#  apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub && \
#  apt-get update && \
#  apt-get install cuda-9-0

# install bezel (build tool for tensorflow)
RUN wget https://github.com/bazelbuild/bazel/releases/download/0.8.1/bazel-0.8.1-installer-linux-x86_64.sh && \
  chmod +x bazel-0.8.1-installer-linux-x86_64.sh && \
  ./bazel-0.8.1-installer-linux-x86_64.sh --user && \
  echo $'source /root/.bazel/bin/bazel-complete.bash \nexport PATH="$PATH:/root/bin" \n' >> ~/.bashrc && \
  /bin/bash -c "source ~/.bashrc" && \
  rm bazel-0.8.1-installer-linux-x86_64.sh
  
ENV PATH="/root/bin:${PATH}"

# install tensorflow from source
RUN git clone https://github.com/tensorflow/tensorflow && \
  cd tensorflow && \
  git checkout r1.4
  
# uninstall tensorflow comes with the base image
RUN printf "y\n" | pip uninstall http://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-0.0.0-cp27-none-linux_x86_64.whl

# set tensorflow config
# Set environment variables for configure.
# taken from https://github.com/gunan/tensorflow-docker/blob/master/gpu-devel/Dockerfile.ubuntu
#ENV PYTHON_BIN_PATH=python${PY_VERSION_SUFFIX} \
#    LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64:${LD_LIBRARY_PATH} \
#    TF_NEED_CUDA=1 \
#    TF_CUDA_VERSION=${CUDA_VERSION} \
#    TF_CUDNN_VERSION=${CUDNN_VERSION} \
#    TF_CUDA_COMPUTE_CAPABILITIES=3.0,3.5,5.2,6.0,6.1,7.0

RUN cd tensorflow && \
  printf "\n\n\nn\nn\nn\nn\nn\nn\nn\ny\n\n\n\n\n5.2\nn\n" | bash configure && \
  ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1 && \
  export LD_LIBRARY_PATH=/usr/local/cuda/lib64/stubs:/usr/local/cuda/lib64:${LD_LIBRARY_PATH} && \
  ls -ahl && \
  bazel build --config=opt --config=cuda //tensorflow/tools/pip_package:build_pip_package
  
RUN cd tensorflow && \
  bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg --gpu && \
  pip install /tmp/tensorflow_pkg/tensorflow_gpu-1.4.1-cp27-cp27mu-linux_x86_64.whl
  
## install PoseCNN dependencies  
RUN git clone https://github.com/stevenlovegrove/Pangolin.git && \
  cd Pangolin && \
  mkdir build && \
  cd build && \
  cmake .. && \
  cmake --build . && \
  make install
  
RUN wget http://bitbucket.org/eigen/eigen/get/3.3.4.tar.gz && \
  tar xzf 3.3.4.tar.gz && \
  rm 3.3.4.tar.gz && \
  cd eigen-eigen-5a0156e40feb && \
  mkdir build && \
  cd build && \
  cmake .. && \
  make && \
  make install
  
# get eigen that is compatible with cuda9.1
# RUN git clone https://github.com/eigenteam/eigen-git-mirror.git && \
#  cd eigen && \
#  git checkout c6cf6bc5bc4c8c305d5b367212676c1177775972 && \
#  mkdir build && \
#  cd build && \
#  make -j 4 && \
#  make install
  
RUN git clone https://github.com/strasdat/Sophus.git && \
  cd Sophus && \
  mkdir build && \
  cd build && \
  cmake .. && \
  make -j 4 && \
  make install
  
RUN git clone https://github.com/jlblancoc/nanoflann.git && \
  cd nanoflann && \
  mkdir build && \
  cd build && \
  cmake .. && \
  make -j 4 && \
  make test && \
  make install
  
RUN wget https://github.com/PointCloudLibrary/pcl/archive/pcl-1.8.1.tar.gz && \
  tar xzf pcl-1.8.1.tar.gz && \
  rm pcl-1.8.1.tar.gz && \
  cd pcl-pcl-1.8.1/ && \
  mkdir build && \
  cd build && \
  cmake .. && \
  make -j 4 && \
  make install
  
# install PoseCNN
# RUN git clone https://github.com/yuxng/PoseCNN.git
#  cd PoseCNN/lib && \
#  sh make.sh && \
#  #cd /usr/local/cuda-9.1/include && \
#  #ln -s crt/math_functions.hpp math_functions.hpp
#  cd /notebooks/PoseCNN/lib && \
#  python setup.py build_ext --inplace
  
# test PoseCNN


CMD ["/run_jupyter.sh", "--allow-root"]
  
# nvidia-docker hooks
LABEL com.nvidia.volumes.needed="nvidia_driver"
#ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64:${LD_LIBRARY_PATH}
#  
## note: ./make.sh, opencv; synthesize: pcl, build kinect_fusion
