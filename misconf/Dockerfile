FROM nvidia/cuda:10.2-base-ubuntu18.04

# 安装必要的工具
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
 && rm -rf /var/lib/apt/lists/*

# 创建工作目录
RUN mkdir /Pytorch_Template
WORKDIR /Pytorch_Template

# 设置非root用户
RUN adduser --disabled-password --gecos '' --shell /bin/bash user \
 && chown -R user:user /Pytorch_Template
RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user
USER user

# All users can use /home/user as their home directory
ENV HOME=/home/user
RUN chmod 777 /home/user

# Install Miniconda and Python 3.8
ENV CONDA_AUTO_UPDATE_CONDA=false
ENV PATH=/home/user/miniconda/bin:$PATH
RUN curl -sLo ~/miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-py38_4.8.2-Linux-x86_64.sh \
 && chmod +x ~/miniconda.sh \
 && ~/miniconda.sh -b -p ~/miniconda \
 && rm ~/miniconda.sh \
 && conda install -y python==3.8.1 \
 && conda clean -ya

# CUDA 10.2-specific steps
RUN conda install -y -c pytorch \
    cudatoolkit=10.2 \
    "pytorch=1.5.0=py3.8_cuda10.2.89_cudnn7.6.5_0" \
    "torchvision=0.6.0=py38_cu102" \
 && conda clean -ya

# 安装依赖包
ADD ../requirements.txt /Pytorch_Template/
RUN pip install -r /requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/ --default-timeout=300

# 复制当前文件夹必要文件到docker环境
Add ../checkpoints /Pytorch_Template/checkpoints
Add ../data /Pytorch_Template/data
Add ../logfile /Pytorch_Template/logfile
Add ../misconf /Pytorch_Template/misconf
Add ../models /Pytorch_Template/models
Add ../result /Pytorch_Template/result
Add ../utils /Pytorch_Template/utils
Add ../config.py /Pytorch_Template/
Add ../main.py /Pytorch_Template/

# Set the default command to python3
CMD ["cd /Pytorch_Template/misconf","chmod -x run.sh","./run.sh train"]