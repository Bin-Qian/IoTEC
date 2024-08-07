FROM osrf/ros:rolling-desktop

# 如果需要，可以复制一些文件到容器中
COPY code  /project/

# 更新软件包列表并安装vim，ping，ifconfig，编译代码
RUN \
    sed -i 's/archive.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list && \
    apt-get update && \
    apt-get install -y vim && \
    apt-get install -y inetutils-ping && \ 
    apt install -y net-tools && \ 
    apt-get install wget && \ 
    wget https://download.pytorch.org/models/resnet50-0676ba61.pth -P /root/.cache/torch/hub/checkpoints/ && \ 
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* 

RUN \
    mv /usr/lib/python3.12/EXTERNALLY-MANAGED /usr/lib/python3.12/EXTERNALLY-MANAGED.bk && \ 
    curl https://bootstrap.pypa.io/get-pip.py --output get-pip.py && \ 
    python3 get-pip.py && \ 
    rm get-pip.py 

RUN \
    pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple torch torchvision 
RUN \
    pip3 install pycocotools 

RUN \ 
    cd /project && \
    . /opt/ros/rolling/setup.sh && \
    colcon build 

# 设置工作目录
WORKDIR /project


