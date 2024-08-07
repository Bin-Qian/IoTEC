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
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* 

RUN \ 
    cd /project && \
    . /opt/ros/rolling/setup.sh && \
    colcon build 

# 设置工作目录
WORKDIR /project


