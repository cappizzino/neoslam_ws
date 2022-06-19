FROM ros:melodic-ros-base-bionic

RUN apt-get -y update && apt-get install -y \
    make \
    gcc \
    git \
    tmux \
    vim

RUN apt-get update && apt-get install -y \
    python-pip

RUN apt-get update && apt-get install -y \ 
    lsb-release && apt-get clean all

RUN pip install setuptools==25.2.0
RUN pip install setuptools-scm==1.15.0rc1

COPY requirements.txt /tmp/
RUN pip install -r /tmp/requirements.txt

RUN apt-get install -y \
    ros-$(rosversion -d)-cv-bridge
    
WORKDIR /home/neoslam_ws/src

RUN git clone https://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@github.com/cappizzino/ROS_programs.git

WORKDIR /home/neoslam_ws

RUN /bin/bash -c '. /opt/ros/melodic/setup.bash; cd /home/neoslam_ws; catkin_make'
RUN echo "source /home/neoslam_ws/devel/setup.bash" >> ~/.bashrc



