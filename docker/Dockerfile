FROM ros:melodic-ros-base-bionic

RUN apt-get -y update && apt-get install -y \
    make \
    gcc \
    git \
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

RUN mkdir -p /workspace
WORKDIR /workspace
