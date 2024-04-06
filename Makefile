mount_abs_dir := $(shell cd ./singularity/mount; pwd)
n_processes := $(shell nproc)

build:
	catkin config --extend /opt/ros/${ROS_DISTRO}
	catkin build

execute-sys:
	./scripts/tmux/start_system.sh

execute-ratslam:
	./scripts/tmux/start_system_ratslam.sh

execute-robotarium:
	./scripts/tmux/start_system_robotarium.sh

install-deps:
	bash -c "./singularity/install/install_tmux.sh"
	bash -c "./singularity/install/install_singularity.sh"

setup-tmux:
	cp -i ./singularity/mount/dottmux.conf ${HOME}/.tmux.conf
	echo 'source ${mount_abs_dir}/addons.sh' >> ${HOME}/.bashrc