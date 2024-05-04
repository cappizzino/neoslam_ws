mount_abs_dir := $(shell cd ./singularity/mount; pwd)
n_processes := $(shell nproc)

build:
	catkin config --extend /opt/ros/${ROS_DISTRO}
	catkin build

execute-sys:
	./scripts/tmux/start_system.sh

execute-ratslam-irataus:
	./scripts/tmux/start_system_ratslam_irataus.sh

execute-ratslam-stlucia2007:
	./scripts/tmux/start_system_ratslam_stlucia2007.sh

execute-neoslam-robotarium:
	./scripts/tmux/start_system_neoslam_robotarium.sh

install-deps:
	bash -c "./singularity/install/install_tmux.sh"
	bash -c "./singularity/install/install_singularity.sh"

setup-tmux:
	cp -i ./singularity/mount/dottmux.conf ${HOME}/.tmux.conf
	echo 'source ${mount_abs_dir}/addons.sh' >> ${HOME}/.bashrc