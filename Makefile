mount_abs_dir := $(shell cd ./singularity/mount; pwd)
n_processes := $(shell nproc)

build:
	catkin config --extend /opt/ros/${ROS_DISTRO}
	catkin build

execute-sim:
	./scripts/shell/cdb-start-single-node.sh
	./scripts/tmux/start_simulation.sh

install-sim:
	wstool update -j ${n_processes} -t src/sw_simulation/
	wstool update -j ${n_processes} -t src/sw_system/
	wstool update -j ${n_processes} -t src/sw_monitor/

install-deps:
	bash -c "./singularity/install/install_tmux.sh"
	bash -c "./singularity/install/install_singularity.sh"

setup-tmux:
	cp -i ./singularity/mount/dottmux.conf ${HOME}/.tmux.conf
	echo 'source ${mount_abs_dir}/addons.sh' >> ${HOME}/.bashrc