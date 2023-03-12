export SHELL=$( which zsh )

PROMPT='[Neoslam]%1~ %# '

source /opt/ros/$ROS_DISTRO/setup.zsh

# source the user_workspace, if it exists
[ -e ~/neoslam_ws/devel/setup.zsh ] && source ~/neoslam_ws/devel/setup.zsh

[ -z "$ROS_PORT" ] && export ROS_PORT=11311
[ -z "$ROS_MASTER_URI" ] && export ROS_MASTER_URI=http://localhost:$ROS_PORT

# if host pc is not Ubuntu 20.04
OS_INFO=$(cat /proc/version)
if ! ([[ "$INFO_OS" == *"Ubuntu"* ]] && [[ "$INFO_OS" == *"20.04"* ]]); then
  export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
  source /usr/share/gazebo/setup.bash
fi

# # source uav_core from within the container
# source /opt/mrs/mrs_workspace/src/uav_core/miscellaneous/shell_additions/shell_additions.sh

# source the linux setup from within
if [ -e /opt/neoslam/host/addons.sh ]; then
  source /opt/neoslam/host/addons.sh
fi
