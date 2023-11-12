#!/bin/bash

source $HOME/.bashrc

# location of the running script
DIR_PATH=$(cd $(dirname $0); pwd)
ROS_BAG_PATH="$DIR_PATH/../../ros_bags"
ROS_CONFIG_PATH="$DIR_PATH/../../ros_config"
ROS_DATA_PATH="$DIR_PATH/../../ros_data"
ROS_LAUNCH_PATH="$DIR_PATH/../../ros_launch"
ROS_RVIZ_PATH="$DIR_PATH/../../ros_rviz"
ROS_MAP_PATH="$DIR_PATH/../../maps"

# check if workspace was built
[[ -f $DIR_PATH/../../devel/setup.bash ]] ||
{ echo "Build the workspace first!"; exit 1; }

# location for storing the bag files
# * do not change unless you know what you are doing
MAIN_DIR=~/"bag_files"

# the project name
# * is used to define folder name in ~/$MAIN_DIR
PROJECT_NAME=neoslam

# the name of the TMUX session
# * can be used for attaching as 'tmux a -t <session name>'
SESSION_NAME="neoslam"

# the IP of the server
# * using this, depending on your use case, might be preferable over 
# * 'localhost', since this information is passed to outside nodes 
# * (e.g. outside node accessing hosted database)
SESSION_IP=$(hostname -I | awk '{print $1}')

# following commands will be executed first in each window
# * do NOT put ; at the end
pre_input="mkdir -p $MAIN_DIR/$PROJECT_NAME; \
export DIR_PATH=$DIR_PATH; \
export ROS_BAG_PATH=$ROS_BAG_PATH; \
export ROS_CONFIG_PATH=$ROS_CONFIG_PATH; \
export ROS_DATA_PATH=$ROS_DATA_PATH; \
export ROS_LAUNCH_PATH=$ROS_LAUNCH_PATH; \
export ROS_RVIZ_PATH=$ROS_RVIZ_PATH; \
export ROS_MAP_PATH=$ROS_MAP_PATH; \
source $DIR_PATH/config/system_hw.sh; \
source $DIR_PATH/config/system_sw.sh; \
source $DIR_PATH/../../singularity/mount/addons.sh; \
source $DIR_PATH/../../devel/setup.bash"

# define commands
# 'name' 'command'
# * DO NOT PUT SPACES IN THE NAMES
# * "new line" after the command    => the command will be called after start
# * NO "new line" after the command => the command will wait for user's <enter>
export SLAM_MODEL="ratslam" # ratslam ; neoslam
export DATASET="robotarium" # corridor ; robotarium ; outdoor ; irataus

# Configuration files
# export SYS_CONFIG_RATSLAM="ratslam_$DATASET.txt"
# export SYS_CONFIG_NEOCORTEX="neocortex_$DATASET.yaml"
export SYS_CONFIG_NEOCORTEX="neocortex.yaml"

input=(
  'NeoSlam' 'waitForRos; roslaunch $ROS_LAUNCH_PATH/neoslam.launch
'
)

input+=(
  'rosbag' 'waitForRos; [ $SYS_ROSBAG_ENABLED -eq 1 ] && rosbag record $SYS_ROSBAG_ARGS $SYS_ROSBAG_TOPICS || exit
'
  'Saver' 'waitForRos; [ $SYS_IMAGE_ENABLED -eq 1 ] && rosrun image_view image_saver image:=$SYS_IMAGE_TOPIC $SYS_IMAGE_ARGS $SYS_IMAGE_PATH __name:=image_saver || exit
'
  'Viewer' 'waitForRos; [ $SYS_RQT_VIEWER_ENABLED -eq 1 ] && rosrun rqt_image_view rqt_image_view image:=$SYS_IMAGE_TOPIC || exit
'
  'roscore' 'checkRos || roscore && exit
'
)

# the name of the window to focus after start
init_window="NeoSlam"

# automatically attach to the new session?
# {true, false}, default true
attach="true"

###########################
### DO NOT MODIFY BELOW ###
###########################

# prefere the user-compiled tmux
if [ -f /usr/local/bin/tmux ]; then
  export TMUX_BIN=/usr/local/bin/tmux
else
  export TMUX_BIN=/usr/bin/tmux
fi

# find the session
FOUND=$( $TMUX_BIN ls | grep $SESSION_NAME )

if [ $? == "0" ]; then

  echo "The session already exists"
  exit
fi

# Absolute path to this script. /home/user/bin/foo.sh
SCRIPT=$(readlink -f $0)
# Absolute path this script is in. /home/user/bin
SCRIPTPATH=`dirname $SCRIPT`

if [ -z ${TMUX} ];
then
  TMUX= $TMUX_BIN new-session -s "$SESSION_NAME" -d
  echo "Starting new session."
else
  echo "Already in tmux, leave it first."
  exit
fi

# get the iterator
ITERATOR_FILE="$MAIN_DIR/$PROJECT_NAME"/iterator.txt
if [ -e "$ITERATOR_FILE" ]
then
  ITERATOR=`cat "$ITERATOR_FILE"`
  ITERATOR=$(($ITERATOR+1))
else
  echo "iterator.txt does not exist, creating it"
  mkdir -p "$MAIN_DIR/$PROJECT_NAME"
  touch "$ITERATOR_FILE"
  ITERATOR="1"
fi
echo "$ITERATOR" > "$ITERATOR_FILE"

# create file for logging terminals' output
LOG_DIR="$MAIN_DIR/$PROJECT_NAME/"
SUFFIX=$(date +"%Y_%m_%d_%H_%M_%S")
SUBLOG_DIR="$LOG_DIR/"$ITERATOR"_"$SUFFIX""
TMUX_DIR="$SUBLOG_DIR/tmux"
IMAGE_DIR="$SUBLOG_DIR/images"
mkdir -p "$SUBLOG_DIR"
mkdir -p "$TMUX_DIR"
mkdir -p "$IMAGE_DIR"

# link the "latest" folder to the recently created one
rm "$LOG_DIR/latest" > /dev/null 2>&1
rm "$MAIN_DIR/latest" > /dev/null 2>&1
ln -sf "$SUBLOG_DIR" "$LOG_DIR/latest"
ln -sf "$SUBLOG_DIR" "$MAIN_DIR/latest"

# create arrays of names and commands
for ((i=0; i < ${#input[*]}; i++));
do
  ((i%2==0)) && names[$i/2]="${input[$i]}"
  ((i%2==1)) && cmds[$i/2]="${input[$i]}"
done

# run tmux windows
for ((i=0; i < ${#names[*]}; i++));
do
  $TMUX_BIN new-window -t $SESSION_NAME:$(($i+1)) -n "${names[$i]}"
done

sleep 3

# start loggers
for ((i=0; i < ${#names[*]}; i++));
do
  $TMUX_BIN pipe-pane -t $SESSION_NAME:$(($i+1)) -o "ts | cat >> $TMUX_DIR/$(($i+1))_${names[$i]}.log"
done

# send commands
for ((i=0; i < ${#cmds[*]}; i++));
do
  $TMUX_BIN send-keys -t $SESSION_NAME:$(($i+1)) "cd $SCRIPTPATH;
${pre_input};
${cmds[$i]}"
done

# identify the index of the init window
init_index=0
for ((i=0; i < ((${#names[*]})); i++));
do
  if [ ${names[$i]} == "$init_window" ]; then
    init_index=$(expr $i + 1)
  fi
done

$TMUX_BIN select-window -t $SESSION_NAME:$init_index

if [[ "$attach" == "true" ]]; then
  $TMUX_BIN -2 attach-session -t $SESSION_NAME
else
  echo "The session was started"
  echo "You can later attach by calling:"
  echo "  tmux a -t $SESSION_NAME"
fi
