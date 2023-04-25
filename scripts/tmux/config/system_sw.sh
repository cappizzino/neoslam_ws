#######################################################################################
#######################################################################################
##################### SYSTEM'S SOFTWARE SPECIFIC CONFIGURATION ########################
#######################################################################################
#######################################################################################
# * use this file to source any additional configuration before launch (e.g. SERVER_IP)

# Path CNN model
export TORCH_HOME=$HOME/neoslam_ws/singularity/mount/model/checkpoints

# Enable Dimension Reduction creation
export DIMENSION_REDUCTION=0
export MATRIX_HOME=$HOME/neoslam_ws/singularity/mount/data

# Image Topic
export SYS_IMAGE_TOPIC=/stereo_camera/left/image_raw

# Image Saver
export SYS_IMAGE_ENABLED=1  # enable / disable image recording
export SYS_IMAGE_ARGS=_save_all_image:=false
export SYS_IMAGE_PATH=_filename_format:="$HOME/bag_files/neoslam/latest/images/image%04d.%s"

# Image Viewer
export SYS_RQT_VIEWER_ENABLED=1

# ROS bag
export SYS_ROSBAG_ENABLED=1     # enable / disable bag recording (be careful to NOT run long term experiments without bags!)
export SYS_ROSBAG_SIZE='1024'   # max size before splitting in Mb (i.e. 0 = infinite, 1024 = 1024Mb = 1Gb)
export SYS_ROSBAG_DURATION='8h'
export SYS_ROSBAG_PATH="$HOME/bag_files/neoslam/latest/"

# Experiment  
export EXPERIMENT="robotarium" # corridor ; robotarium ; outdoor
case $EXPERIMENT in
  corridor)
    export SYS_ROSBAG_NAME=_2022-04-07-11-08-05_corridor.bag
    ;;
  robotarium)
    export SYS_ROSBAG_NAME=_2022-04-07-14-14-35_robotarium.bag
    ;;
  outdoor)
    export SYS_ROSBAG_NAME=_2022-04-07-11-19-55_outdoor_morning.bag
    ;;
esac

export SYS_CONFIG_RATSLAM="config_husky_hwu_$EXPERIMENT.txt.in"
export SYS_CONFIG_NEOCORTEX="husky_hwu_$EXPERIMENT.yaml"

export SYS_ROSBAG_ARGS="
    --regex
    --split
    --size=$SYS_ROSBAG_SIZE
    --duration=$SYS_ROSBAG_DURATION
    --output-prefix=$SYS_ROSBAG_PATH
"
export SYS_ROSBAG_TOPICS="
    /odometry/filtered
    /husky(.*)
    /feats_cnn
    /feats_htm 
    /feats_lsbh 
    /info
"
# ROS specific configs
export ROSCONSOLE_FORMAT='[${severity}] [${node}] [${function}] [${line}]: ${message}'
