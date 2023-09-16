#######################################################################################
#######################################################################################
##################### SYSTEM'S SOFTWARE SPECIFIC CONFIGURATION ########################
#######################################################################################
#######################################################################################
# * use this file to source any additional configuration before launch (e.g. SERVER_IP)

# Path CNN model
export TORCH_HOME=$ROS_DATA_PATH/model/checkpoints

# Enable Dimension Reduction creation
export DIMENSION_REDUCTION=0
export MATRIX_HOME=$ROS_DATA_PATH/matrix

# Image Saver
export SYS_IMAGE_ENABLED=0  # enable / disable image recording
export SYS_IMAGE_ARGS=_save_all_image:=false
export SYS_IMAGE_PATH=_filename_format:="$HOME/bag_files/neoslam/latest/images/image%04d.%s"

# Image republished
export SYS_IMAGE_TOPIC_REPUBLISED_ENABLED=1
export SYS_IMAGE_TOPIC=/image_raw

# Image Viewer
export SYS_RQT_VIEWER_ENABLED=0

# ROS bag
export SYS_ROSBAG_ENABLED=1     # enable / disable bag recording (be careful to NOT run long term experiments without bags!)
export SYS_ROSBAG_SIZE='1024'   # max size before splitting in Mb (i.e. 0 = infinite, 1024 = 1024Mb = 1Gb)
export SYS_ROSBAG_DURATION='8h'
export SYS_ROSBAG_PATH="$HOME/bag_files/neoslam/latest/"

export SYS_ROSBAG_ARGS="
    --regex
    --split
    --size=$SYS_ROSBAG_SIZE
    --duration=$SYS_ROSBAG_DURATION
    --output-prefix=$SYS_ROSBAG_PATH
"

export SYS_ROSBAG_TOPICS="
    /ExperienceMap/Map 
    /ExperienceMap/RobotPose 
    /LocalView/Template 
    /PoseCell/TopologicalAction
    /odom
    /feats_cnn
    /feats_htm 
    /feats_lsbh
    /image_raw
    /info
    /overhead/pose
"

# ROS specific configs
export ROSCONSOLE_CONFIG_FILE=$HOME/neoslam_ws/rosconsole.conf
export ROSCONSOLE_FORMAT='[${severity}] [${node}] [${function}] [${line}]: ${message}'
