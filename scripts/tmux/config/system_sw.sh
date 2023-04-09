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

# Image path
export SYS_IMAGE_ENABLED=1  # enable / disable image recording
export SYS_IMAGE_PATH="$HOME/bag_files/neoslam/latest/images/"
export SYS_IMAGE_ARGS="
    --save_all_image=false
    --filename_format=$SYS_IMAGE_PATH/image%04d.%s
"
export SYS_IMAGE_TOPICS=/stereo_camera/left/image_raw

# ROS bag
export SYS_ROSBAG_ENABLED=1     # enable / disable bag recording (be careful to NOT run long term experiments without bags!)
export SYS_ROSBAG_SIZE='0'      # max size before splitting in Mb (i.e. 0 = infinite, 1024 = 1024Mb = 1Gb)
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
    /imu
    /cmd_vel
    /husky(.*)
    /feats_cnn
    /feats_htm 
    /feats_lsbh 
    /info
"
# ROS specific configs
export ROSCONSOLE_FORMAT='[${severity}] [${node}] [${function}] [${line}]: ${message}'
