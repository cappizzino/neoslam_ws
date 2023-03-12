#!/bin/bash

MAP_NAME=${1:-'map'}
MAP_TOPIC=${2:-'/map'}
SAVE_DIRECTORY="$( cd $( dirname $0 )/../../maps; pwd )"

cd "$SAVE_DIRECTORY" && rosrun map_server map_saver -f "$MAP_NAME" map:="$MAP_TOPIC"