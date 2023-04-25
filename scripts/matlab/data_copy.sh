#!/bin/bash

DATASET_NAME=${1:-'exp'}
BAG_FOLDER=~/"bag_files"
PROJECT_NAME=neoslam

# get the iterator
ITERATOR_FILE="$BAG_FOLDER/$PROJECT_NAME"/iterator.txt
if [ -e "$ITERATOR_FILE" ]
then
  ITERATOR=`cat "$ITERATOR_FILE"`
  ITERATOR=$(($ITERATOR))
else
  echo "iterator.txt does not exist"
fi

# # location of the running script
# DIR_PATH=$(pwd)/$DATASET_NAME

# Source path
SOURCE_PATH=$BAG_FOLDER/latest/.
echo "$SOURCE_PATH"

# Destination path
mkdir -p ./experiments/$DATASET_NAME
DESTINATION_PATH=$(pwd)/experiments/$DATASET_NAME/
echo "$DESTINATION_PATH"

cp -r $SOURCE_PATH $DESTINATION_PATH
