#!/bin/bash

# force run w/ sudo
if [ $EUID != 0 ]; then
    sudo "$0" "$@"
    exit $?
fi

version='v2.11.1'

cd /usr/local

    tmp=$(mktemp -d -p .)

    cd $tmp ; curl -L --fail https://github.com/heroiclabs/nakama/releases/download/${version}/nakama-${version#?}-linux-amd64.tar.gz | 
        tar -xz && sudo cp nakama ../bin/ || 
        (echo "Failed to get Nakama binaries..." && exit 1)

    rm -rf "$tmp"

cd - >> /dev/null

which nakama
