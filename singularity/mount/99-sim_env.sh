#!/bin/bash

SIM_LOCATION=/opt/neoslam

# link bash and zsh rc files
[ ! -e ~/.bashrc ] &&  ln -s $SIM_LOCATION/host/singularity_bashrc.sh ~/.bashrc
[ ! -e ~/.zshrc ] && ln -s $SIM_LOCATION/host/singularity_zshrc.sh ~/.zshrc
[ ! -e ~/.profile ] && ln -s $SIM_LOCATION/host/singularity_profile.sh ~/.profile

# link .tmux.conf
[ ! -e ~/.tmux.conf ] && [ -e $SIM_LOCATION/host/dottmux.conf ] && ln -s $SIM_LOCATION/host/dottmux.conf ~/.tmux.conf

# link .ssh key-pair
[ -e $SIM_LOCATION/host/ssh ] && [ ! -e ~/.ssh ] && ln -s $SIM_LOCATION/host/ssh ~/.ssh

touch ~/.sudo_as_admin_successful

export PS1="[Neoslam] ${debian_chroot:+($debian_chroot)}\[\033[01;32m\]\u@\h\[\033[00m\]:\[\033[01;34m\]\w\[\033[00m\]\$ "
