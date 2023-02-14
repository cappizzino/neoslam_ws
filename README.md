# NeoSLAM
NeoSLAM is an algorithm based on neocortex models.

## Introduction
This repository holds the version information of all the git repositories of every package that makes up the **neoslam_ws**.

This workspace has a set of bash scripts that will aid, not only, the deployment of the system, but also the inspection of such. The aforementioned scripts require TMUX, a terminal multiplexer.

A Makefile was also included for convenience. **This is the recommended way to start the system.**

**OBS:** With this methodology, we no longer have to run the entire system through a single launch file, which has proven to be impractical for any necessary debugging process. It is now possible to run every single sub-system into their own separate tab within one TMUX session.


## Requirements

**In case you do not wish to install the system natively**, a set of [Singularity recipes](singularity/recipes/) were implemented to contain the system's installation. In such case, please refer to the following [README](singularity/README.md) after completing the following preparation steps:
1. The repository uses a SSH key to allow updating the workspace from within the container. To use it, execute this first:
    ```bash
    chmod 400 singularity/mount/ssh/id_rsa
    ```
2. Install Singularity. This can be achieved through the [install_singularity.sh](scripts/shell/install_singularity.sh) script.

**Make, TMUX, wstool, catkin_tools:**
```bash
sudo apt install make                       # *if not already installed in your system*
sudo apt install tmux xclip                 # terminal multiplexer (alternative to GNU screen)
sudo apt install python-wstool              # workspace version control tool
sudo apt install python-catkin-tools        # catkin_tools to build the workspace
```

**Other dependencies:**
```bash
make install-deps                           # this will take care of source / binary installations in your system. For more information check the shell scripts.
```


## Installation
To install the workspace,
```bash
# system packages
make install-sys

# OR

# simulation packages
make install-sim
```


## Launch the system
To launch the simulation,
```bash
make execute-sim
```
or to deploy the real system,
```bash
make execute-sys
```


## Other Makefile options

* ```make build``` - configures and builds the workspace.
* ```make setup-tmux``` - configures a user-friendly tmux configuration (optional).
